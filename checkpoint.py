import base64,json,os,httpx,sys,asyncio,threading,types,time

REPO=os.getenv('GITHUB_REPOSITORY','lumina-dev-ace/Riemann-Search-Project')
BRANCH=os.getenv('CHECKPOINT_BRANCH','riemann-checkpoint')
PATH=os.getenv('CHECKPOINT_FILE','checkpoint.json')
TOKEN=os.getenv('CHECKPOINT_TOKEN','')
ROLE=os.getenv('RH_ROLE','web')
URL=f'https://api.github.com/repos/{REPO}/contents/{PATH}'

async def load_checkpoint():
    if not TOKEN:return None
    headers={'Authorization':f'Bearer {TOKEN}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
    async with httpx.AsyncClient(timeout=20) as c:
        r=await c.get(URL,headers=headers,params={'ref':BRANCH})
        if r.status_code==404:return None
        r.raise_for_status()
        return json.loads(base64.b64decode(r.json()['content']).decode())

async def save_checkpoint(data):
    if not TOKEN:return False
    headers={'Authorization':f'Bearer {TOKEN}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
    async with httpx.AsyncClient(timeout=20) as c:
        r=await c.get(URL,headers=headers,params={'ref':BRANCH})
        sha=r.json().get('sha') if r.status_code==200 else None
        body={'message':f'Update search checkpoint n={data.get("next_n",1)}','content':base64.b64encode(json.dumps(data,indent=2).encode()).decode(),'branch':BRANCH}
        if sha:body['sha']=sha
        r=await c.put(URL,headers=headers,json=body);r.raise_for_status();return True

# The web service is the dashboard/control plane. The dedicated Render
# Background Worker owns the numerical computation.
def _install_patches():
    for _ in range(240):
        mod=sys.modules.get('app')
        obj=getattr(mod,'search',None) if mod is not None else None
        app_obj=getattr(mod,'app',None) if mod is not None else None
        if mod is not None and obj is not None and app_obj is not None:
            if not getattr(obj,'_checkpoint_patches_installed',False):
                original_checkpoint=mod.checkpoint
                original_start=obj.start

                async def safe_checkpoint(paused=False, extra=None):
                    # Never let a worker's periodic save accidentally unpause
                    # a run after the dashboard has pressed Stop.
                    if not paused:
                        try:
                            remote=await load_checkpoint()
                            if remote and remote.get('paused') is True:
                                paused=True
                        except Exception:
                            pass
                    return await original_checkpoint(paused, extra)

                async def request_start():
                    try:
                        cp=await load_checkpoint()
                        if cp and cp.get('candidate_found'):
                            return
                        n=max(mod.START_N,int(cp.get('next_n',mod.START_N)) if cp else mod.START_N)
                        payload={'next_n':n,'zeros_checked':int(cp.get('zeros_checked',0)) if cp else 0,'updated_at_utc':mod.now(),'paused':False,'search_start_n':mod.START_N,'dashboard_start_requested':True}
                        await save_checkpoint(payload)
                    except Exception as ex:
                        try: obj.error=f'Start request failed: {type(ex).__name__}: {ex}'
                        except Exception: pass

                async def persistent_start(self):
                    if ROLE == 'web':
                        if getattr(self,'running',False):
                            return
                        cp=await load_checkpoint()
                        if cp and cp.get('candidate_found'):
                            self.stopped=True
                            self.reason='A candidate is already recorded; start is disabled for this run.'
                            return
                        n=max(mod.START_N,int(cp.get('next_n',mod.START_N)) if cp else mod.START_N)
                        mod.setstate('next_n',n)
                        mod.setstate('current_n',n)
                        mod.setstate('phase','start requested — waiting for background worker')
                        self.stopped=False
                        self.reason='Background worker start requested'
                        self.error=''
                        self.running=True
                        asyncio.create_task(request_start())
                        return
                    await original_start()

                async def safe_stop(self):
                    if not self.running:
                        self.stopped=True
                        self.reason='Already stopped'
                        try: mod.setstate('phase','stopped')
                        except Exception: pass
                        return
                    self.reason='Stopping safely after the current numerical operation...'
                    try: mod.setstate('phase','stopping')
                    except Exception: pass
                    self.running=False
                    self.stopped=True
                    async def save_stop():
                        try:
                            cp=await load_checkpoint()
                            n=max(mod.START_N,int(cp.get('next_n',mod.START_N)) if cp else int(mod.state('next_n')))
                            payload={'next_n':n,'zeros_checked':int(cp.get('zeros_checked',0)) if cp else mod.count('zeros'),'updated_at_utc':mod.now(),'paused':True,'search_start_n':mod.START_N,'dashboard_stop_requested':True}
                            await save_checkpoint(payload)
                        except Exception as ex:
                            try: self.error=f'Stop checkpoint failed: {type(ex).__name__}: {ex}'
                            except Exception: pass
                    asyncio.create_task(save_stop())
                    try: mod.setstate('phase','stopped')
                    except Exception: pass

                mod.checkpoint=safe_checkpoint
                obj.start=types.MethodType(persistent_start,obj)
                obj.stop=types.MethodType(safe_stop,obj)
                obj._checkpoint_patches_installed=True

            # Do not auto-resume numerical work inside the web service.
            if not getattr(app_obj,'_auto_resume_handler_installed',False):
                app_obj._auto_resume_handler_installed=True
            return
        time.sleep(0.05)

threading.Thread(target=_install_patches,daemon=True).start()
