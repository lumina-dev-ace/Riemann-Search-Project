import base64,json,os,httpx,sys,asyncio,threading,types,time

REPO=os.getenv('GITHUB_REPOSITORY','lumina-dev-ace/Riemann-Search-Project')
BRANCH=os.getenv('CHECKPOINT_BRANCH','riemann-checkpoint')
PATH=os.getenv('CHECKPOINT_FILE','checkpoint.json')
TOKEN=os.getenv('CHECKPOINT_TOKEN','')
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

# Install small runtime patches after app.py has created the Search instance.
# Start must return immediately to the browser; checkpointing is fire-and-forget.
# Stop changes the running flag immediately and lets the current native FLINT
# operation finish at a safe boundary.
def _install_patches():
    for _ in range(240):
        mod=sys.modules.get('app')
        obj=getattr(mod,'search',None) if mod is not None else None
        app_obj=getattr(mod,'app',None) if mod is not None else None
        if mod is not None and obj is not None and app_obj is not None:
            if not getattr(obj,'_checkpoint_patches_installed',False):
                original_start=obj.start

                async def persist_running_intent():
                    try:
                        await mod.checkpoint(False, {'auto_resume': True})
                    except Exception as ex:
                        try: obj.error=f'Initial checkpoint failed: {type(ex).__name__}: {ex}'
                        except Exception: pass

                async def persistent_start(self):
                    await original_start()
                    if self.running:
                        # Do not make the Start HTTP request wait on GitHub.
                        asyncio.create_task(persist_running_intent())

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
                    # Save the stop intent without blocking the UI on the GitHub request.
                    async def save_stop():
                        try: await mod.checkpoint(True, {'auto_resume': False})
                        except Exception: pass
                    asyncio.create_task(save_stop())
                    try: mod.setstate('phase','stopped')
                    except Exception: pass

                obj.start=types.MethodType(persistent_start,obj)
                obj.stop=types.MethodType(safe_stop,obj)
                obj._checkpoint_patches_installed=True

            if not getattr(app_obj,'_auto_resume_handler_installed',False):
                async def auto_resume():
                    try:
                        cp=await load_checkpoint()
                        if (cp and cp.get('paused') is False
                                and not getattr(obj,'running',False)
                                and getattr(obj,'_safe_resume_allowed',True)
                                and mod.latest('candidates') is None):
                            await obj.start()
                    except Exception as ex:
                        obj.error=f'Auto-resume failed: {type(ex).__name__}: {ex}'
                app_obj.add_event_handler('startup',auto_resume)
                app_obj._auto_resume_handler_installed=True
            return
        time.sleep(0.05)

threading.Thread(target=_install_patches,daemon=True).start()
