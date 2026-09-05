import base64,json,os,httpx

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
