from __future__ import annotations
import asyncio, json, os, sqlite3, threading
from datetime import datetime, timezone
import mpmath as mp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from ollama import Client

DB=os.getenv('RH_DB_PATH','riemann_lab.sqlite3'); DBLOCK=threading.RLock()
SEARCH_DPS=int(os.getenv('RH_DPS','80')); VERIFY_DPS=int(os.getenv('RH_VERIFY_DPS','160'))
BATCH=max(1,int(os.getenv('RH_BATCH_SIZE','1'))); MAX_N=int(os.getenv('RH_MAX_N','0'))
INTERVAL=max(0.0,float(os.getenv('RH_SEARCH_INTERVAL','0.05')))
OFFLINE_EVERY=max(1,int(os.getenv('RH_OFFLINE_SCAN_INTERVAL','25')))
MODEL=os.getenv('OLLAMA_MODEL','gpt-oss:120b-cloud'); API_KEY=os.getenv('OLLAMA_API_KEY','')
DISCOVERY=mp.mpf('1e-25'); FINAL=mp.mpf('1e-40')

def now(): return datetime.now(timezone.utc).isoformat()
def db():
    c=sqlite3.connect(DB,check_same_thread=False); c.row_factory=sqlite3.Row; return c

def init():
    with DBLOCK:
        c=db(); c.executescript('''CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS zeros(id INTEGER PRIMARY KEY,n INTEGER,real_part TEXT,imaginary_part TEXT,residual TEXT,deviation TEXT,search_dps INTEGER,verify_dps INTEGER,verified INTEGER,suspicious INTEGER,method TEXT,created_at TEXT);
CREATE TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY,source TEXT,real_part TEXT,imaginary_part TEXT,residual TEXT,deviation TEXT,verify_dps INTEGER,status TEXT,evidence TEXT,created_at TEXT);
CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY,candidate_id INTEGER,model TEXT,report TEXT,created_at TEXT);'''); c.commit(); c.close()

def state(k,d='1'):
    with DBLOCK:
        c=db(); r=c.execute('SELECT value FROM state WHERE key=?',(k,)).fetchone(); c.close(); return r['value'] if r else d

def setstate(k,v):
    with DBLOCK:
        c=db(); c.execute('INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(k,str(v))); c.commit(); c.close()

def addzero(n,s,rs,rv,dev,verified,suspicious):
    with DBLOCK:
        c=db(); c.execute('INSERT INTO zeros VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?)',(n,mp.nstr(s.real,VERIFY_DPS),mp.nstr(s.imag,VERIFY_DPS),mp.nstr(rs,40),mp.nstr(dev,40),SEARCH_DPS,VERIFY_DPS,int(verified),int(suspicious),'zetazero + high precision Newton',now())); c.commit(); c.close()

def addcandidate(source,s,res,dev,evidence):
    with DBLOCK:
        c=db(); cur=c.execute('INSERT INTO candidates VALUES(NULL,?,?,?,?,?,?,?,?,?)',(source,mp.nstr(s.real,VERIFY_DPS),mp.nstr(s.imag,VERIFY_DPS),mp.nstr(res,60),mp.nstr(dev,60),VERIFY_DPS,'numerically_survived',json.dumps(evidence),now())); c.commit(); i=cur.lastrowid; c.close(); return i

def addreport(cid,report):
    with DBLOCK:
        c=db(); c.execute('INSERT INTO reports VALUES(NULL,?,?,?,?)',(cid,MODEL,report,now())); c.commit(); c.close()

def latest(table):
    with DBLOCK:
        c=db(); r=c.execute(f'SELECT * FROM {table} ORDER BY id DESC LIMIT 1').fetchone(); c.close(); return dict(r) if r else None

def count(table):
    with DBLOCK:
        c=db(); n=c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]; c.close(); return n

def zeros(limit=30):
    with DBLOCK:
        c=db(); r=c.execute('SELECT * FROM zeros ORDER BY id DESC LIMIT ?',(limit,)).fetchall(); c.close(); return [dict(x) for x in r]

def verify(s):
    with mp.workdps(VERIFY_DPS):
        s=mp.mpc(s); refined=s
        for _ in range(8):
            f=mp.zeta(refined); df=mp.diff(mp.zeta,refined)
            if abs(df)<mp.mpf('1e-120'): break
            refined-=f/df
        res=abs(mp.zeta(refined)); dev=abs(mp.re(refined)-mp.mpf('.5'))
        return res,refined,dev

def exploratory(t):
    with mp.workdps(VERIFY_DPS):
        for r in ['.10','.20','.30','.40','.45','.55','.60','.70','.80','.90']:
            for d in ['-.35','-.15','0','.15','.35']:
                seed=mp.mpc(r,mp.mpf(t)+mp.mpf(d))
                try:
                    root=mp.findroot(mp.zeta,(seed,seed+mp.mpc('.003','.003')),solver='secant',tol=mp.mpf('1e-70'),maxsteps=60,verify=False)
                    res=abs(mp.zeta(root)); dev=abs(mp.re(root)-mp.mpf('.5'))
                    if mp.im(root)>10 and 0<mp.re(root)<1 and res<mp.mpf('1e-50') and dev>DISCOVERY: return root
                except Exception: pass
    return None

def ai_report(e):
    if not API_KEY: return 'OLLAMA_API_KEY is not set. Numerical evidence follows:\n\n'+json.dumps(e,indent=2)
    client=Client(host='https://ollama.com', headers={'Authorization':f'Bearer {API_KEY}'})
    instructions='''You are a mathematical research-report assistant. A computational Riemann-zeta search produced a candidate that survived numerical verification. Explain exactly why the program stopped, preserve all numerical evidence, and distinguish numerical evidence from a formal mathematical proof. Do not claim the Riemann Hypothesis is disproved unless the supplied evidence actually establishes that. Do not invent facts or computations.'''
    r=client.chat(model=MODEL,messages=[{'role':'system','content':instructions},{'role':'user','content':'Generate a cautious research report from this evidence:\n'+json.dumps(e,indent=2)}])
    return r['message']['content']

class Search:
    def __init__(self): self.running=False; self.stopped=False; self.reason=''; self.error=''; self.last=None; self.task=None
    def status(self): return {'running':self.running,'stopped':self.stopped,'reason':self.reason,'error':self.error,'next_n':int(state('next_n','1')),'zeros_checked':count('zeros'),'candidate_count':count('candidates'),'search_dps':SEARCH_DPS,'verify_dps':VERIFY_DPS,'last_zero':self.last,'candidate':latest('candidates'),'report':latest('reports')}
    async def start(self):
        if self.running:return
        if latest('candidates') is not None:
            self.stopped=True; self.reason='A verified candidate already exists; start is disabled for this run.'; return
        self.stopped=False; self.reason=''; self.error=''; self.running=True; self.task=asyncio.create_task(self.run())
    async def stop(self,reason='Manually stopped'):
        self.running=False; self.stopped=False; self.reason=reason
    async def candidate(self,source,s,res,dev,extra):
        e={'source':source,'real_part':mp.nstr(s.real,VERIFY_DPS),'imaginary_part':mp.nstr(s.imag,VERIFY_DPS),'zeta_abs':mp.nstr(res,70),'critical_line_deviation':mp.nstr(dev,70),'search_precision_digits':SEARCH_DPS,'verification_precision_digits':VERIFY_DPS,'extra':extra,'timestamp_utc':now()}
        cid=addcandidate(source,s,res,dev,e); self.running=False; self.stopped=True; self.reason='A numerically verified nontrivial zero candidate has Re(s) != 1/2.'
        try:addreport(cid,await asyncio.to_thread(ai_report,e))
        except Exception as ex:self.error=f'AI report generation failed: {type(ex).__name__}: {ex}'
    async def run(self):
        try:
            while self.running and not self.stopped:
                start=int(state('next_n','1'))
                for n in range(start,start+BATCH):
                    if not self.running:return
                    if MAX_N and n>MAX_N:
                        self.running=False; self.stopped=False; self.reason=f'Configured maximum n={MAX_N} reached.'; return
                    with mp.workdps(SEARCH_DPS): s=mp.zetazero(n); sr=abs(mp.zeta(s))
                    vr,root,dev=verify(s); verified=mp.im(root)>10 and 0<mp.re(root)<1 and vr<mp.mpf('1e-50'); suspicious=dev>FINAL
                    addzero(n,root,sr,vr,dev,verified,suspicious); setstate('next_n',n+1)
                    self.last={'n':n,'real_part':mp.nstr(root.real,50),'imaginary_part':mp.nstr(root.imag,50),'residual':mp.nstr(vr,40),'deviation':mp.nstr(dev,40),'verified':verified}
                    if verified and suspicious:return await self.candidate(f'nth-zero enumeration n={n}',root,vr,dev,{'n':n,'search_residual':mp.nstr(sr,50)})
                    if n%OFFLINE_EVERY==0:
                        x=await asyncio.to_thread(exploratory,root.imag)
                        if x is not None:
                            vr2,r2,d2=verify(x)
                            if mp.im(r2)>10 and 0<mp.re(r2)<1 and vr2<mp.mpf('1e-50') and d2>FINAL:return await self.candidate('2-D off-critical-line exploratory search',r2,vr2,d2,{'seed_height':mp.nstr(root.imag,50)})
                    await asyncio.sleep(INTERVAL)
        except asyncio.CancelledError: pass
        except Exception as ex:self.running=False; self.stopped=False; self.reason='Search paused because of an execution error.'; self.error=repr(ex)

init(); search=Search(); app=FastAPI(title='Riemann Hypothesis Search Lab')

@app.on_event('startup')
async def startup():
    if os.getenv('RH_AUTO_START','true').lower() not in {'0','false','no'}: await search.start()

@app.get('/',response_class=HTMLResponse)
async def home(): return HTML
@app.get('/api/status')
async def status(): return search.status()
@app.post('/api/start')
async def start():
    await search.start(); return search.status()
@app.post('/api/stop')
async def stop(): await search.stop(); return search.status()
@app.get('/api/zeros')
async def api_zeros(limit:int=30): return zeros(min(max(limit,1),500))
@app.get('/api/candidate')
async def api_candidate(): return latest('candidates')
@app.get('/api/report')
async def api_report(): return latest('reports')
@app.get('/api/health')
async def health(): return {'ok':True,'running':search.running,'zeros_checked':count('zeros')}

HTML='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Riemann Search Lab</title><style>body{font-family:system-ui;background:#0d1117;color:#e6edf3;max-width:1100px;margin:auto;padding:25px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:15px;margin:12px 0}.v{font-size:24px;font-weight:700}button{padding:10px 16px;margin:5px}pre{white-space:pre-wrap;overflow:auto}table{width:100%}td,th{padding:7px;border-bottom:1px solid #30363d;text-align:left}</style></head><body><h1>🧮 Riemann Hypothesis Search Lab</h1><p>Continuous numerical search with high-precision verification.</p><button onclick="fetch('/api/start',{method:'POST'}).then(f)">Start</button><button onclick="fetch('/api/stop',{method:'POST'}).then(f)">Stop</button><div class="grid"><div class="card">Status<div id="s" class="v">—</div></div><div class="card">Next n<div id="n" class="v">—</div></div><div class="card">Zeros checked<div id="c" class="v">—</div></div><div class="card">Verification digits<div id="d" class="v">—</div></div></div><div class="card"><h2>Latest zero</h2><pre id="z">—</pre></div><div class="card"><h2>Candidate</h2><pre id="ca">None</pre></div><div class="card"><h2>AI report</h2><pre id="r">None</pre></div><script>async function f(){let x=await (await fetch('/api/status')).json();s.textContent=x.running?'RUNNING':x.stopped?'CANDIDATE FOUND':'PAUSED';n.textContent=x.next_n;c.textContent=x.zeros_checked;d.textContent=x.verify_dps+' digits';z.textContent=JSON.stringify(x.last_zero,null,2);ca.textContent=x.candidate?JSON.stringify(x.candidate,null,2):'None';r.textContent=x.report?x.report.report:'None'}f();setInterval(f,2000)</script></body></html>'''

if __name__=='__main__':
    import uvicorn; uvicorn.run('app:app',host='0.0.0.0',port=int(os.getenv('PORT','8000')))
