from __future__ import annotations
import asyncio,json,os,sqlite3,threading,time
from datetime import datetime,timezone
import mpmath as mp
from flint import acb,ctx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from ollama import Client
from checkpoint import load_checkpoint,save_checkpoint

START_N=10_000_000_000_001
DB=os.getenv('RH_DB_PATH','riemann_lab.sqlite3')
VERIFY_DPS=int(os.getenv('RH_VERIFY_DPS','160'))
FLINT_SEARCH_DPS=int(os.getenv('RH_FLINT_SEARCH_DPS','50'))
BATCH=max(1,int(os.getenv('RH_BATCH_SIZE','8')))
MAX_N=int(os.getenv('RH_MAX_N','0'))
OFFLINE_EVERY=max(1,int(os.getenv('RH_OFFLINE_SCAN_INTERVAL','25')))
INTERVAL=max(0.0,float(os.getenv('RH_SEARCH_INTERVAL','0.02')))
MODEL=os.getenv('OLLAMA_MODEL','gpt-oss:120b-cloud');API_KEY=os.getenv('OLLAMA_API_KEY','')
DISCOVERY=mp.mpf('1e-25');FINAL=mp.mpf('1e-40');LOCK=threading.RLock()

def now():return datetime.now(timezone.utc).isoformat()
def db():
 c=sqlite3.connect(DB,check_same_thread=False);c.row_factory=sqlite3.Row;return c
def init():
 with LOCK:
  c=db();c.executescript('''CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);CREATE TABLE IF NOT EXISTS zeros(id INTEGER PRIMARY KEY,n INTEGER,real_part TEXT,imaginary_part TEXT,residual TEXT,deviation TEXT,search_dps INTEGER,verify_dps INTEGER,verified INTEGER,suspicious INTEGER,method TEXT,created_at TEXT);CREATE TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY,source TEXT,real_part TEXT,imaginary_part TEXT,residual TEXT,deviation TEXT,verify_dps INTEGER,status TEXT,evidence TEXT,created_at TEXT);CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY,candidate_id INTEGER,model TEXT,report TEXT,created_at TEXT);''');c.commit();c.close()
def state(k,default=None):
 if default is None:default=str(START_N)
 with LOCK:
  c=db();r=c.execute('SELECT value FROM state WHERE key=?',(k,)).fetchone();c.close();return r['value'] if r else default
def setstate(k,v):
 with LOCK:
  c=db();c.execute('INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(k,str(v)));c.commit();c.close()
def count(t):
 with LOCK:
  c=db();n=c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0];c.close();return n
def latest(t):
 with LOCK:
  c=db();r=c.execute(f'SELECT * FROM {t} ORDER BY id DESC LIMIT 1').fetchone();c.close();return dict(r) if r else None
def zeros(limit=30):
 with LOCK:
  c=db();r=c.execute('SELECT * FROM zeros ORDER BY id DESC LIMIT ?',(limit,)).fetchall();c.close();return [dict(x) for x in r]
def addzero(n,real,imag,res,dev,verified,suspicious,method):
 with LOCK:
  c=db();c.execute('INSERT INTO zeros VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?)',(n,real,imag,res,dev,FLINT_SEARCH_DPS,VERIFY_DPS,int(verified),int(suspicious),method,now()));c.commit();c.close()
def addcandidate(source,real,imag,res,dev,e):
 with LOCK:
  c=db();cur=c.execute('INSERT INTO candidates VALUES(NULL,?,?,?,?,?,?,?,?,?)',(source,real,imag,res,dev,VERIFY_DPS,'numerically_survived',json.dumps(e),now()));c.commit();i=cur.lastrowid;c.close();return i
def addreport(cid,report):
 with LOCK:
  c=db();c.execute('INSERT INTO reports VALUES(NULL,?,?,?,?)',(cid,MODEL,report,now()));c.commit();c.close()

def flint_zero_batch(start,count_):
 ctx.dps=FLINT_SEARCH_DPS
 return list(acb.zeta_zeros(start,count_))

def flint_verify_zero(n):
 ctx.dps=VERIFY_DPS
 z=acb.zeta_zero(n)
 # The returned Arb enclosure is a certified high-precision enclosure of the nth critical-line zero.
 residual=z.zeta().abs_upper()
 real=z.real.str(VERIFY_DPS,radius=False,more=True)
 imag=z.imag.str(VERIFY_DPS,radius=False,more=True)
 res=residual.str(70,radius=False,more=True)
 return real,imag,res

def exploratory(t):
 # Exploratory only; this does not establish exhaustiveness.
 with mp.workdps(60):
  for r in ('.10','.20','.30','.40','.45','.55','.60','.70','.80','.90'):
   for d in ('-.35','-.15','0','.15','.35'):
    seed=mp.mpc(r,mp.mpf(t)+mp.mpf(d))
    try:
     root=mp.findroot(mp.zeta,(seed,seed+mp.mpc('.003','.003')),solver='secant',tol=mp.mpf('1e-45'),maxsteps=40,verify=False)
     res=abs(mp.zeta(root));dev=abs(mp.re(root)-mp.mpf('.5'))
     if mp.im(root)>10 and 0<mp.re(root)<1 and res<mp.mpf('1e-40') and dev>DISCOVERY:return root
    except Exception:pass
 return None

def ai_report(e):
 if not API_KEY:return 'OLLAMA_API_KEY is not set. Numerical evidence follows:\n\n'+json.dumps(e,indent=2)
 client=Client(host='https://ollama.com',headers={'Authorization':f'Bearer {API_KEY}'})
 system='You are a mathematical research-report assistant. Explain why the program stopped, preserve numerical evidence, distinguish numerical evidence from formal proof, and never claim RH is disproved unless the supplied evidence establishes that. Do not invent facts or computations.'
 r=client.chat(model=MODEL,messages=[{'role':'system','content':system},{'role':'user','content':'Generate a cautious research report from this evidence:\n'+json.dumps(e,indent=2)}])
 return r['message']['content']

async def checkpoint(paused=False,extra=None):
 payload={'next_n':int(state('next_n')),'zeros_checked':count('zeros'),'updated_at_utc':now(),'paused':paused,'search_start_n':START_N}
 if extra:payload.update(extra)
 try:
  ok=await save_checkpoint(payload)
  if not ok:search.error='GitHub checkpoint is not configured or could not be saved.'
  return ok
 except Exception as ex:
  search.error=f'Checkpoint save failed: {type(ex).__name__}: {ex}';return False

class Search:
 def __init__(self):self.running=False;self.stopped=False;self.reason='';self.error='';self.last=None;self.task=None
 def status(self):
  return {'running':self.running,'stopped':self.stopped,'reason':self.reason,'error':self.error,'start_n':START_N,'next_n':int(state('next_n',str(START_N))),'current_n':int(state('current_n',state('next_n',str(START_N)))),'phase':state('phase','idle'),'zeros_checked':count('zeros'),'candidate_count':count('candidates'),'search_dps':FLINT_SEARCH_DPS,'verify_dps':VERIFY_DPS,'batch_size':BATCH,'engine':'FLINT/Arb Platt zero engine','checkpoint':'github' if os.getenv('CHECKPOINT_TOKEN') else 'local-only','last_zero':self.last,'candidate':latest('candidates'),'report':latest('reports')}
 async def start(self):
  if self.running:return
  cp=await load_checkpoint();saved_n=int(cp.get('next_n',START_N)) if cp else START_N
  n=max(START_N,saved_n);setstate('next_n',n);setstate('current_n',n);setstate('phase','starting')
  if latest('candidates') is not None:self.stopped=True;self.reason='A verified candidate already exists; start is disabled for this run.';return
  self.stopped=False;self.reason='';self.error='';self.running=True;self.task=asyncio.create_task(self.run())
 async def stop(self):
  self.running=False;self.stopped=False;self.reason='Manually stopped';setstate('phase','stopping');await checkpoint(True)
 async def candidate(self,source,real,imag,res,dev,extra):
  e={'source':source,'real_part':real,'imaginary_part':imag,'zeta_abs':res,'critical_line_deviation':dev,'search_precision_digits':FLINT_SEARCH_DPS,'verification_precision_digits':VERIFY_DPS,'extra':extra,'timestamp_utc':now()}
  cid=addcandidate(source,real,imag,res,dev,e);self.running=False;self.stopped=True;self.reason='A numerically verified nontrivial zero candidate has Re(s) != 1/2';setstate('phase','candidate_found');await checkpoint(True,{'candidate':e,'candidate_found':True})
  try:addreport(cid,await asyncio.to_thread(ai_report,e))
  except Exception as ex:self.error=f'AI report generation failed: {type(ex).__name__}: {ex}'
 async def run(self):
  try:
   while self.running:
    start=max(int(state('next_n',str(START_N))),START_N)
    if MAX_N and start>MAX_N:
     self.running=False;self.reason=f'Configured maximum n={MAX_N} reached';setstate('phase','max_reached');await checkpoint(True);return
    batch=min(BATCH,MAX_N-start+1) if MAX_N else BATCH
    setstate('current_n',start);setstate('phase',f'computing batch {start}..{start+batch-1}')
    t0=time.monotonic();zs=await asyncio.to_thread(flint_zero_batch,start,batch);elapsed=time.monotonic()-t0
    if len(zs)!=batch:raise RuntimeError(f'FLINT returned {len(zs)} zeros for requested batch of {batch}')
    setstate('phase','high_precision_verification')
    for i,z in enumerate(zs):
     n=start+i
     if not self.running:return
     real,imag,res=await asyncio.to_thread(flint_verify_zero,n)
     real_mp=mp.mpf(real);dev=abs(real_mp-mp.mpf('.5'))
     verified=mp.mpf(res)<mp.mpf('1e-50')
     suspicious=dev>FINAL
     addzero(n,real,imag,res,mp.nstr(dev,60),verified,suspicious,'FLINT/Arb Platt zeta-zero + 160-digit Arb verification')
     setstate('next_n',n+1);setstate('current_n',n);setstate('phase','zero_completed')
     self.last={'n':n,'real_part':real,'imaginary_part':imag,'residual_upper_bound':res,'deviation':mp.nstr(dev,60),'verified':verified,'batch_seconds':round(elapsed,3)}
     if suspicious and verified:
      return await self.candidate(f'nth-zero enumeration n={n}',real,imag,res,mp.nstr(dev,60),{'n':n,'batch_size':batch})
    if start//BATCH != (start+batch-1)//BATCH:await checkpoint()
    else:await checkpoint()
    if (start+batch-1)%OFFLINE_EVERY==0:
     setstate('phase','off_critical_exploration');x=await asyncio.to_thread(exploratory,mp.mpf(zs[-1].imag.str(50,radius=False,more=True)))
     if x is not None:
      # Re-evaluate exploratory candidate at higher mpmath precision before stopping.
      with mp.workdps(VERIFY_DPS):
       rr=mp.findroot(mp.zeta,(x,x+mp.mpc('.001','.001')),solver='secant',tol=mp.mpf('1e-70'),maxsteps=80,verify=False)
       res2=abs(mp.zeta(rr));dev2=abs(mp.re(rr)-mp.mpf('.5'))
      if mp.im(rr)>10 and 0<mp.re(rr)<1 and res2<mp.mpf('1e-50') and dev2>FINAL:
       return await self.candidate('2-D off-critical-line exploratory search',mp.nstr(mp.re(rr),VERIFY_DPS),mp.nstr(mp.im(rr),VERIFY_DPS),mp.nstr(res2,70),mp.nstr(dev2,70),{'seed_height':mp.nstr(zs[-1].imag,50)})
    await asyncio.sleep(INTERVAL)
  except asyncio.CancelledError:pass
  except Exception as ex:
   self.running=False;self.reason='Search paused because of an execution error';self.error=repr(ex);setstate('phase','error');await checkpoint(True,{'error':repr(ex)})

init();search=Search();app=FastAPI(title='Riemann Hypothesis Search Lab')
@app.get('/',response_class=HTMLResponse)
async def home():return HTML
@app.get('/api/status')
async def status():return search.status()
@app.post('/api/start')
async def start():await search.start();return search.status()
@app.post('/api/stop')
async def stop():await search.stop();return search.status()
@app.get('/api/zeros')
async def api_zeros(limit:int=30):return zeros(min(max(limit,1),500))
@app.get('/api/candidate')
async def api_candidate():return latest('candidates')
@app.get('/api/report')
async def api_report():return latest('reports')
@app.get('/api/health')
async def health():return {'ok':True,'running':search.running,'start_n':START_N,'next_n':int(state('next_n')),'current_n':int(state('current_n',state('next_n'))),'phase':state('phase','idle'),'checkpoint':'github' if os.getenv('CHECKPOINT_TOKEN') else 'local-only','engine':'FLINT/Arb Platt'}
HTML='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Riemann Search Lab</title><style>body{font-family:system-ui;background:#0d1117;color:#e6edf3;max-width:1100px;margin:auto;padding:25px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:15px;margin:12px 0}.v{font-size:24px;font-weight:700}button{padding:10px 16px;margin:5px}pre{white-space:pre-wrap;overflow:auto}small{color:#8b949e}</style></head><body><h1>🧮 Riemann Hypothesis Search Lab</h1><p>Continuous numerical investigation beginning at zero index <b>10,000,000,000,001</b>.</p><p><small>Fast engine: FLINT/Arb with Platt's zeta-zero algorithms. Enumerated zeros are verified at 160 decimal digits. Off-critical-line exploration remains exploratory and is not an exhaustive proof of absence.</small></p><button onclick="start()">Start / Resume</button><button onclick="stop()">Stop & Save</button><div class="grid"><div class="card">Status<div id="s" class="v">—</div></div><div class="card">Next n<div id="n" class="v">—</div></div><div class="card">Current n<div id="cn" class="v">—</div></div><div class="card">Zeros recorded<div id="c" class="v">—</div></div><div class="card">Phase<div id="p" class="v">—</div></div><div class="card">Verification<div id="d" class="v">—</div></div><div class="card">Engine<div id="en" class="v">—</div></div><div class="card">Checkpoint<div id="cp" class="v">—</div></div></div><div class="card"><h2>Latest zero</h2><pre id="z">—</pre></div><div class="card"><h2>Candidate</h2><pre id="ca">None</pre></div><div class="card"><h2>AI report</h2><pre id="r">None</pre></div><div class="card"><h2>Error</h2><pre id="e">None</pre></div><script>async function start(){render(await (await fetch('/api/start',{method:'POST'})).json())}async function stop(){render(await (await fetch('/api/stop',{method:'POST'})).json())}function render(x){s.textContent=x.running?'RUNNING':x.stopped?'STOPPED':'PAUSED';n.textContent=x.next_n;cn.textContent=x.current_n;c.textContent=x.zeros_checked;p.textContent=x.phase;d.textContent=x.verify_dps+' digits';en.textContent=x.engine;cp.textContent=x.checkpoint;z.textContent=JSON.stringify(x.last_zero,null,2);ca.textContent=x.candidate?JSON.stringify(x.candidate,null,2):'None';r.textContent=x.report?x.report.report:'None';e.textContent=x.error||'None'}async function poll(){try{render(await (await fetch('/api/status')).json())}catch(e){}}poll();setInterval(poll,1500)</script></body></html>'''
if __name__=='__main__':
 import uvicorn;uvicorn.run('app:app',host='0.0.0.0',port=int(os.getenv('PORT','8000')))
