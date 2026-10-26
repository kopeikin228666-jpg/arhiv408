#!/usr/bin/env python3
"""ОТДЕЛ КОНТРОЛЯ — ревизия. Запускать на каждом обходе и ПЕРЕД любым отчётом боссу «в работе».
registry.json: [{"job","owner","agent_id","watch":[пути/папки],"started":unix,"deadline":"текст","status":"running|done|failed"}]
Правило: если по job в статусе running нет новых/изменённых файлов в watch > 40 мин — STUCK.
"""
import json,os,time,glob,subprocess
R='/home/claude/voynich/team/control/registry.json'
jobs=json.load(open(R)) if os.path.exists(R) else []
now=time.time();out=[]
for j in jobs:
    if j.get('status')!='running': continue
    latest=0
    for w in j.get('watch',[]):
        for f in glob.glob(w,recursive=True)+([w] if os.path.isfile(w) else []):
            try: latest=max(latest,os.path.getmtime(f))
            except: pass
            if os.path.isdir(f):
                for r,_,fs in os.walk(f):
                    for x in fs:
                        try: latest=max(latest,os.path.getmtime(os.path.join(r,x)))
                        except: pass
    ref=max(latest,j.get('started',now))
    idle=(now-ref)/60
    state='OK' if idle<40 else 'STUCK'
    out.append(f"{state:5} | {j['job']} | {j['owner']} | без движения {idle:.0f} мин | срок: {j.get('deadline','')}")
# инфраструктура
lis=subprocess.run(['pgrep','-f','tg_listen.py'],capture_output=True,text=True).stdout.split()
out.append(('OK   ' if lis else 'STUCK')+' | слушатель Телеги | Сисадмин')
g=subprocess.run(['git','-C','/home/claude/arhiv408_repo','status','--porcelain','-uno'],capture_output=True,text=True).stdout.strip()
out.append(('OK   ' if not g else 'WARN ')+' | GitHub-бэкап | Сисадмин'+('' if not g else ' (есть незапушенные изменения)'))
print('\n'.join(out) if out else 'нет активных задач')
