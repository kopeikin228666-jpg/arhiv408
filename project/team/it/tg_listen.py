#!/usr/bin/env python3
"""Слушает @OFFICE408_bot. Когда босс пишет — сохраняет в /root/.tg/inbox.jsonl,
отвечает «принял, передал директору» и выходит (выход будит директора)."""
import json,urllib.request,time,os,sys
TOK=open('/root/.tg/token').read().strip();ME=open('/root/.tg/chat_id').read().strip()
API=f'https://api.telegram.org/bot{TOK}/'
OFF='/root/.tg/offset'
def get(m,**p):
    q='&'.join(f'{k}={v}' for k,v in p.items())
    return json.load(urllib.request.urlopen(API+m+('?'+q if q else ''),timeout=70))
def send(t):
    req=urllib.request.Request(API+'sendMessage',json.dumps({'chat_id':ME,'text':t}).encode(),{'Content-Type':'application/json'})
    urllib.request.urlopen(req,timeout=30)
off=int(open(OFF).read()) if os.path.exists(OFF) else 0
deadline=time.time()+float(sys.argv[1] if len(sys.argv)>1 else 6900)
while time.time()<deadline:
    try: r=get('getUpdates',offset=off,timeout=50)
    except Exception as e: time.sleep(5);continue
    got=[]
    for u in r.get('result',[]):
        off=u['update_id']+1;open(OFF,'w').write(str(off))
        m=u.get('message') or {}
        if str(m.get('chat',{}).get('id'))!=ME: continue
        txt=m.get('text') or m.get('caption') or ('[фото]' if m.get('photo') else '[вложение]')
        rp=m.get('reply_to_message') or {}
        rec={'t':m.get('date'),'text':txt,'reply_to':(rp.get('caption') or rp.get('text') or '')[:120]};got.append(rec)
        open('/root/.tg/inbox.jsonl','a').write(json.dumps(rec,ensure_ascii=False)+'\n')
    if got:
        try: send('✅ Принял, передал директору. Отвечу здесь.')
        except Exception: pass
        print('NEW:',json.dumps(got,ensure_ascii=False));sys.exit(0)
print('TIMEOUT')
