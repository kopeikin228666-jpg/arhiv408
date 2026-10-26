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
        cq=u.get('callback_query')
        if cq and cq.get('data')=='done':
            try: urllib.request.urlopen(urllib.request.Request(API+'answerCallbackQuery',json.dumps({'callback_query_id':cq['id'],'text':'Уже решено'}).encode(),{'Content-Type':'application/json'}),timeout=20)
            except Exception: pass
            continue
        if cq and str(cq.get('from',{}).get('id'))==ME:
            rec={'t':int(time.time()),'button':cq.get('data'),'text':'[кнопка] '+str(cq.get('data'))};got.append(rec)
            open('/root/.tg/inbox.jsonl','a').write(json.dumps(rec,ensure_ascii=False)+'\n')
            try:
                req=urllib.request.Request(API+'answerCallbackQuery',json.dumps({'callback_query_id':cq['id'],'text':'Принято ✅'}).encode(),{'Content-Type':'application/json'});urllib.request.urlopen(req,timeout=20)
                mm=cq.get('message') or {}
                req=urllib.request.Request(API+'editMessageReplyMarkup',json.dumps({'chat_id':ME,'message_id':mm.get('message_id'),'reply_markup':{'inline_keyboard':[[{'text':'✔️ Решено: '+('ДА' if str(cq.get('data','')).split(':')[-1] in ('yes','back') else 'НЕТ'),'callback_data':'done'}]]}}).encode(),{'Content-Type':'application/json'});urllib.request.urlopen(req,timeout=20)
            except Exception: pass
            continue
        m=u.get('message') or {}
        if str(m.get('chat',{}).get('id'))!=ME: continue
        txt=m.get('text') or m.get('caption') or ('[фото]' if m.get('photo') else '[вложение]')
        vo=m.get('voice') or m.get('audio') or m.get('video_note')
        if vo:
            try:
                fp=get('getFile',file_id=vo['file_id'])['result']['file_path']
                os.makedirs('/home/claude/tg_voice',exist_ok=True)
                dst='/home/claude/tg_voice/%s_%s'%(m.get('date'),os.path.basename(fp))
                open(dst,'wb').write(urllib.request.urlopen(f'https://api.telegram.org/file/bot{TOK}/{fp}',timeout=60).read())
                txt='[голосовое %sс] %s'%(vo.get('duration','?'),dst)
            except Exception as e: txt='[голосовое, не скачалось: %s]'%e
        rp=m.get('reply_to_message') or {}
        rec={'t':m.get('date'),'text':txt,'reply_to':(rp.get('caption') or rp.get('text') or '')[:120]};got.append(rec)
        open('/root/.tg/inbox.jsonl','a').write(json.dumps(rec,ensure_ascii=False)+'\n')
    if got:
        try: send('✅ Принял, передал директору. Отвечу здесь.')
        except Exception: pass
        print('NEW:',json.dumps(got,ensure_ascii=False));sys.exit(0)
print('TIMEOUT')
