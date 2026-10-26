#!/usr/bin/env python3
"""Бот OFFICE408 → боссу в Телегу.
tg.py text "сообщение"        — текст (HTML)
tg.py video путь "подпись"     — видео
tg.py photo путь "подпись"     — картинка
tg.py setup                    — найти chat_id после того, как босс нажал Start
Токен: /root/.tg/token, chat_id: /root/.tg/chat_id
"""
import sys,json,urllib.request,uuid,os
TOK=open('/root/.tg/token').read().strip()
API=f'https://api.telegram.org/bot{TOK}/'
def call(m,data=None,files=None):
    if files:
        b=uuid.uuid4().hex;body=b''
        for k,v in (data or {}).items():
            body+=f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k,p in files.items():
            body+=f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{os.path.basename(p)}"\r\n\r\n'.encode()+open(p,'rb').read()+b'\r\n'
        body+=f'--{b}--\r\n'.encode()
        req=urllib.request.Request(API+m,body,{'Content-Type':f'multipart/form-data; boundary={b}'})
    else:
        req=urllib.request.Request(API+m,json.dumps(data or {}).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=120))
def chat(): return open('/root/.tg/chat_id').read().strip()
if __name__=='__main__':
    a=sys.argv[1:]
    if a[0]=='setup':
        u=call('getUpdates')
        ids=[(x.get('message') or {}).get('chat',{}).get('id') for x in u['result']]
        ids=[i for i in ids if i]
        if not ids: print('нет чатов — босс ещё не нажал Start');sys.exit(1)
        open('/root/.tg/chat_id','w').write(str(ids[-1]));print('ok')
    elif a[0]=='text': print(call('sendMessage',{'chat_id':chat(),'text':a[1],'parse_mode':'HTML','disable_web_page_preview':True})['ok'])
    elif a[0]=='video': print(call('sendVideo',{'chat_id':chat(),'caption':a[2] if len(a)>2 else '','parse_mode':'HTML'},{'video':a[1]})['ok'])
    elif a[0]=='photo': print(call('sendPhoto',{'chat_id':chat(),'caption':a[2] if len(a)>2 else '','parse_mode':'HTML'},{'photo':a[1]})['ok'])
