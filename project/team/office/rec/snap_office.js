// Фото офиса для Телеги: node snap_office.js agents.json out.png
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
const [,,agentsFile,out]=process.argv;const agents=JSON.parse(fs.readFileSync(agentsFile,'utf8'));
const html=fs.readFileSync(path.join(__dirname,'..','office408.html'),'utf8');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1400,height:900},deviceScaleFactor:1});
 await p.addInitScript((agents)=>{const snap=l=>({docs:l.map(d=>({id:d._id,data:()=>{const c={...d};delete c._id;return c}})),empty:!l.length});
  const q=n=>{const o={orderBy:()=>o,limit:()=>o,where:()=>o,onSnapshot:cb=>{cb(snap(n==='agents'?agents:[]));return()=>{}}};return o};
  window.claude={use:async n=>n==='db'?{collection:q,doc:()=>({onSnapshot:cb=>{cb({exists:false});return()=>{}}})}:null};},agents);
 await p.route(/fonts\.(googleapis|gstatic)\.com/,r=>r.abort());
 const tmp=path.join(path.dirname(out),'_office.html');fs.writeFileSync(tmp,html);
 await p.goto('file://'+tmp);await p.waitForTimeout(9000);
 await p.locator('#stage').screenshot({path:out});await b.close();})();
