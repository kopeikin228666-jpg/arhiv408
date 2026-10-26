// Records the office "НА СВО" firing scene.
// usage: node record_firing.js <agents.json> <firedId> <why> <out.webm-dir>
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
const [,,agentsFile,firedId,why,outDir]=process.argv;
const agents=JSON.parse(fs.readFileSync(agentsFile,'utf8'));
const html=fs.readFileSync(path.join(__dirname,'..','office408.html'),'utf8');
(async()=>{
  const b=await chromium.launch();
  const ctx=await b.newContext({viewport:{width:1200,height:820},deviceScaleFactor:1,recordVideo:{dir:outDir,size:{width:1200,height:820}}});
  const p=await ctx.newPage();
  await p.addInitScript(({agents,firedId,why})=>{
    const subs=[];let store={agents:agents.map(a=>({...a}))};
    const snap=list=>({docs:list.map(d=>({id:d._id,data:()=>{const c={...d};delete c._id;return c}})),empty:!list.length});
    function q(name){const o={orderBy:()=>o,limit:()=>o,where:()=>o,onSnapshot:(cb)=>{subs.push([name,cb]);cb(snap(store[name]||[]));return ()=>{}},add:async()=>({}),get:async()=>snap(store[name]||[])};return o}
    const db={collection:q,doc:(path)=>({onSnapshot:(cb)=>{cb({exists:path==='meta/now',data:()=>({phase:'ПРИКАЗ ОБ УВОЛЬНЕНИИ',detail:'по решению надзирателей'})});return ()=>{}},update:async()=>{},set:async()=>{}})};
    window.claude={use:async n=>n==='db'?db:null};
    window.__fire=()=>{const a=store.agents.find(x=>x._id===firedId);a.status='fired';a.firedAt=Date.now();a.firedWhy=why;a.firedTime=new Date().toTimeString().slice(0,5);for(const [n,cb] of subs)if(n==='agents')cb(snap(store.agents))};
  },{agents,firedId,why});
  await p.route(/fonts\.(googleapis|gstatic)\.com/,r=>r.abort());
  const tmp=path.join(outDir,'office.html');fs.writeFileSync(tmp,html);
  await p.goto('file://'+tmp,{waitUntil:'domcontentloaded'});
  await p.waitForTimeout(3500);
  await p.evaluate(()=>window.__fire());
  await p.waitForTimeout(13000);
  const box=await p.locator('#stage').boundingBox();
  fs.writeFileSync(path.join(outDir,'box.json'),JSON.stringify(box));
  await ctx.close();await b.close();
})();
