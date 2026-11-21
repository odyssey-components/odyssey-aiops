import { root, product } from "../../scripts/workspace.mjs";
// Product-owned headless interactions with synthetic API state; no screenshots or real data.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createRequire} from 'node:module';
import {join} from 'node:path';
const web=join(product,'web');
const {chromium,expect}=createRequire(join(root,'repositories/odyssey-design/packages/odyssey-web-testkit/package.json'))('@playwright/test');
const server=spawn(join(web,'node_modules/.bin/vite'),['preview','--port','0'],{cwd:web,stdio:['ignore','pipe','inherit']});
const closed=once(server,'close');
let resolveOrigin;const ready=new Promise(resolve=>{resolveOrigin=resolve;});
server.stdout.on('data',chunk=>{const found=String(chunk).match(/http:\/\/127\.0\.0\.1:\d+/);if(found) resolveOrigin(found[0]);});
let browser;
try {
 const origin=await Promise.race([ready,new Promise((_,reject)=>setTimeout(()=>reject(new Error('preview timeout')),20000).unref())]);
 browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:390,height:844}});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));
 const course='random-map-v1',id='12345678-1234-1234-1234-123456789abc';
 const dataset={id,course,title:'合成地图组',description:'测试地图',seed:7,maps:16,validation:64,test:100,createdAt:'2026-11-19T02:03:04Z',sha256:'a'.repeat(64)};
 const metrics=[{epoch:0,total:2,successRate:.125,evaluatedMaps:64,split:'validation'},{epoch:2,total:2,successRate:.5,evaluatedMaps:64,split:'validation'}];
 const job={id,course,action:'train',state:'succeeded',startedAt:dataset.createdAt,logs:'epoch 2/2',metrics};
 await page.route('**/api/training**',route=>route.fulfill({json:{datasets:[dataset],job,active:null}}));
 await page.route('**/api/models',route=>route.fulfill({json:[]}));
 await page.goto(`${origin}/?course=${course}`);
 await expect(page.getByRole('button',{name:'模型',exact:true})).toBeDisabled();
 await page.getByRole('button',{name:'专家',exact:true}).click();
 await expect(page.locator('#mode-expert')).toHaveAttribute('aria-pressed','true');
 await expect.poll(()=>page.locator('#mode-expert').evaluate(element=>getComputedStyle(element).boxShadow)).toBe('none');
 const seams=await page.locator('#mode-expert').evaluate(element=>{
  const style=getComputedStyle(element),next=element.nextElementSibling;
  return {shadow:style.boxShadow,zIndex:Number(style.zIndex),nextZIndex:Number(getComputedStyle(next).zIndex),
   overlap:element.getBoundingClientRect().right-next.getBoundingClientRect().left,border:parseFloat(style.borderRightWidth)};
 });
 assert.equal(seams.shadow,'none','joined selected button must not double its border with an inset shadow');
 assert.ok(seams.zIndex>seams.nextZIndex,'selected border must paint above its following neighbor');
 assert.ok(Math.abs(seams.overlap-seams.border)<.1,'joined buttons overlap by exactly one border');

 await page.locator('#mode-expert').press('ArrowLeft');
 await expect(page.locator('#mode-manual')).toBeFocused();
 assert.notEqual(await page.locator('#mode-manual').evaluate(element=>getComputedStyle(element).boxShadow),'none','keyboard focus must remain visible');

 await page.locator('#mode-manual').press('Space');
 await expect(page.locator('#mode-manual')).toHaveAttribute('aria-pressed','true');
 await page.getByRole('tab',{name:'训练任务',exact:true}).click();
 await page.locator('#add-training-task').click();
 await expect(page.getByRole('spinbutton',{name:'初始化种子',exact:true})).toHaveValue('7');
 await page.getByRole('spinbutton',{name:'训练轮数',exact:true}).fill('3');
 await expect(page.locator('#start-training')).toBeEnabled();
 await page.getByRole('button',{name:'关闭',exact:true}).click();
 await page.getByRole('button',{name:`查看训练任务 ${id}`,exact:true}).click();
 await expect(page.locator('#success-latest')).toHaveText('50.00%');
 const slider=page.getByRole('slider',{name:'查看通关率轮次',exact:true});
 await slider.focus();await slider.press('Home');
 await expect(page.locator('#success-detail')).toContainText('第 0 / 2 轮');
 await slider.press('ArrowRight');
 await expect(page.locator('#success-detail')).toContainText('第 2 / 2 轮');
 await page.getByRole('tab',{name:'地图组',exact:true}).click();
 await expect(page.getByRole('rowheader',{name:'合成地图组',exact:true})).toBeVisible();
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'narrow page must not overflow horizontally');
 assert.deepEqual(errors,[]);
 console.log('PASS shared controls: keyboard modes, disabled model, named fields, sparse slider and narrow layout');
} finally {await browser?.close();server.kill();await closed;}
