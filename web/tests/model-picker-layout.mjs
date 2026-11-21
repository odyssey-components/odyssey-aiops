import { root } from "../../scripts/workspace.mjs";
// Product-owned layout test for actual shared Select markup and built styles.
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {ProAdminProvider} from '@odyssey/ui/patterns/admin-provider';
import {ModelPicker} from '../src/components/models/ModelPicker';

const {chromium}=createRequire(root+'repositories/odyssey-design/packages/odyssey-web-testkit/package.json')('@playwright/test');
const assets=new URL('../dist/assets/',import.meta.url);
const styles=(await readdir(assets)).filter(name=>name.endsWith('.css'));
const css=(await Promise.all(styles.map(name=>readFile(new URL(name,assets),'utf8')))).join('\n');
const run='20261119110341084830-random-map-v1-f5b6d7a4-68cd-4cea-8fa3-a88878787878'.repeat(2);
const html=renderToStaticMarkup(createElement(ProAdminProvider,{appearance:{storageKey:'model-layout'}},createElement(ModelPicker,{
 course:'random-map-v1',initialRuns:[{run,lesson:'random-map-v1'}],selectedRun:run,
 onCatalogChange(){},onLoad(){},info:run+' · 160 轮',message:{kind:'neutral',text:''},
})));
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage();
 for(const [width,cardWidth] of [[390,358],[820,788],[1440,1200],[1440,420]]){
  await page.setViewportSize({width,height:900});
  await page.setContent(`<style>${css}body{margin:0;padding:16px}.model-card{width:100%;max-width:${cardWidth}px}</style>${html}`);
  const bounds=await page.locator('.model-card').evaluate(card=>{
   const trigger=card.querySelector('#model-list'),button=card.querySelector('#refresh-models'),value=trigger.querySelector('[data-od-slot="select-value"]');
   const rect=node=>{const r=node.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height};};
   return {card:rect(card),trigger:rect(trigger),button:rect(button),overflow:document.documentElement.scrollWidth>innerWidth,
    textStyle:{overflow:getComputedStyle(value).overflow,textOverflow:getComputedStyle(value).textOverflow,whiteSpace:getComputedStyle(value).whiteSpace},
    title:card.querySelector('#current-model').title};
  });
  assert.ok(!bounds.overflow,JSON.stringify(bounds));
  assert.ok(bounds.trigger.x>=bounds.card.x && bounds.trigger.right<=bounds.card.right+.5);
  assert.ok(bounds.button.x>=bounds.card.x && bounds.button.right<=bounds.card.right+.5);
  const overlap=bounds.trigger.x<bounds.button.right && bounds.button.x<bounds.trigger.right && bounds.trigger.y<bounds.button.bottom && bounds.button.y<bounds.trigger.bottom;
  assert.equal(overlap,false,JSON.stringify(bounds));
  assert.equal(bounds.textStyle.textOverflow,'ellipsis');assert.equal(bounds.textStyle.whiteSpace,'nowrap');
  assert.equal(bounds.title,run+'/model.onnx');
 }
 console.log('PASS model picker: long names contained, no overlap/overflow, ellipsis/full title at 390/820/1440 and a narrow desktop sidebar.');
}finally{await browser.close();}
