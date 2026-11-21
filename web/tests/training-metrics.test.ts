import {expect,test} from 'bun:test';
import {createTrainingMetricCollector,extractLegacyMetrics,mergeLegacyMetrics,parseTrainingMetric,upsertTrainingMetric} from '../src/utilities/training-metrics';
import type {TrainingMetric} from '../src/utilities/training-contract';
const point=(epoch=0,total=1000)=>({epoch,total,successRate:0.123456789,evaluatedMaps:64,split:'validation',loss:1,accuracy:0.8});
const line=(value:unknown)=>'ODYMAZE_METRIC '+JSON.stringify(value);

test('structured metrics validate measured success, bounds, and job identity',()=>{
 expect(parseTrainingMetric(line(point()))?.successRate).toBe(0.123456789);
 for(const value of [{...point(),successRate:null},{...point(),epoch:1001},{...point(),total:5001},
  {...point(),evaluatedMaps:1},{...point(),accuracy:2},{...point(),loss:-1},{...point(),epoch:0.5}]) expect(parseTrainingMetric(line(value))).toBeNull();
 expect(parseTrainingMetric('ODYMAZE_METRIC {"successRate":NaN}')).toBeNull();
 expect(parseTrainingMetric(line(point()),{course:'fixed-map-v1',total:1000})).toBeNull();
 expect(parseTrainingMetric(line(point()),{course:'random-map-v1',total:999})).toBeNull();
 expect(parseTrainingMetric(line({...point(),successRate:undefined}))).toBeNull();
});

test('chunk boundaries, duplicate epochs and final unterminated lines retain exact points',()=>{
 let logs='';const points:TrainingMetric[]=[];
 const collector=createTrainingMetricCollector({course:'random-map-v1',total:1000,onLog:text=>logs+=text,onMetric:metric=>upsertTrainingMetric(points,metric)});
 const text='训练开始\n'+line(point())+'\n'+line({...point(),successRate:0.5})+'\n'+line(point(1));
 for(let i=0;i<text.length;i+=7) collector.push(text.slice(i,i+7));
 collector.end();
 expect(points.map(p=>p.epoch)).toEqual([0,1]);expect(points[0].successRate).toBe(0.5);expect(logs).toBe('训练开始\n');
});

test('1001 measured epochs survive human log truncation and malformed oversized lines',()=>{
 let logs='';const points:TrainingMetric[]=[];
 const collector=createTrainingMetricCollector({course:'random-map-v1',total:1000,onLog:text=>logs=(logs+text).slice(-24000),onMetric:metric=>upsertTrainingMetric(points,metric)});
 collector.push('ODYMAZE_METRIC '+ 'x'.repeat(20000));collector.push('\n');
 for(let epoch=0;epoch<=1000;epoch++) collector.push('human output '.repeat(50)+'\n'+line(point(epoch))+'\n');
 collector.push('ODYMAZE_METRIC invalid\n');collector.end();
 expect(points).toHaveLength(1001);expect(logs).toHaveLength(24000);expect(logs).not.toContain('ODYMAZE_METRIC');
 // Failure/cancellation only changes lifecycle state; collected points remain available.
 const failed={state:'failed',metrics:points};expect(failed.metrics.at(-1)?.epoch).toBe(1000);
});

test('legacy logs retain sampled epochs without fabricating gaps or accuracy success',()=>{
 const logs='epoch 10/30; loss 1; accuracy 99%; validation success 12.5%\nepoch 20/30; validation success 25%\nepoch 30/30; accuracy 100%';
 expect(extractLegacyMetrics(logs,'random-map-v1').map(p=>[p.epoch,p.successRate])).toEqual([[10,0.125],[20,0.25]]);
 expect(extractLegacyMetrics(logs,'random-map-v1',29)).toEqual([]);
});

test('successive legacy polls append new epochs and preserve exact structured duplicates',()=>{
 const points:TrainingMetric[]=[];
 const first='epoch 10/30; validation success 12.3%';
 mergeLegacyMetrics(points,first,'random-map-v1');
 expect(points.map(point=>point.epoch)).toEqual([10]);
 upsertTrainingMetric(points,{...point(10,30),split:'validation'});
 mergeLegacyMetrics(points,first+'\nepoch 20/30; validation success 25%','random-map-v1');
 expect(points.map(point=>point.epoch)).toEqual([10,20]);
 expect(points[0].successRate).toBe(0.123456789);
 expect(points[1].successRate).toBe(0.25);
});
