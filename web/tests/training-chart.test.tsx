import {expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {chartMetrics,metricSummary,nearestMetric,metricsCsv} from '../src/utilities/training-chart-data';
import {TrainingSuccessChart} from '../src/components/training/TrainingSuccessChart';
import type {TrainingMetric} from '../src/utilities/training-contract';
const point=(epoch:number,successRate:number):TrainingMetric=>({epoch,total:5,successRate,evaluatedMaps:64,split:'validation'});

test('chart preserves measured epochs and precision; peak first tie and initial change use actual scores',()=>{
  const points=chartMetrics([point(3,.5),point(0,.125),point(1,.25),point(2,.5),point(3,.375)]);
  expect(points.map(item=>item.epoch)).toEqual([0,1,2,3]);
  expect(metricSummary(points)).toEqual({latest:point(3,.375),best:point(2,.5),change:25});
  expect(nearestMetric([point(1,.047),point(5,.375)],3)?.epoch).toBe(1);
  expect(chartMetrics([point(1,NaN),point(2,2),point(-1,.5)])).toEqual([]);
  expect(metricSummary([])).toBeNull();
  expect(metricSummary([point(0,.5),point(1,.5)])?.best.epoch).toBe(0);
  expect(metricSummary([point(2,.5)])?.change).toBeNull();
});

test('CSV contains measured points only and retains exact fraction',()=>{
  const csv=metricsCsv([point(0,0),{...point(3,3/64),loss:.25,accuracy:.5}]);
  expect(csv).toContain('3,5,0.046875,64,validation,0.25,0.5');
  expect(csv.split('\n').filter(Boolean)).toHaveLength(3);
});

test('chart has honest empty, single baseline, sparse measurements and separate fixed-map label',()=>{
  const empty=renderToStaticMarkup(<TrainingSuccessChart metrics={[]} course="random-map-v1"/>);
  expect(empty).toContain('暂无通关率数据');
  expect(empty).not.toContain('id="success-line"');
  const baseline=renderToStaticMarkup(<TrainingSuccessChart metrics={[{...point(0,0),total:0}]} course="random-map-v1"/>);
  expect(baseline).not.toContain('NaN');expect(baseline).not.toContain('Infinity');
  const sparse=renderToStaticMarkup(<TrainingSuccessChart metrics={[point(1,3/64),point(5,.375)]} course="random-map-v1"/>);
  expect(sparse).toContain('验证集通关率，共2个实测点');
  expect(sparse).toContain('37.50%');expect(sparse).toContain('导出 CSV');
  const fixed=renderToStaticMarkup(<TrainingSuccessChart metrics={[{...point(0,0),split:'fixed',evaluatedMaps:1}]} course="fixed-map-v1"/>);
  expect(fixed).toContain('固定地图通关率');expect(fixed).not.toContain('验证集通关率');
});
