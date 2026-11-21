import {TrainingView} from '../src/views/TrainingView';
import {expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {Decisions} from '../src/components/maze/Decisions';
import {TrainingTaskForm} from '../src/components/training/NewTrainingDialog';
import {TrainingTaskForm} from '../src/components/training/NewTrainingDialog';
import {TrainingSuccessChart} from '../src/components/training/TrainingSuccessChart';
import {startGame} from '../src/utilities/game';

test('predictions retain measured percentages and expose bounded progress semantics',()=>{
  const game=startGame('fixed-map-v1',7);
  const html=renderToStaticMarkup(<Decisions {...game} mode="model" values={[.125,.375,.5]} course="fixed-map-v1" previousAction={0} pending={false}/>);
  expect(html.match(/role="progressbar"/g)).toHaveLength(3);
  expect(html).toContain('aria-valuenow="12.5"');
  expect(html).toContain('aria-valuenow="37.5"');
  expect(html).toContain('aria-valuenow="50"');
  expect(html).toContain('12.5%');expect(html).toContain('50.0%');
});

test('training cannot start before its dataset loads and its numeric fields retain bounds',()=>{
  const html=renderToStaticMarkup(<TrainingTaskForm course="random-map-v1" datasets={[]} blocked={true} error="" onClose={()=>{}} onSubmit={()=>{}}/>);
  const seed=html.match(/<input\b[^>]*max="2147483647"[^>]*>/)?.[0];
  expect(seed).toContain('min="0"');expect(seed).toContain('max="2147483647"');
  const epochs=html.match(/<input\b[^>]*max="1000"[^>]*>/)?.[0];
  expect(epochs).toContain('min="0"');expect(epochs).toContain('max="1000"');
  expect(html).toMatch(/<button[^>]*disabled=""[^>]*id="start-training"/);
  expect(html.match(/data-od-slot="field-label"/g)).toHaveLength(3);
});

test('single baseline has a named disabled slider without a fictitious later epoch',()=>{
  const html=renderToStaticMarkup(<TrainingSuccessChart course="random-map-v1" metrics={[{epoch:0,total:0,successRate:0,evaluatedMaps:64,split:'validation'}]}/>);
  expect(html).toContain('查看通关率轮次');
  expect(html).toContain('aria-valuenow="0"');
  expect(html).toMatch(/<input(?=[^>]*type="range")(?=[^>]*disabled="")[^>]*>/);
});

test('training workspace defaults to the task list rather than an embedded training form',()=>{
 const html=renderToStaticMarkup(<TrainingView course="random-map-v1" onFinished={async()=>{}} onSelectTask={()=>{}}/>);
 expect(html).toContain('本课训练任务列表');expect(html).toContain('添加训练任务');expect(html).not.toContain('id="start-training"');
});
