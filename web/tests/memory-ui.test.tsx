import {expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {RecurrentMemory} from '../src/components/maze/RecurrentMemory';
import {validateMazeSearch} from '../src/utilities/search';
import {MEMORY_LESSONS} from '../src/lessons/memory_content';

test('fourth lesson URL is restored but hidden state is not page selection',()=>{
  const parsed=validateMazeSearch({course:'memory-map-v1',point:'3',seed:'42',mode:'model'});
  expect(parsed.course).toBe('memory-map-v1');
  expect(parsed.point).toBe(3);
  expect(parsed.mode).toBe('model');
  expect(MEMORY_LESSONS).toHaveLength(5);
});

test('GRU panel shows finite hidden values and exact observation count with an independent forget control',()=>{
  const html=renderToStaticMarkup(<RecurrentMemory state={{hidden:Array(32).fill(.25),observations:3}} disabled={false} onForget={()=>{}}/>);
  expect((html.match(/class="hidden-unit"/g)??[]).length).toBe(32);
  expect(html).toContain('已处理 3 次观察');
  expect(html).toContain('清空模型记忆');
  expect(html).not.toContain('单个数值没有固定的人类语义');
  const empty=renderToStaticMarkup(<RecurrentMemory state={null} disabled={false} onForget={()=>{}}/>);
  expect(empty).toContain('加载本课模型后显示隐藏状态');
  expect(empty).toContain('disabled');
});
