import {expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {Board} from '../src/components/maze/Board';
import {Exploration} from '../src/components/maze/Exploration';
import {startGame} from '../src/utilities/game';
import {validateMazeSearch} from '../src/utilities/search';

test('local board renders remembered cells and never exposes hidden goal or walls',()=>{
  const game=startGame('local-map-v1',7);
  const html=renderToStaticMarkup(<Board {...game}/>);
  expect(html).toContain('出口尚未发现');
  expect(html).not.toContain('goal-label');
  expect((html.match(/data-cell="\?"/g)??[]).length).toBe(40);
  expect((html.match(/in-view/g)??[]).length).toBe(9);
  expect(html).not.toContain('出口 (5, 5)');
  expect(game.memory).not.toBeNull();
  const sensor=renderToStaticMarkup(<Exploration memory={game.memory!} state={game.state}/>);
  expect(sensor).toContain('已观察 9 / 49 格');
  expect(sensor).toContain('已访问 1 格');
  expect(sensor).not.toContain('data-cell="G"');
});

test('local lesson URL validates without changing legacy defaults',()=>{
  expect(validateMazeSearch({course:'local-map-v1',seed:'42'}).course).toBe('local-map-v1');
  expect(validateMazeSearch({}).course).toBe('fixed-map-v1');
});
