import type { GameState, Maze } from '../../core';
import { generateMaze as randomMaze, validMaze as validRandomMaze } from './random_map_v1';

export interface Memory { known: string[]; visited: number[] }
const DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1]];
export function generateMaze(seed: number): Maze {
  return { ...randomMaze(seed), max_steps: 200 };
}
export function validMaze(value: unknown): value is Maze {
  const maze = value as Maze;
  return Boolean(maze && maze.max_steps === 200 && validRandomMaze({ ...maze, max_steps: 100 }));
}
export function observe(maze: Maze, state: GameState, memory: Memory): Memory {
  const known = memory.known.map(row => [...row]);
  for (let y = Math.max(0, state.y - 1); y <= Math.min(6, state.y + 1); y++) {
    for (let x = Math.max(0, state.x - 1); x <= Math.min(6, state.x + 1); x++) known[y][x] = maze.rows[y][x];
  }
  const visited = [...memory.visited];
  visited[state.y * 7 + state.x] = 1;
  return { known: known.map(row => row.join('')), visited };
}
export function initialMemory(maze: Maze, state: GameState): Memory {
  return observe(maze, state, { known: Array<string>(7).fill('???????'), visited: Array<number>(49).fill(0) });
}
export function encode(memory: Memory, state: GameState): number[] {
  const cells = memory.known.flatMap(row => [...row]);
  const agent = Array<number>(49).fill(0); agent[state.y * 7 + state.x] = 1;
  return [...cells.map(cell => Number(cell !== '?')), ...cells.map(cell => Number(cell === '#')),
    ...cells.map(cell => Number(cell === 'G')), ...memory.visited, ...agent,
    ...Array.from({ length: 4 }, (_, i) => Number(i === state.direction))];
}
export function expertAction(memory: Memory, state: GameState): number {
  if (state.status === 'success' || state.status === 'timeout' || memory.known[state.y][state.x] === 'G') throw new Error('专家不能在终止状态行动');
  const queue = [{ x: state.x, y: state.y, first: -1 }];
  const seen = new Set([state.y * 7 + state.x]);
  let frontier: typeof queue[number] | undefined;
  let goal: typeof queue[number] | undefined;
  for (let index = 0; index < queue.length; index++) {
    const cell = queue[index];
    if (cell.first !== -1) {
      if (memory.known[cell.y][cell.x] === 'G') { goal = cell; break; }
      if (!frontier && !memory.visited[cell.y * 7 + cell.x]) frontier = cell;
    }
    DIRECTIONS.forEach(([dx, dy], direction) => {
      const x = cell.x + dx, y = cell.y + dy, key = y * 7 + x;
      if (x < 0 || x >= 7 || y < 0 || y >= 7 || seen.has(key) || !['.', 'G'].includes(memory.known[y][x])) return;
      seen.add(key); queue.push({ x, y, first: cell.first === -1 ? direction : cell.first });
    });
  }
  const hasKnownGoal = memory.known.some(row => row.includes('G'));
  const target = hasKnownGoal ? goal : frontier;
  if (!target) throw new Error('记忆中没有可达的探索目标');
  const turn = (target.first - state.direction + 4) % 4;
  return turn === 0 ? 2 : turn === 1 ? 1 : 0;
}
