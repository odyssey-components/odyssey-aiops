import type { GameState, Maze } from '../../core';
export { generateMaze, validMaze } from './local_map_v1';

export function encode(maze: Maze, state: GameState, previousAction = -1): number[] {
  if (![-1, 0, 1, 2].includes(previousAction)) throw new Error('上一动作无效');
  const cells: string[] = [];
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    cells.push(maze.rows[state.y + dy]?.[state.x + dx] ?? '#');
  }
  const oneHot = (length: number, value: number) => Array.from({ length }, (_, i) => Number(i === value));
  return [...cells.map(cell => Number(cell === '#')), ...cells.map(cell => Number(cell === 'G')),
    ...oneHot(7, state.x), ...oneHot(7, state.y), ...oneHot(4, state.direction), ...oneHot(4, previousAction + 1)];
}
