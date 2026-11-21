import type { GameState, Maze } from '../../core';
export function encode(maze: Maze, state: GameState): number[] {
  const width = maze.rows[0].length, height = maze.rows.length;
  const result = Array(width + height + 4).fill(0);
  result[state.x] = 1;
  result[width + state.y] = 1;
  result[width + height + state.direction] = 1;
  return result;
}
