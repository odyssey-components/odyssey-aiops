import type { GameState, Maze } from '../../core';
export function generateMaze(seed: number): Maze {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('种子必须是 0 到 4294967295 的整数');
  let state = seed;
  const random = () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
  const rows = Array.from({length:7}, () => Array<string>(7).fill('#'));
  for (let y=1; y<6; y++) for (let x=1; x<6; x++) rows[y][x] = random() < .28 ? '#' : '.';
  let x=1, y=1;
  while (x<5 || y<5) {
    rows[y][x]='.';
    if (x===5) y++; else if (y===5 || random()<.5) x++; else y++;
  }
  rows[5][5]='G';
  return {id:`odymaze-random-${seed}`,rows:rows.map(row=>row.join('')),start:{x:1,y:1,direction:0},max_steps:100};
}
export function validMaze(value: unknown): value is Maze {
  const maze = value as Maze;
  if (!maze || !Array.isArray(maze.rows) || maze.rows.length!==7 || maze.max_steps!==100
    || maze.start?.x!==1 || maze.start.y!==1 || maze.start.direction!==0
    || typeof maze.id!=='string' || !/^odymaze-random-\d+$/.test(maze.id)) return false;
  const seed = Number(maze.id.slice('odymaze-random-'.length));
  if (!Number.isInteger(seed) || seed<0 || seed>0xffffffff) return false;
  return maze.rows.every((row,y)=>typeof row==='string' && row.length===7 && [...row].every((cell,x)=>
    (x===0 || y===0 || x===6 || y===6) ? cell==='#' : x===5 && y===5 ? cell==='G' : cell==='.' || cell==='#'))
    && JSON.stringify(maze.rows)===JSON.stringify(generateMaze(seed).rows);
}
export function encode(maze: Maze, state: GameState): number[] {
  const walls = maze.rows.flatMap(row=>[...row].map(cell=>Number(cell==='#')));
  const goal = maze.rows.flatMap(row=>[...row].map(cell=>Number(cell==='G')));
  const agent = Array<number>(49).fill(0); agent[state.y*7+state.x]=1;
  return [...walls,...goal,...agent,...Array.from({length:4},(_,i)=>Number(i===state.direction))];
}
