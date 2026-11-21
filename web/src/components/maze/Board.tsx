import { HEADINGS, type GameState, type Maze } from "../../../core";
import type { Memory } from "../../lessons/local_map_v1";

export function Board({ state, visited, maze, memory, id = "board" }: { state: GameState; visited: Set<string>; maze: Maze; memory?: Memory | null; id?: string }) {
  const rows = memory?.known ?? maze.rows;
  const goalY = rows.findIndex(row => row.includes("G"));
  const goalX = goalY < 0 ? -1 : rows[goalY].indexOf("G");
  return <div className="board-wrap">
    <div id={id} className="maze-board" role="img" aria-label={`迷宫：角色位于 (${state.x}, ${state.y})，朝向${HEADINGS[state.direction]}，${goalY < 0 ? "出口尚未发现" : `已发现出口 (${goalX}, ${goalY})`}`}
      style={{ gridTemplateColumns: `repeat(${maze.rows[0].length}, minmax(0, 1fr))` }}>
      {rows.flatMap((row, y) => [...row].map((cell, x) => {
        const key = `${x},${y}`;
        const classes = ["tile", cell === "?" ? "unknown" : cell === "#" ? "wall" : "floor"];
        if (memory && Math.abs(x-state.x)<=1 && Math.abs(y-state.y)<=1) classes.push("in-view");
        if (visited.has(key)) classes.push("visited");
        if (cell === "G") classes.push("goal");
        if (x === maze.start.x && y === maze.start.y && cell !== "G") classes.push("start");
        if (x === state.x && y === state.y) classes.push("occupied");
        return <div key={key} className={classes.join(" ")} data-cell={cell}>
          {cell === "?" && <span className="fog-label" aria-hidden="true">?</span>}
          {cell === "G" && <span className="goal-label">G</span>}
          {x === state.x && y === state.y && <span className="agent">
            <svg className="agent-direction" viewBox="0 0 24 24" aria-hidden="true" focusable="false"
              style={{ transform: `rotate(${state.direction * 90}deg)` }}>
              <path d="M3 3L21 12L3 21L8 12Z" fill="currentColor" />
            </svg>
          </span>}
        </div>;
      }))}
    </div>
  </div>;
}
