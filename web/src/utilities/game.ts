import { bestAction, expertAction, HEADINGS, initialState, predictRecurrent, probabilities, step, terminal, modelLesson, type LessonId, type Model } from "../../core";
import fixedMaze from "../../maze.json";
import { generateMaze } from "../lessons/random_map_v1";

import * as local from "../lessons/local_map_v1";

export type Mode = "manual" | "expert" | "model";

export function startGame(lesson: LessonId = "fixed-map-v1", seed = 7) {
  const maze = (lesson === "local-map-v1" || lesson === "memory-map-v1") ? local.generateMaze(seed) : lesson === "random-map-v1" ? generateMaze(seed) : fixedMaze;
  return {
    lesson, maze, seed,
    episode: {},
    actions: [] as number[],
    state: initialState(maze),
    memory: (lesson === "local-map-v1" || lesson === "memory-map-v1") ? local.initialMemory(maze, initialState(maze)) : null as local.Memory | null,
    recurrent: null as {hidden:number[];values:number[];observations:number} | null,
    previousAction: -1,
    mode: "manual" as Mode,
    model: null as Model | null,
    modelBackend: "json" as "json" | "onnx",
    prediction: null as number[] | null,
    inferenceRequest: null as object | null,
    inferenceError: null as string | null,
    running: false,
    loading: false,
    speed: 650,
    visited: new Set([`${maze.start.x},${maze.start.y}`]),
    log: [] as { step: number; action: number; description: string }[],
  };
}

type Game = ReturnType<typeof startGame>;
function prime(game: Game): Game {
  if (game.modelBackend === "onnx") return {...game,recurrent:null,prediction:null,inferenceRequest:game.model ? {} : null,inferenceError:null};
  if (game.lesson !== "memory-map-v1" || !game.model) return {...game,recurrent:null};
  const result=predictRecurrent(game.model,game.maze,game.state,Array<number>(32).fill(0),game.previousAction);
  return {...game,recurrent:{hidden:result.hidden,values:result.probabilities,observations:1}};
}

type Command =
  | { type: "configure"; lesson:LessonId; seed:number; mode:Mode; speed:number; clearModel?:boolean }
  | { type: "course"; lesson: LessonId }
  | { type: "maze"; seed: number }
  | { type: "step"; action?: number }
  | { type: "mode"; mode: Mode }
  | { type: "speed"; speed: number }
  | { type: "loaded"; model: Model; mode?:Mode; external?:boolean }
  | { type: "prediction"; request:object; values:number[]; hidden:number[] | null }
  | { type: "prediction-failed"; request:object; message:string }
  | { type: "tick" | "toggle" | "stop" | "reset" | "loading" | "failed" | "forget" };

export function updateGame(game: Game, command: Command): Game {
  switch (command.type) {
    case "configure": {
      const changed=command.lesson!==game.lesson || command.seed!==game.seed;
      const base=changed ? {...startGame(command.lesson,command.seed),model:command.lesson===game.lesson ? game.model : null,modelBackend:game.modelBackend} : {...game,running:false,loading:false};
      const configured={...base,model:command.clearModel ? null : base.model,speed:command.speed,mode:command.mode==="model" && (!base.model || command.clearModel) ? "manual" as Mode : command.mode};
      return changed || command.clearModel ? prime(configured) : configured;
    }
    case "course":
      return { ...startGame(command.lesson, game.seed), speed: game.speed };
    case "maze":
      if (game.lesson === "fixed-map-v1") return game;
      return prime({ ...startGame(game.lesson, command.seed), mode: game.mode, model: game.model, modelBackend: game.modelBackend, speed: game.speed });
    case "mode":
      return game.loading || (command.mode === "model" && (!game.model || (game.modelBackend === "onnx" && (!game.prediction || !!game.inferenceError)))) ? game : { ...game, mode: command.mode, running: false };
    case "speed":
      return { ...game, speed: command.speed };
    case "stop":
      return game.running ? { ...game, running: false } : game;
    case "toggle":
      if (game.running) return { ...game, running: false };
      return game.loading || (game.modelBackend === "onnx" && game.model && (!game.prediction || !!game.inferenceError || !!game.inferenceRequest)) || game.mode === "manual" || terminal(game.state) ? game : { ...game, running: !game.running };
    case "reset":
      return prime({ ...startGame(game.lesson, game.seed), mode: game.mode, model: game.model, modelBackend: game.modelBackend, speed: game.speed });
    case "loading":
      return { ...game, loading: true, running: false };
    case "loaded":
      return prime({ ...startGame(modelLesson(command.model), game.seed), mode: command.mode ?? "model", model: command.model, modelBackend: command.external ? "onnx" : "json", speed: game.speed });
    case "forget":
      return game.lesson === "memory-map-v1" ? prime({...game,running:false}) : game;
    case "prediction": {
      if (game.modelBackend !== "onnx" || !game.model || command.request !== game.inferenceRequest) return game;
      const validValues=command.values.length===3 && Array.from(command.values).every(value=>Number.isFinite(value) && value>=0 && value<=1) && Math.abs(command.values.reduce((a,b)=>a+b,0)-1)<1e-5;
      const validHidden=game.lesson!=="memory-map-v1" || (command.hidden?.length===32 && Array.from(command.hidden).every(Number.isFinite));
      if (!validValues || !validHidden) return updateGame(game,{type:"prediction-failed",request:command.request,message:"模型推理结果无效"});
      const values=[...command.values];
      return {...game,prediction:values,recurrent:game.lesson==="memory-map-v1" ? {hidden:[...command.hidden!],values,observations:(game.recurrent?.observations ?? 0)+1} : null,inferenceRequest:null,inferenceError:null};
    }
    case "prediction-failed":
      return game.modelBackend === "onnx" && game.model && command.request===game.inferenceRequest
        ? {...game,running:false,inferenceRequest:null,prediction:null,inferenceError:command.message} : game;
    case "failed":
      return { ...game, loading: false };
    case "tick":
    case "step": {
      if (game.loading || (game.modelBackend === "onnx" && game.model && (!!game.inferenceRequest || !!game.inferenceError || !game.prediction)) || terminal(game.state) || (command.type === "tick") !== game.running) return game;
      const before = game.state;
      const action = game.mode === "expert" ? (game.memory ? local.expertAction(game.memory, before) : expertAction(game.maze, before))
        : game.mode === "model" && game.model ? bestAction(game.modelBackend === "onnx" ? game.prediction! : game.recurrent ? game.recurrent.values : probabilities(game.model, game.maze, before, game.memory))
        : command.type === "step" ? command.action : undefined;
      if (action === undefined) return game;
      const state = step(game.maze, before, action);
      const description = action !== 2 ? `朝${HEADINGS[state.direction]}`
        : before.x === state.x && before.y === state.y ? "撞墙，位置未改变"
        : `(${before.x}, ${before.y}) → (${state.x}, ${state.y})`;
      const external=game.modelBackend === "onnx" && !!game.model;
      const result = !external && game.recurrent && game.model ? predictRecurrent(game.model,game.maze,state,game.recurrent.hidden,action) : null;
      return {
        ...game, state, previousAction:action, prediction:external ? null : game.prediction,inferenceRequest:external ? {} : game.inferenceRequest,inferenceError:null, recurrent: external ? game.recurrent : result ? {hidden:result.hidden,values:result.probabilities,observations:game.recurrent!.observations+1} : null, memory: game.memory ? local.observe(game.maze, state, game.memory) : null, running: game.running && !terminal(state),
        visited: new Set([...game.visited, `${state.x},${state.y}`]),
        actions: [...game.actions,action],
        log: [...game.log.slice(-7), { step: state.steps, action, description }],
      };
    }
  }
}
