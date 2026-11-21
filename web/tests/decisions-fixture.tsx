// Product-owned component interaction fixture; never imported by application routes.
import {createRoot} from 'react-dom/client';
import {flushSync} from 'react-dom';
import {Decisions} from '../src/components/maze/Decisions';
import {startGame} from '../src/utilities/game';
const game=startGame('fixed-map-v1');
const identities=[{},{}];
let root:ReturnType<typeof createRoot>;
export function renderPrediction({values,pending,steps,identity=0}:{values:number[]|null;pending:boolean;steps:number;identity?:number}) {
  if(!root) {const container=document.createElement('div');container.id='prediction-fixture';document.body.replaceChildren(container);root=createRoot(container);}
  flushSync(()=>root.render(<Decisions state={{...game.state,steps}} mode="model" values={values} maze={game.maze} course="fixed-map-v1" memory={null} previousAction={-1} pending={pending} predictionIdentity={identities[identity]}/>));
}
