import { root } from './workspace.mjs';
import {createHash} from 'node:crypto';
import {access,readFile} from 'node:fs/promises';
import {join,resolve,basename} from 'node:path';
import {parseArgs} from 'node:util';

const {values}=parseArgs({args:process.argv.slice(2),options:{release:{type:'string'},venv:{type:'string'},python:{type:'string'},'torch-index':{type:'string'},offline:{type:'boolean'},project:{type:'string'},device:{type:'string'}}});
if(!values.release)throw new Error('请指定 --release <发行目录>');
const project=values.project??'maze';
const device=values.device??'cpu';
if(!['cpu','cuda','rocm'].includes(device))throw new Error('--device 必须为 cpu、cuda 或 rocm');
if(!['maze','go1'].includes(project))throw new Error('--project 必须为 maze 或 go1');
const release=resolve(values.release),venv=resolve(values.venv??join(root,'.venv'));
const python=join(venv,process.platform==='win32'?'Scripts/python.exe':'bin/python');
async function exists(path:string) {try{await access(path);return true;}catch{return false;}}
async function run(cmd:string[]) {const child=Bun.spawn(cmd,{stdout:'inherit',stderr:'inherit'});const code=await child.exited;if(code!==0)throw new Error(`${cmd[0]} failed (${code})`);}
const manifest=JSON.parse(await readFile(join(release,'release.json'),'utf8')) as {format:string;packages:{filename:string;sha256:string}[]};
if(manifest.format!=='aiops-local-release/v1'||manifest.packages.length!==4)throw new Error('发行包清单无效');
const wheels:string[]=[];
for(const item of manifest.packages) {
 if(basename(item.filename)!==item.filename||!item.filename.endsWith('.whl'))throw new Error('wheel文件名无效');
 const path=join(release,'wheels',item.filename);
 if(createHash('sha256').update(await readFile(path)).digest('hex')!==item.sha256)throw new Error('wheel校验失败');
 if(!item.filename.startsWith(project==='maze'?'aiops_go1_training-':'aiops_maze_training-'))wheels.push(item.filename.startsWith('aiops_go1_training-')?`${path}[${device==='cuda'?'gpu':'cpu'}]`:path);
}
if(!await exists(python)) {
 if(await exists(venv))throw new Error('目标目录已存在且不是可用Python环境，请更换 --venv');
 await run(['uv','venv','--python',values.python??'3.14',venv,...(values.offline?['--offline']:[])]);
}
await run(['uv','pip','install','--python',python,...(values.offline?['--offline']:[]),...(values['torch-index']?['--index',values['torch-index']]:[]),
 '--reinstall-package',`aiops-${project}-training`,'--reinstall-package','odylab-training','--reinstall-package','odyssey-config',...wheels]);
if(project==='maze')await run([python,'-I','-c',`import importlib.metadata as m;import aiops_maze_training;from aiops_maze_training.lessons.fixed_map_v1.environment import MAZE_PATH,load_maze;print('installed:',m.version('aiops-maze-training'));print('package:',aiops_maze_training.__file__);assert MAZE_PATH.is_file();assert 'resources' in MAZE_PATH.parts;assert load_maze()['rows']`]);
else await run([python,'-I','-c',`import importlib.metadata as m;import aiops_go1_training;print('installed:',m.version('aiops-go1-training'));print('package:',aiops_go1_training.__file__)`]);
console.log(`训练 Python：${python}`);
