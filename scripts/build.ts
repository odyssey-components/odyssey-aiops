import { product, root } from './workspace.mjs';
import {cp,mkdir,readdir,readFile,writeFile,rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join,resolve} from 'node:path';
import {parseArgs} from 'node:util';

const {values}=parseArgs({args:process.argv.slice(2),options:{output:{type:'string'},'wheels-only':{type:'boolean'},offline:{type:'boolean'}}});
const output=resolve(values.output??join(root,'tmp/aiops-release'));
await mkdir(output,{recursive:true});
if((await readdir(output)).length)throw new Error('发行目录必须为空，请指定新的 --output 目录');
const wheels=join(output,'wheels'),stage=join(output,'.build/training');
await mkdir(wheels,{recursive:true});await mkdir(stage,{recursive:true});
async function run(cmd:string[],cwd=root) {
 const child=Bun.spawn(cmd,{cwd,stdout:'inherit',stderr:'inherit'});const code=await child.exited;
 if(code!==0)throw new Error(`${cmd[0]} failed (${code})`);
}
const buildFlags=values.offline?['--offline']:[];
await cp(join(product,'training/maze/src'),join(stage,'src'),{recursive:true,filter:path=>!path.includes('__pycache__')&&!path.endsWith('.pyc')});
await writeFile(join(stage,'pyproject.toml'),await readFile(join(product,'training/maze/pyproject.toml'),'utf8')+'\n[tool.uv.workspace]\nmembers = []\n');
await mkdir(join(stage,'src/aiops_maze_training/resources'),{recursive:true});
await cp(join(product,'web/maze.json'),join(stage,'src/aiops_maze_training/resources/fixed_maze.json'));
await run(['uv','build','--wheel','--no-sources',...buildFlags,'--out-dir',wheels,stage]);
const sharedPython=join(root,'repositories/odyssey-python/python');
for(const project of [join(sharedPython,'config'),join(sharedPython,'training'),join(product,'training/go1')]) {
 await run(['uv','build','--wheel','--no-sources',...buildFlags,'--out-dir',wheels,project]);
}
if(!values['wheels-only']) {
 await run(['bun','run','build'],join(product,'web'));
 await cp(join(product,'web/dist'),join(output,'web'),{recursive:true});
 await cp(join(product,'deploy/nginx.conf'),join(output,'nginx.conf'));
 await run(['go','build','-o',join(output,process.platform==='win32'?'aiops-server.exe':'aiops-server'),'./cmd/aiops-server'],join(product,'server'));
}
const packages=[];
for(const filename of (await readdir(wheels)).filter(name=>name.endsWith('.whl')).sort()) {
 packages.push({filename,sha256:createHash('sha256').update(await readFile(join(wheels,filename))).digest('hex')});
}
await writeFile(join(output,'release.json'),JSON.stringify({format:'aiops-local-release/v1',createdAt:new Date().toISOString(),packages,frontend:values['wheels-only']?null:'web',server:values['wheels-only']?null:'aiops-server'},null,2)+'\n');
await rm(join(output,'.build'),{recursive:true,force:true});
console.log(`发行目录：${output}`);
