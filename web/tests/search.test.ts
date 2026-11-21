import {expect,test} from 'bun:test';
import {validateMazeSearch} from '../src/utilities/search';
test('training tab is restorable and invalid tab values fall back to experiment',()=>{
  expect(validateMazeSearch({tab:'training'}).tab).toBe('training');
  expect(validateMazeSearch({tab:'future'}).tab).toBeUndefined();
});
test('URL state validates scalar fields and preserves unrelated values',()=>{
  expect(validateMazeSearch({course:'random-map-v1',point:'5',seed:'4294967295',mode:'expert',speed:'40',model:'run-7',other:'keep'})).toEqual({course:'random-map-v1',point:5,seed:4294967295,mode:'expert',speed:40,model:'run-7',other:'keep'});
});
test('invalid URL values safely default without array or blank coercion',()=>{
  for(const value of [[],['2'],'',NaN,Infinity,null,{}]) {
    const actual=validateMazeSearch({course:value,point:value,seed:value,mode:value,speed:value,model:value});
    expect({...actual,modelError:undefined}).toEqual({course:'fixed-map-v1',point:1,seed:7,mode:'manual',speed:650,model:undefined,modelError:undefined});
  }
  expect(validateMazeSearch({model:'../secret',course:'unknown',seed:-1}).model).toBeUndefined();
});

test('default model identity accepts Unicode safe basenames matching read-only API',()=>{
  expect(validateMazeSearch({model:'实验一 2026'}).model).toBe('实验一 2026');
  for(const model of ['.','..','a/b','a\\b','a\0b','x'.repeat(201)]) expect(validateMazeSearch({model}).model).toBeUndefined();
});

test('dataset workspace restores through URL without losing course',()=>{
  expect(validateMazeSearch({tab:'datasets',course:'local-map-v1'}).tab).toBe('datasets');
  expect(validateMazeSearch({tab:'datasets',course:'local-map-v1'}).course).toBe('local-map-v1');
});

test('dataset details accept UUID only within datasets workspace',()=>{
  const dataset='12345678-1234-1234-1234-123456789abc';
  expect(validateMazeSearch({tab:'datasets',dataset}).dataset).toBe(dataset);
  for(const value of ['../secret','',[],{},'not-found']) expect(validateMazeSearch({tab:'datasets',dataset:value}).dataset).toBeUndefined();
  expect(validateMazeSearch({tab:'training',dataset}).dataset).toBeUndefined();
});

test('training task details restore only valid IDs within the training workspace',()=>{
  const task='12345678-1234-1234-1234-123456789abc';
  expect(validateMazeSearch({tab:'training',task}).task).toBe(task);
  expect(validateMazeSearch({tab:'training',task:'../job'}).task).toBeUndefined();
  expect(validateMazeSearch({tab:'datasets',task}).task).toBeUndefined();
});

test('models workspace restores with course and excludes other detail IDs',()=>{
  const value=validateMazeSearch({tab:'models',course:'memory-map-v1',dataset:'12345678-1234-1234-1234-123456789abc',task:'12345678-1234-1234-1234-123456789abc'});
  expect(value.tab).toBe('models');expect(value.course).toBe('memory-map-v1');
  expect(value.dataset).toBeUndefined();expect(value.task).toBeUndefined();
});


test('model details keep active model independent and accept only scoped safe basenames',()=>{
 expect(validateMazeSearch({tab:'models',model:'active',modelDetail:'另一个模型'}).modelDetail).toBe('另一个模型');
 expect(validateMazeSearch({tab:'models',model:'active',modelDetail:'other'}).model).toBe('active');
 for(const modelDetail of ['../secret','a/b','a\\b','.',[],{},'x'.repeat(201)]) expect(validateMazeSearch({tab:'models',modelDetail}).modelDetail).toBeUndefined();
 expect(validateMazeSearch({tab:'datasets',modelDetail:'other'}).modelDetail).toBeUndefined();
});


test('module URL preserves old Maze defaults and isolates Go1 IDs',()=>{
 const id='12345678-1234-1234-1234-123456789abc';
 expect(validateMazeSearch({}).module).toBeUndefined();
 expect(validateMazeSearch({module:'go1',go1Tab:'models',go1Model:id}).go1Model).toBe(id);
 expect(validateMazeSearch({module:'go1',go1Task:'../bad'}).go1Task).toBeUndefined();
 expect(validateMazeSearch({module:'maze',go1Task:id,go1Tab:'models'}).go1Task).toBeUndefined();
});

test('Go1 simulation detail restores only within simulation and replay pages',()=>{
 const id='12345678-1234-1234-1234-123456789abc';
 expect(validateMazeSearch({module:'go1',go1Tab:'simulation',go1Simulation:id}).go1Simulation).toBe(id);
 expect(validateMazeSearch({module:'go1',go1Tab:'replay',go1Simulation:id}).go1Simulation).toBe(id);
 expect(validateMazeSearch({module:'go1',go1Tab:'models',go1Simulation:id}).go1Simulation).toBeUndefined();
 expect(validateMazeSearch({module:'go1',go1Tab:'simulation',go1Simulation:'../escape'}).go1Simulation).toBeUndefined();
});

test('Go1 backend choice restores from URL and rejects unsupported values',()=>{
 expect(validateMazeSearch({module:'go1',go1Device:'rocm'}).go1Device).toBeUndefined();
 expect(validateMazeSearch({module:'go1',go1Device:'auto'}).go1Device).toBeUndefined();
 expect(validateMazeSearch({go1Device:'cuda'}).go1Device).toBeUndefined();
});

test('Go1 replay dialog ID is scoped to models and simulations',()=>{
 const id='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
 const go1Simulation='bbbbbbbb-cccc-dddd-eeee-ffffffffffff';
 expect(validateMazeSearch({module:'go1',go1Tab:'simulation',go1Simulation,go1Replay:id})).toMatchObject({go1Simulation,go1Replay:id});
 expect(validateMazeSearch({module:'go1',go1Tab:'models',go1Replay:id}).go1Replay).toBe(id);
 expect(validateMazeSearch({module:'go1',go1Tab:'simulation',go1Replay:'../bad'}).go1Replay).toBeUndefined();
});
