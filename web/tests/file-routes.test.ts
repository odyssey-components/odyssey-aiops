import {expect,test} from 'bun:test';
import {go1Sections,legacyPageLocation,mazeSections} from '../src/utilities/page-url';
import {validateMazeSearch} from '../src/utilities/search';

test('legacy destinations map to registered concrete file routes and retain query state',async()=>{
 const tree=await Bun.file(new URL('../src/routeTree.gen.ts',import.meta.url)).text();
 for(const [group,sections] of [['go1',go1Sections],['maze',mazeSections]] as const){
  for(const slug of Object.values(sections)){
   expect(tree).toContain(`'/${group}/${slug}'`);
  }
 }
 expect(tree).not.toContain('/go1/$section');
 expect(tree).not.toContain('/maze/{-$section}');
 expect(legacyPageLocation({module:'go1',go1Tab:'recording',keep:'yes'})).toEqual({to:'/go1/test-cases',search:{keep:'yes'}});
 expect(legacyPageLocation({})).toEqual({to:'/maze',search:{}});
 const id='aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';
 expect(validateMazeSearch({module:'go1',go1Scene:id}).go1Scene).toBe(id);
});

test('home and the Maze parent route redirect to Go1 training by default', async()=>{
 const {isRedirect}=await import('@tanstack/react-router');
 const {Route:home}=await import('../src/routes/index');
 const {Route:maze}=await import('../src/routes/maze');
 for(const [route,search] of [[home,{}],[home,{tab:'simulation'}],[maze,{}]] as const){
  let result:unknown;
  try{
   route.options.beforeLoad?.({search,location:{}} as never);
  }catch(error){
   result=error;
  }
  expect(isRedirect(result) && result.options.to).toBe('/go1/training');
 }
 const tree=await Bun.file(new URL('../src/routeTree.gen.ts',import.meta.url)).text();
 expect(tree.match(/getParentRoute: \(\) => MazeRoute,/g)).toHaveLength(Object.keys(mazeSections).length+1);
});
