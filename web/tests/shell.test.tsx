import { expect, test } from 'bun:test';
import { renderToStaticMarkup } from 'react-dom/server';
import { MazeShell } from '../src/components/layout/MazeShell';
import { ProAdminProvider } from '@odyssey/ui/patterns/admin-provider';

test('course workspace renders shared shell with primary maze and secondary courses', () => {
  const html=renderToStaticMarkup(<ProAdminProvider appearance={{storageKey:'odymaze-web-appearance'}}><MazeShell course="random-map-v1" onCourse={()=>{}}><p>experiment body</p></MazeShell></ProAdminProvider>);
  expect(html).toContain('data-od-slot="pro-admin-shell"');
  expect(html).toContain('第二课：随机地图');
  expect(html).toContain('第一课：固定地图');
  expect(html).toMatch(/<button[^>]*title="Maze"[^>]*><span>Maze<\/span><\/button>/);
  expect(html).not.toContain('data-lesson=');
  expect(html).toMatch(/<button[^>]*title="第二课：随机地图"[^>]*aria-current="page"[^>]*><span>第二课：随机地图<\/span><\/button>/);
  expect(html).toContain('experiment body');
  expect(html).toContain('data-od-slot="pro-admin-theme-menu-trigger"');
});


test('Go1 shell hides Maze navigation by default',()=>{
 const html=renderToStaticMarkup(<ProAdminProvider appearance={{storageKey:'odymaze-web-appearance'}}><MazeShell module="go1" course="fixed-map-v1" onCourse={()=>{}}><p>Go1 body</p></MazeShell></ProAdminProvider>);
 expect(html).toContain('Go1');expect(html).not.toContain('title="Maze"');expect(html).not.toContain('第一课：固定地图');
});
