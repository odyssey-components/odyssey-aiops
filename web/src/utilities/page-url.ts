import {validateMazeSearch,type MazeSearch} from './search';

export const mazeSections = {
 training:'training',datasets:'datasets',models:'models',recordings:'recordings',
 playback:'playback',reinjection:'reinjection',simulation:'simulations',
} as const satisfies Record<NonNullable<MazeSearch['tab']>,string>;

export const go1Sections = {
 training:'training',models:'models',scenes:'scenes',replay:'replay',recording:'test-cases',
 'test-case-sets':'test-case-sets',simulation:'simulations',
 'vision-injection':'vision-injection',
} as const satisfies Record<NonNullable<MazeSearch['go1Tab']>|'training',string>;

export function pageSearch(raw:Record<string,unknown>) {
 const {module:_,tab:__,go1Tab:___,modelError:____,go1Device:_____,...search}=raw;
 return search;
}

export function pageLocation(state:MazeSearch,query:Record<string,unknown>=state) {
 const search=pageSearch(query);
 const to=state.module==='go1'
  ?`/go1/${go1Sections[state.go1Tab??'training']}` as const
  :state.tab?`/maze/${mazeSections[state.tab]}` as const:'/maze' as const;
 return {to,search};
}

export const legacyPageLocation=(raw:Record<string,unknown>)=>pageLocation(validateMazeSearch(raw),raw);
