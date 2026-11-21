import {useNavigate} from '@tanstack/react-router';
import {listModels} from '../apis/models';
import {Go1Page} from '../modules/go1/views/Go1Page';
import {pageLocation} from '../utilities/page-url';
import {validateMazeSearch,type MazeSearch} from '../utilities/search';
import {MazePage} from './MazePage';

function usePageNavigation() {
 const navigate=useNavigate();
 return (next:MazeSearch)=>{void navigate({...pageLocation(next),hash:previous=>previous??''});};
}

export const loadMazeModels=()=>listModels().catch(()=>null);

export function MazeRoutePage({query,tab,initialRuns}:{
 query:Record<string,unknown>;tab?:MazeSearch['tab'];initialRuns:Awaited<ReturnType<typeof loadMazeModels>>;
}) {
 const navigate=usePageNavigation();
 const search=validateMazeSearch({...query,tab});
 return <MazePage initialRuns={initialRuns} search={search} onSearchChange={navigate}/>;
}

export function Go1RoutePage({query,tab}:{query:Record<string,unknown>;tab?:MazeSearch['go1Tab']}) {
 const navigate=usePageNavigation();
 const search=validateMazeSearch({...query,module:'go1',go1Tab:tab});
 return <Go1Page search={search} onChange={navigate}/>;
}
