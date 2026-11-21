import {createFileRoute} from '@tanstack/react-router';
import {pageSearch} from '../utilities/page-url';
import {MazeRoutePage,loadMazeModels} from '../views/PageRoutes';

export const Route=createFileRoute('/maze/')({
 validateSearch:pageSearch,
 loader:loadMazeModels,
 component:Page,
});

function Page() {
 return <MazeRoutePage query={Route.useSearch()} initialRuns={Route.useLoaderData()}/>;
}
