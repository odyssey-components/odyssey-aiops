import {createFileRoute} from '@tanstack/react-router';
import {pageSearch} from '../utilities/page-url';
import {MazeRoutePage,loadMazeModels} from '../views/PageRoutes';

export const Route=createFileRoute('/maze/training')({
 validateSearch:pageSearch,
 loader:loadMazeModels,
 component:Page,
});

function Page() {
 return <MazeRoutePage query={Route.useSearch()} tab="training" initialRuns={Route.useLoaderData()}/>;
}
