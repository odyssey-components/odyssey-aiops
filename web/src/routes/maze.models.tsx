import {createFileRoute} from '@tanstack/react-router';
import {pageSearch} from '../utilities/page-url';
import {MazeRoutePage,loadMazeModels} from '../views/PageRoutes';

export const Route=createFileRoute('/maze/models')({
 validateSearch:pageSearch,
 loader:loadMazeModels,
 component:Page,
});

function Page() {
 return <MazeRoutePage query={Route.useSearch()} tab="models" initialRuns={Route.useLoaderData()}/>;
}
