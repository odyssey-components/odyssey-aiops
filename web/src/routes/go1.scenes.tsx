import {createFileRoute} from '@tanstack/react-router';
import {pageSearch} from '../utilities/page-url';
import {Go1RoutePage} from '../views/PageRoutes';

export const Route=createFileRoute('/go1/scenes')({
 validateSearch:pageSearch,
 component:Page,
});

function Page() {
 return <Go1RoutePage query={Route.useSearch()} tab="scenes"/>;
}
