import {createFileRoute} from '@tanstack/react-router';
import {SessionGate} from '../access/session';

export const Route=createFileRoute('/login')({
 component:()=> <SessionGate loginRoute/>,
});
