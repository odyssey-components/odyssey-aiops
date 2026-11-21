import { ProAdminProvider, AppearanceDocumentSync } from "@odyssey/ui/patterns/admin-provider";
import { createRootRoute, Outlet, useRouterState } from "@tanstack/react-router";
import { appearance } from "../appearance";
import { SessionGate } from "../access/session";
import "../../styles.css";

function Root() {
  const loginRoute = useRouterState({select: state => state.matches.some(match => match.routeId === "/login")});
  return <ProAdminProvider appearance={{ storageKey: appearance.storageKey, defaultColorScheme: "system", defaultDensity: "standard" }}>
    <AppearanceDocumentSync {...appearance} />
    {loginRoute ? <Outlet /> : <SessionGate><Outlet /></SessionGate>}
  </ProAdminProvider>;
}

export const Route = createRootRoute({component: Root});
