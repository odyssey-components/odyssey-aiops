export function apiPath(path: string): string {
  return `${(import.meta.env?.BASE_URL ?? "/").slice(0, -1)}${path}`;
}
