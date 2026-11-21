import { existsSync } from "node:fs";
import { basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const product = dirname(dirname(fileURLToPath(import.meta.url)));
const container = dirname(product);
const parent = dirname(container);
const parentWorkspace = fileURLToPath(new URL("../../../go.work", import.meta.url));
export const root = basename(container) === "repositories" && existsSync(parentWorkspace) ? parent : product;
