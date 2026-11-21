#!/usr/bin/env bash
set -euo pipefail
deploy=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
product=$(CDPATH='' cd -- "$deploy/../.." && pwd)
root=$(cd "$product" && bun -e 'import { root } from "./scripts/workspace.mjs"; console.log(root)')
mkdir -p "$root/tmp"
cd "$root"
work=$(mktemp -d "$root/tmp/aiops-image-build.XXXXXX")
source=${ODYSSEY_GO1_SOURCE:-"$HOME/.odyssey/go1/source"}
commit=f4fb0d12ac68842c1567bf1de6f771ce8e6671b5

test "$(git -C "$source" rev-parse HEAD)" = "$commit"
git -C "$source" diff --quiet HEAD -- unitree_go1
test -z "$(git -C "$source" ls-files --others --exclude-standard -- unitree_go1)"
mkdir -p "$work/api" "$work/worker/go1-source/.git" "$work/web"
cp -a "$source/unitree_go1" "$work/worker/go1-source/"
for name in HEAD index objects refs packed-refs; do
  if [[ -e "$source/.git/$name" ]]; then
    cp -a "$source/.git/$name" "$work/worker/go1-source/.git/"
  fi
done

uv pip install --python "$root/.venv/bin/python" --target "$work/worker/site-packages" --torch-backend cpu \
  "$root/repositories/odyssey-python/python/config" "$root/repositories/odyssey-python/python/training" "$product/training/go1[cpu,simulation]" "$product/training/maze" \
  torch==2.14.1 jax==0.12.0.1 jaxlib==0.12.0.1 mujoco==3.17.0 mujoco-mjx==3.17.0 \
  numpy==2.5.4 pydantic==2.14.3 mcap==1.6.0 PyOpenGL==3.1.10 onnx==1.23.2
cd "$product/server"
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o "$work/api/aiops-server" ./cmd/aiops-server
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o "$work/worker/aiops-worker" ./cmd/aiops-worker
(
  cd "$product/web"
  AIOPS_WEB_BASE=/aiops/ bun run build --outDir "$work/web/dist"
)
cp "$deploy/nginx.conf" "$work/web/"
docker build --platform linux/amd64 -f "$deploy/api.Dockerfile" -t aiops-api:release "$work/api"
docker build --platform linux/amd64 --network host -f "$deploy/worker.Dockerfile" -t aiops-worker:release "$work/worker"
docker build --platform linux/amd64 -f "$deploy/web.Dockerfile" -t aiops-web:release "$work/web"
