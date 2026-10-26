#!/usr/bin/env bash
set -euo pipefail
# The official Milvus Standalone Compose file contains etcd + MinIO + Milvus.
# Pin a tested 2.6 release; override MILVUS_VERSION deliberately on upgrades.
MILVUS_VERSION="${MILVUS_VERSION:-v2.6.25}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE="${SCRIPT_DIR}/milvus-standalone.yml"
URL="https://github.com/milvus-io/milvus/releases/download/${MILVUS_VERSION}/milvus-standalone-docker-compose.yml"
if [ ! -f "$FILE" ]; then
  echo "Downloading official Milvus ${MILVUS_VERSION} compose from ${URL}"
  curl --fail --location --retry 3 "$URL" -o "$FILE"
fi
docker compose -f "$FILE" up -d
echo "Milvus target: http://localhost:19530; Web UI: http://localhost:9091/webui/"
