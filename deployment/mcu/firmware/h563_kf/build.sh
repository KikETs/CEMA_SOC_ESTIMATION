#!/usr/bin/env bash
set -euo pipefail

configuration="${1:-Debug}"
case "${configuration}" in
  Debug|Release) ;;
  *)
    echo "Usage: $0 [Debug|Release]" >&2
    exit 2
    ;;
esac

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(mktemp -d /tmp/stm32h563zi-ai-kf-workspace.XXXXXX)"
trap 'rm -rf "${workspace}"' EXIT

"${STM32CUBEIDE_HEADLESS_BUILD:-/opt/st/stm32cubeide_2.2.0/headless-build.sh}" \
  -data "${workspace}" \
  -import "${project_dir}" \
  -cleanBuild "STM32H563ZI_AI_KF_BENCH/${configuration}" \
  -no-indexer \
  -printErrorMarkers
