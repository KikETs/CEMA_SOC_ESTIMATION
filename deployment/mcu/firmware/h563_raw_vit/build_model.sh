#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model_id>" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
drop_dir="$(cd "${project_dir}/../.." && pwd)"
model_id="$1"

"${PYTHON:-python}" \
  "${drop_dir}/scripts/install_raw_vit_model.py" "${model_id}"
make -C "${project_dir}" MODEL_ID="${model_id}" -j"$(nproc)"
