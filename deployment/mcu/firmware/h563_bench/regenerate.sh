#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage_root="${CEMA_CUBEMX_STAGE_ROOT:-${HOME}/STM32H563ZI_AI_KF_BENCH_GENERATED}"
stage_project="${stage_root}/STM32H563ZI_AI_KF_BENCH"
ascii_ioc="${CEMA_CUBEMX_ASCII_IOC:-${HOME}/STM32H563ZI_AI_KF_BENCH.ioc}"
cubemx_root="${STM32CUBEMX_ROOT:-${HOME}/STM32CubeMX}"
command_file="$(mktemp)"

cleanup() {
  rm -f "${ascii_ioc}"
  rm -f "${command_file}"
  rm -rf "${stage_root}"
}
trap cleanup EXIT

ln -sfn "${project_dir}/STM32H563ZI_AI_KF_BENCH.ioc" "${ascii_ioc}"
rm -rf "${stage_root}"
mkdir -p "${stage_root}"
sed \
  -e "s|@IOC@|${ascii_ioc}|g" \
  -e "s|@PROJECT_ROOT@|${stage_root}|g" \
  "${project_dir}/cubemx_generate.txt" > "${command_file}"

DISPLAY="${DISPLAY:-:0}" \
XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}" \
"${cubemx_root}/jre/bin/java" \
  -jar "${cubemx_root}/STM32CubeMX" \
  -q "${command_file}"

rsync -a "${stage_project}/" "${project_dir}/"
