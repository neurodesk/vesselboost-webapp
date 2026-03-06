#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${1:-web/js/app/config.js}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

VERSION="$(sed -nE "s/^export const VERSION = '([^']+)';$/\1/p" "${CONFIG_FILE}" | head -n1)"

if [[ -z "${VERSION}" ]]; then
  echo "Failed to parse VERSION from ${CONFIG_FILE}" >&2
  exit 1
fi

echo "${VERSION}"
