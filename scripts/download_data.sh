#!/usr/bin/env bash
# Download the Bio-ML (OAEI 2024) OMIM-ORDO task from Zenodo record 13119437
# and extract it into data/raw/bio-ml/omim-ordo/.
#
# Usage: bash scripts/download_data.sh
set -euo pipefail

RECORD="13119437"
TASK="omim-ordo"
URL="https://zenodo.org/api/records/${RECORD}/files/${TASK}.zip/content"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/raw/bio-ml"
ZIP="${DEST}/${TASK}.zip"

mkdir -p "${DEST}"

if [ -d "${DEST}/${TASK}" ] && [ -n "$(ls -A "${DEST}/${TASK}" 2>/dev/null)" ]; then
  echo "Already present: ${DEST}/${TASK} (delete it to re-download)"
  exit 0
fi

echo "Downloading ${URL}"
if ! curl -fL --retry 3 -o "${ZIP}" "${URL}"; then
  echo "ERROR: download failed. Download ${TASK}.zip manually from"
  echo "  https://zenodo.org/records/${RECORD}"
  echo "and extract it into ${DEST}/${TASK}/"
  exit 1
fi

# Extract into a fresh directory; the archive may or may not have a top-level folder.
TMP="$(mktemp -d "${DEST}/extract.XXXXXX")"
unzip -q "${ZIP}" -d "${TMP}"
if [ -d "${TMP}/${TASK}" ]; then
  mv "${TMP}/${TASK}" "${DEST}/${TASK}"
else
  mv "${TMP}" "${DEST}/${TASK}"
fi
rm -rf "${TMP}" "${ZIP}"

echo "Extracted to ${DEST}/${TASK}"
find "${DEST}/${TASK}" -type f | sort
