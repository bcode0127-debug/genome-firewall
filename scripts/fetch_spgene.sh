#!/usr/bin/env bash
# fetch_spgene.sh — bulk download BV-BRC PATRIC spgene.tab annotations for the
# K. pneumoniae cohort. Resumable, 6 parallel workers, logs failures.
set -u

ROOT="/Users/pranayp/Desktop/hacknation"
ID_LIST="$ROOT/data/interim/genome_list.txt"
OUT_DIR="$ROOT/data/interim/spgene"
FAILED="$ROOT/data/interim/failed_ids.txt"

mkdir -p "$OUT_DIR"
: > "$FAILED"          # rebuild failure log fresh each run
export OUT_DIR FAILED

TOTAL=$(grep -cve '^[[:space:]]*$' "$ID_LIST")
echo "Starting fetch of $TOTAL ids with 6 parallel workers..."

fetch_one() {
  id="$1"
  out="$OUT_DIR/${id}.spgene.tab"

  # Resume: skip if already present and >1KB
  if [ -f "$out" ] && [ "$(wc -c < "$out" | tr -d ' ')" -gt 1024 ]; then
    return 0
  fi

  if ! curl --ssl-reqd --user anonymous:guest -f -s \
       --retry 2 --max-time 60 \
       -o "$out" \
       "ftp://ftp.bv-brc.org/genomes/${id}/${id}.PATRIC.spgene.tab"; then
    rm -f "$out"                 # drop any partial file
    echo "$id" >> "$FAILED"      # short append is atomic under O_APPEND
  fi
}
export -f fetch_one

# Single-process progress monitor: count files, print each time we cross a
# 200 boundary. One process => no lock needed. Killed once downloads finish.
(
  last=0
  while :; do
    n=$(find "$OUT_DIR" -name '*.spgene.tab' -type f | wc -l | tr -d ' ')
    if [ "$n" -ge "$((last + 200))" ]; then
      last=$(( (n / 200) * 200 ))
      echo "progress: $n / $TOTAL files present"
    fi
    sleep 5
  done
) &
MON_PID=$!

grep -ve '^[[:space:]]*$' "$ID_LIST" | tr -d '\r' \
  | xargs -P 6 -I {} bash -c 'fetch_one "$@"' _ {}

kill "$MON_PID" 2>/dev/null

succeeded=$(find "$OUT_DIR" -name '*.spgene.tab' -type f | wc -l | tr -d ' ')
failed=$(wc -l < "$FAILED" | tr -d ' ')
bytes=$(find "$OUT_DIR" -name '*.spgene.tab' -type f -print0 \
  | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END {print s+0}')

echo "DONE"
echo "succeeded (files present): $succeeded"
echo "failed: $failed"
echo "total bytes: $bytes"
