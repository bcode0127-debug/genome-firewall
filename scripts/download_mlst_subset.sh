#!/usr/bin/env bash
# Downloads FASTAs for the 200-genome MLST validation subset into
# data/interim/fasta/, skipping any already present (e.g. the earlier
# 20-genome AMRFinderPlus subset). Resumable: safe to re-run.
set -u
ROOT="/Users/pranayp/Desktop/hacknation"
ID_LIST="$ROOT/data/interim/subset200_mlst.csv"
OUT_DIR="$ROOT/data/interim/fasta"
FAILED="$ROOT/data/interim/fasta_download_failed.txt"
LOG="$ROOT/data/interim/fasta_download.log"

mkdir -p "$OUT_DIR"
: > "$FAILED"
: > "$LOG"

total=$(tail -n +2 "$ID_LIST" | wc -l | tr -d ' ')
echo "Starting download of $total ids (skipping any already present)..." | tee -a "$LOG"

n=0
ok=0
skip=0
fail=0
tail -n +2 "$ID_LIST" | cut -d, -f1 | while read -r id; do
  [ -z "$id" ] && continue
  n=$((n+1))
  out="$OUT_DIR/${id}.fna"
  if [ -f "$out" ] && [ "$(wc -c < "$out" | tr -d ' ')" -gt 1024 ]; then
    echo "SKIP $id" >> "$LOG"
    continue
  fi
  if curl --ssl-reqd --user anonymous:guest -f -s --retry 2 --max-time 120 \
       -o "$out" \
       "ftp://ftp.bv-brc.org/genomes/${id}/${id}.fna"; then
    echo "OK $id" >> "$LOG"
  else
    rm -f "$out"
    echo "$id" >> "$FAILED"
    echo "FAIL $id" >> "$LOG"
  fi
  if [ $((n % 25)) -eq 0 ]; then
    echo "progress: $n / $total processed" >> "$LOG"
  fi
done

echo "DONE" >> "$LOG"
present=$(find "$OUT_DIR" -name '*.fna' -type f | wc -l | tr -d ' ')
failed=$(wc -l < "$FAILED" | tr -d ' ')
echo "files present: $present, failed: $failed" >> "$LOG"
