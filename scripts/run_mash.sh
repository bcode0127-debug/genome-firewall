#!/usr/bin/env bash
# Sketches every genome in the 200-genome MLST subset with mash (s=10000),
# combines into one sketch file, then computes an all-vs-all mash distance
# matrix. Run in the `mash` conda env.
set -eu
ROOT="/Users/pranayp/Desktop/hacknation"
ID_LIST="$ROOT/data/interim/subset200_mlst.csv"
FASTA_DIR="$ROOT/data/interim/fasta"
SKETCH_DIR="$ROOT/data/interim/mash_sketches"
COMBINED="$ROOT/data/interim/mash_sketches/combined.msh"
DIST_OUT="$ROOT/data/interim/mash_dist.tsv"

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate mash

mkdir -p "$SKETCH_DIR"

ids_file=$(mktemp)
tail -n +2 "$ID_LIST" | cut -d, -f1 > "$ids_file"
total=$(wc -l < "$ids_file" | tr -d ' ')
echo "Sketching $total genomes (s=10000)..."

n=0
while read -r id; do
  [ -z "$id" ] && continue
  n=$((n+1))
  fasta="$FASTA_DIR/${id}.fna"
  out="$SKETCH_DIR/${id}"
  if [ -f "${out}.msh" ]; then
    continue
  fi
  if [ ! -s "$fasta" ]; then
    echo "MISSING FASTA: $id" >&2
    continue
  fi
  mash sketch -s 10000 -o "$out" "$fasta" > /dev/null 2>> "$ROOT/data/interim/mash_sketch.log"
  if [ $((n % 25)) -eq 0 ]; then
    echo "sketched $n / $total"
  fi
done < "$ids_file"

echo "Combining sketches..."
mash paste "$SKETCH_DIR/combined" "$SKETCH_DIR"/*.msh

echo "Running all-vs-all mash dist..."
mash dist -s 10000 "$COMBINED" "$COMBINED" > "$DIST_OUT"

rm -f "$ids_file"
lines=$(wc -l < "$DIST_OUT" | tr -d ' ')
echo "DONE: $lines pairwise distances written to $DIST_OUT"
