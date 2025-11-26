#!/bin/bash

# Do not use set -e -u globally; instead, handle undefined variables locally
# set -uo pipefail

# Only assign src/dst if not already set in the environment
if [[ -z "${src:-}" ]]; then
    src="${1:-}"
fi
if [[ -z "${dst:-}" ]]; then
    dst="${2:-}"
fi

if [[ -z "$src" || -z "$dst" ]]; then
    echo "Usage: source transfer_wavs.sh <source_dir> <target_dir>"
    return 1  # use 'return' so sourcing doesn't exit the shell
fi

# Absolute path for robustness
src="$(realpath "$src")"
mkdir -p "$dst"
export src dst

copy_file() {
    file="$1"
    file="$(realpath "$file")"

    rel="${file#$src/}"
    first_level_dir="${rel%%/*}"
    target_dir="$dst/$first_level_dir"
    mkdir -p "$target_dir"

    filename=$(basename "$file")
    out="$target_dir/$filename"

    if [[ -e "$out" ]]; then
        n=1
        base="${filename%.*}"
        ext="${filename##*.}"
        while [[ -e "$target_dir/${base}_$n.$ext" ]]; do
            ((n++))
        done
        out="$target_dir/${base}_$n.$ext"
    fi

    rsync -a --no-i-r --whole-file "$file" "$out"
}

export -f copy_file

jobs=$(nproc --all)

trap 'echo; echo "Interrupted by user. Stopping..."; return 0' SIGINT

find "$src" -mindepth 1 -type f -iname "*.wav" -print0 \
| parallel -0 -j "$jobs" --line-buffer --halt soon,fail=1 --bar copy_file {}
