#!/bin/bash

indir="$1"
outdir="$2"

mkdir -p "$outdir"

export outdir

# Find all regular files (filter later by FFmpeg)
find "$indir" -type f | parallel --bar '
    # Output filename: basename without extension
    base=$(basename "{}")
    base_noext="${base%.*}"

    # Convert to mono 16-bit PCM and split into 2-second segments
    ffmpeg -loglevel error -y -i "{}" -ac 1 -acodec pcm_s16le \
        -f segment -segment_time 2 -reset_timestamps 1 \
        "'"$outdir"'/${base_noext}_%03d.wav"
'

find "$outdir" -name "*.wav" -type f -size -70k -delete
