#!/usr/bin/env bash
# Use the station's existing ffprobe. Do not install tools or change container networking.
set -eu
(( $# )) || exec docker exec ms3_videostreaming ffprobe
args=("$@")
url=${args[$#-1]}
if [[ "$url" == http*/pipeline/record/* ]]; then
    # Some MS3 versions do not support the range requests ffprobe needs to seek an MP4.
    exec docker exec ms3_videostreaming sh -ec '
        url=$1; shift
        clip=$(mktemp /tmp/capacity-clip.XXXXXXXX)
        trap '\''rm -f -- "$clip"'\'' EXIT
        curl --fail --silent --show-error --max-time 30 --output "$clip" "$url"
        ffprobe "$@" "$clip"
    ' capacity-clip "$url" "${@:1:$#-1}"
fi
if [[ "$url" == rtsp://host.docker.internal:8554/* && -n "${CAPACITY_MTX_HOST:-}" ]]; then
    for arg in "$@"; do
        if [[ "$arg" == -read_intervals ]]; then
            url="rtsp://${CAPACITY_MTX_HOST}:8554/${url#rtsp://host.docker.internal:8554/}"
            break
        fi
    done
fi
exec docker exec ms3_videostreaming ffprobe "${@:1:$#-1}" "$url"
