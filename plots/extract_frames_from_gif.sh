# Usage `sh extract_frames_from_gif.sh <file> <destination_dir>`
mkdir -p "${1%.*}"
ffmpeg -i "$1" "${1%.*}/frame_%d.png"
