ffmpeg -i "$1" \
    -vf "fps=5,scale=-1:1080:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 -y "${1%.*}.gif"
