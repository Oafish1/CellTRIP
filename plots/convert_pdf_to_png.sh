# pdftoppm
# pdftoppm "$1" "${1%.*}" -png -f 1 -singlefile -r 600 -transp
# ImageMagick (for alpha)
convert -density 600 "$1" -quality 90 "${1%.*}".png
