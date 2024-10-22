# Profile script and save to file
python -m cProfile -s time -o profile.prof $1
snakeviz profile.prof
