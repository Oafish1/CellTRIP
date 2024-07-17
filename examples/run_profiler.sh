# Profile script and save to file
python -m cProfile -s time -o profile.prof train.py
snakeviz profile.prof
