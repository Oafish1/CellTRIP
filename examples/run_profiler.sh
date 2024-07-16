# Run and record to file
# python -u train.py 2>&1 | tee output.txt

# Run with arguments
# python -u train.py 5

# Profile script and save to file
python -m cProfile -s time -o profile.prof train.py
snakeviz profile.prof
