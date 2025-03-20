# Profile script and save to file
python -m cProfile -o time_profile.prof $1
snakeviz time_profile.prof --hostname 0.0.0.0 --server
