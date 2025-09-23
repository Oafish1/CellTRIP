# Must python -m ipykernel install --user --name $2 to recognize conda envs
jupyter nbconvert --execute --to notebook "$1" --inplace --allow-errors --ExecutePreprocessor.kernel_name="$2" --ExecutePreprocessor.timeout=-1
