# Delete existing compiles
rm -r celltrip/{**,.}/{*.c,*.so}
# Recompile
python setup.py build_ext -if
