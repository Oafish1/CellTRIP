# Profile script and save to file
python -m memray run -o memray_profile.bin --force $*
python -m memray flamegraph -o memray_profile.html --force memray_profile.bin
