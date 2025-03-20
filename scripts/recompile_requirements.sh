# All compile
python -m pip freeze -r requirements.in | sed '/@/d' > ../requirements.txt
