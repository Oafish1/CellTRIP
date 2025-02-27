# Regular compile
# python -m piptools compile
# Developer compile
# python -m piptools compile -r requirements-dev.in
# All compile
python -m pip freeze > requirements-dev.txt
