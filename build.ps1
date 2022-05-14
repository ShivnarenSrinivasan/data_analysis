py -m build
py -m twine upload --skip-existing dist/* --verbose
start-sleep 60
py -m pip install --upgrade data-analysis-helpers
