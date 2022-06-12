# first time setup in repo
gh repo clone Electrocyte/weather
cd weather
pipenv install pandas jupyter seaborn matplotlib
pipenv shell # activate [VE] virtual environment for jupyter notebook
git add . 
git commit
git push

pipenv install statsmodels
git add . 
git commit
git push


# new computer
gh repo clone Electrocyte/weather
pipenv install
pipenv shell

# to run
pipenv shell
jupyter-notebook

# separate installation per computer
pipenv shell
pipenv install
python -m pip install git+https://github.com/statsmodels/statsmodels
