# first time setup in repo
gh repo clone Electrocyte/weather
cd weather
pipenv install pandas jupyter seaborn matplotlib sklearn common-utils common
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

OR

cd weather/
code .

# separate installation per computer
pipenv shell
pipenv install
python -m pip install git+https://github.com/statsmodels/statsmodels



# make new repo, cd to d/Github/
gh repo clone Electrocyte/perlin-noise
cd perlin-noise/
pipenv install pandas jupyter seaborn matplotlib noise Pillow
git add . 
git commit -m 'packages installed'
git push


cd /mnt/d/GitHub/
gh repo clone Electrocyte/NLP-training
cd /mnt/d/GitHub/NLP-training/
pipenv install pandas numpy jupyter sklearn seaborn matplotlib tensorflow h5py scipy
pipenv install PIL lr_utils 
pipenv install planar_utils
git add . 
git commit -m "install libs"
git push
pipenv shell
jupyter-notebook