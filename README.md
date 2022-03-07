# DeepLearning

Materials for my Bayesian Data Analysis course

## Setting up the python environment

In this course you will be working with python using jupyter notebooks or jupyter lab. So first you have to set up a proper python environment. I strongly encourage you to use some form of a virtual environment. I recommend the [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or its smaller subset [miniconda](https://docs.conda.io/en/latest/miniconda.html). Personally I recommend using 
[mambaforge](https://github.com/conda-forge/miniforge#mambaforge). 
After installing `mambaforge` create a new virtual environment `bda` (or any other name you want):

```
conda create -n bda python=3.9
```
Then activate the environment  by running
```
conda activate bda
```
Now you can install required packages (if you are using Anaconda some maybe already installed):

```
mamba install  jupyterlab jupytext myst-nb
mamba install numpy scipy  matplotlib
```
If you didn't install `mamba` then you can subsitute `conda` for `mamba`. I tend to use `mamba` as it is markedly faster then `conda`.  
After installing all required packages you can start `jupyter lab` by running 
```
jypyter lab
```

## MyST format

The notebooks in the repository are stored in [MyST (Markedly Structured Text Format)](https://myst-parser.readthedocs.io/en/latest/) format. Thanks to the `jupytext` package you can open them right in the jupyter lab, by clicking the file name with righthand mouse button and choosing `open with` and then `Notebook`. If you are using jupyter notebook the you have to convert them prior to opening by running   
```shell
jupytext --to notebook <md file name>
```

## Using python in lab

When using the computers in lab, please log to your linux account and then run
```
source /app/Python/3.9.7/VE/defaults/bin/activate
```
The you can run 
```
jupyter lab
```

