# How to setup Pytorch and fastai library

In this we will try to set up our pytorch environment using conda.

# Create a virtual environment

**using miniconda**

write following command in your comman prompt or terminal

`conda create --name torch python=3.7`


# activate and add jupyter

to add jupyter notebooks in your virtual environment use the following commands:

`conda activate torch`

`conda install nb_conda`

# register environment on jupyter

to add your environment in jupyter notebooks kernel use:

`python -m ipykernel install --user --name pytorch --display-name "Python 3.7 (pytorch)"`

# install 	cpu only pytroch (latest)

`conda install pytorch -c pytorch` 

# install fastai2 and torchvision (latest)

`conda install -c fastai -c pytorch fastai`

# install dependencies

activate and go to this address

`(torch) C:location_of_conda_envs\conda\pkgs`

# now just use pip 

`pip install ipywidgets`

# To add Voila

`pip install voila`

`jupyter serverextension enable voila --sys-prefix`

# to make voila working in all directories

`conda install -c conda-forge jupyter`

`conda install notebook`

Hope this will helps.