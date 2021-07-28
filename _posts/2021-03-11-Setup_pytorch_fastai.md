# How to setup Pytorch and fastai library

In this we will try to set up our pytorch environment using conda.

This will help you to install PyTorch for both CPU and GPU. 

First, install Miniconda. Miniconda is the minimal set of features of Anaconda python distribution.
Download it from the [here](https://docs.conda.io/en/latest/miniconda.html).

![Miniconda install](/images/minconda_install.png)

Now, we will use only miniconda for the packages installation.

1. **Create a virtual environment.**

    write following command in your comman prompt or terminal

    `conda create --name torch python=3.7`

2. **Activate and add jupyter**

    To add jupyter notebooks in your virtual environment use the following commands:

    `conda activate torch`

    `conda install nb_conda`

3. **Register environment on jupyter**

    To add your environment in jupyter notebooks kernel use:

    `python -m ipykernel install --user --name pytorch --display-name "Python 3.7 (pytorch)"`

4. **Install pytorch (CPU or GPU)**

    For installing cpu only pytroch (latest)

    `conda install pytorch -c pytorch`

    Or

    For installing GPU supported pytorch (latest)

    `conda install pytorch cudatoolkit -c pytorch`

5. **Install fastai2 and torchvision (latest)**

    `conda install -c fastai -c pytorch fastai`

6. **Install dependencies**

    Activate and go to this address

    `(torch) C:location_of_conda_envs\conda\pkgs`

    now, just use pip 

    `pip install ipywidgets`

7. **To add Voila support** [optional]

    `pip install voila`

    `jupyter serverextension enable voila --sys-prefix`

8. **To make voila working in all directories** [optional]

    `conda install -c conda-forge jupyter`

    `conda install notebook`

Hope this will helps.
