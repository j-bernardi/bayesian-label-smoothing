# bayesian-label-smoothing
A repository to prove the Bayesian Label Smoothing theory proposed by my Master's thesis

# Setup

## Auto setup (servers)
```bash
apt-get dos2unix
dos2unix -n first_time_setup.sh unix_compat_setup.sh
source unix_compat_setup.sh
```

A script that sets up a fresh GPU server with unix packages. Written in dos, so convert to unix (removes return carriage \r)

It installs anaconda and creates an environment with the required python packages. Finally it installs [this git repo](https://github.com/j-bernardi/bayesian-label-smoothing.git) and downloads the [data](https://www.kaggle.com/farhanhubble/multimnistm2nist) from the Kaggle servers.

## Manual setup

### Clone project
Get the project:
```bash
git clone https://github.com/j-bernardi/bayesian-label-smoothing.git
cd bayesian-label-smoothing
bash set_python_path_lin.sh
```

### Conda env
Here are the required packages for this project
```bash
conda create -n py36-gpu python=3.6 -y
conda activate py36-gpu
conda install -y tensorflow-gpu==2.1.0 matplotlib
conda clean -a -y # make space for data
```

### Get data

There is 10 data examples in `sample_data/`. Presuming you want the full dataset, log in to kaggle.com and [download from here](https://www.kaggle.com/farhanhubble/multimnistm2nist)) into a file called `data/download.zip`. Compressed, it is only 16MB, so it can be uploaded to a server relatively easily. Then:
```bash
unzip data/download.zip -d data/
```

Unzipped, this is about 2.5GB, stored as int64. Note to save loading time and space on disk:

All the x labels are only image data in range [0, 255] and y labels have 11 classes one-hot encoded in the last dimension, making it around 11x larger. Therefor uint8 is sufficient to represent all the data. See `data/downgrade.py` to save space if needed. Note default tensorflow models are float32, so it will need upgrading at some point.

# Usage

```bash
python main.py config_dir {sample|N}
```

`config_dir`: A directory, optionally existing and containing a file named `config.py`. If not existing, the config in `defaults/config.py` will be used.

`sample|N`: If the string "sample", the data from `sample_data/` will be used (10 examples). If an integer N, full data is loaded from `data/` (see `first_time_setup.sh`), and sliced `[:N]`.

# Aims

I investigate the use of a Bayesian prior on the human-induced boundary labelling error, to produce a better-informed smoothed label in terms of magnitude and spread.

I aim to improve the training process for image segmentation tasks, achieving more precise segmentation boundaries as a result, which are essential in the [original medical context](https://deepmind.com/research/publications/clinically-applicable-diagnosis-and-referral-retinal-disease).

# Experiment log

[Google docs link](https://docs.google.com/document/d/1qQnq6UoZ1EMvqMxJtW3bz45_LgJuF5qJooZ_jsB_YoM/edit?usp=sharing)

## Preliminary results
![A bar graph of preliminary results](https://github.com/j-bernardi/bayesian-label-smoothing/blob/main/results/smoothing/plot_round_1.png?raw=true)

The preliminary results for 5 random weight initialisation trainings (same data order).

Observations:
- We can see some exciting results for the average class accuracy excluding the background (e.g. target classes)
  - weighted uniform smoothing has an advantage over the uniform prior, thus far
  - weighted adjacent does not have the same advantage - the parameter choice might not be optimal
- It appears fixed-magnitude adjacent smoothing may have too large a smoothing magnitude. Smoothing parameter tuning pending.
