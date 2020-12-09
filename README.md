# bayesian-label-smoothing
A repository to prove the Bayesian Label Smoothing theory proposed by my Master's thesis

# Setup

## Auto setup (servers)
`first_time_setup.sh`

A script that sets up a fresh GPU server with unix packages.

It installs anaconda and creates an environment with the required python packages. Finally it installs [this git repo](https://github.com/j-bernardi/bayesian-label-smoothing.git) and downloads the [data](https://www.kaggle.com/farhanhubble/multimnistm2nist) from the Kaggle servers.

## Manual setup

### Clone project
Get the project:
```bash
git clone https://github.com/j-bernardi/bayesian-label-smoothing.git
cd bayesian-label-smoothing
bash set_python_path_lin.sh
```

### Get data
If you want the full dataset, use this to get it from kaggle (or download manually from [here](https://www.kaggle.com/farhanhubble/multimnistm2nist))
```bash
mkdir data
wget 'https://storage.googleapis.com/kaggle-data-sets/37151/56512/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201128%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201128T001219Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=a48ea7d1276375093c6492e606a0b05a5b4e95843141ab8285dd3a528dd797f37b6c8afd66804ad1f1a8edd3cb3f783c59f73b537023bac3964072876f565ba38dee432cdeb917caca85a93b7a52503c327963f63e91bd2da5685b02628095bccfbb3589be45f5e188c0c8ec4564f1c46d1d63e530a65f157a7dfdab840dda83e936ac5f01aa97d325c4f29ff9c5f8f6c895fd82dffdfe32f5c5cd48b187fcc122ad332e8ca6b33a3f5ba6dfb2fd5ce637aada2967a55673318f7e08eb6917f6c6d8f691b9e34bfdaae3df79eaaaccaa79457b368bcce9a4d433a2f671c8eb91158b922533dade9664345e6dc2a431d6bcdbd78cdcb48d6652b9e18238fc56d1' -O data/download.zip
unzip data/download.zip -d data/
```

### Conda env
Here are the required packages for this project
```bash
conda create -n py36-gpu python=3.6 -y
conda activate py36-gpu
conda install -y tensorflow-gpu==2.1.0 matplotlib
conda clean -a -y # make space for data
```

# Usage

```bash
python main.py config_dir {sample|N}
```

`config_dir`: A directory, optionally existing and containing a file named `config.py`. If not existing, the config in `defaults/config.py` will be used.

`sample|N`: If the string "sample", the data from `sample_data/` will be used (10 examples). If an integer N, full data is loaded from `data/` (see `first_time_setup.sh`), and sliced `[:N]`.

# Aims

I investigate the use of a Bayesian prior on the human-induced boundary labelling error, to produce a better-informed smoothed label in terms of magnitude and spread.

I aim to improve the training process for image segmentation tasks, achieving more precise segmentation boundaries as a result, which are essential in the [original medical context](https://deepmind.com/research/publications/clinically-applicable-diagnosis-and-referral-retinal-disease).

Results pending.