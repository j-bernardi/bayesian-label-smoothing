# USAGE:
# Needs to be converted from dos, not changed in commit yet
# Then run with source, e.g.:
#   apt-get install dos2unix && dos2unix -n first_time_setup.sh unix_compat_setup.sh && source unix_compat_setup.sh
#	source first_time_setup.sh
# 
# If conversion not run, expect to receive one of errors:
#    '\r': command not found
#    could not locate package libxtst6

# Machine requirements
apt update
apt-get update
apt-get upgrade
apt-get -y install wget git unzip vim libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Anaconda
# Check for latest version by navigating to https://repo.anaconda.com/archive/
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ./anaconda.sh
sh anaconda.sh -b -p $HOME/anaconda3
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda init
source $HOME/.bashrc
rm anaconda.sh  # save space

# Set up conda env with requirements
# TODO - a venv might save space
conda create -n py36-gpu python=3.6 -y
conda activate py36-gpu
conda install -y tensorflow-gpu==2.1.0 matplotlib pandas
conda clean -a -y # make space for data

# Set up training repo
git clone https://github.com/j-bernardi/bayesian-label-smoothing.git
cd bayesian-label-smoothing
source set_python_path_lin.sh

# Test
python main.py test sample

echo "Manually download and scp the zipfile from:"
echo "https://www.kaggle.com/farhanhubble/multimnistm2nist"
echo "Then run unzip data/download.zip -d data/"
echo "See readme if space saving required"
