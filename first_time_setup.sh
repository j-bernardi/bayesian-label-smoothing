# Machine requirements
apt-get install wget git unzip vim libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y

# Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda.sh
bash anaconda.sh -b -p $HOME/anaconda3
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda init
source $HOME/.bashrc
rm anaconda.sh  # save space

# Set up conda env with requirements
# TODO - a venv might save space
conda create -n py36-gpu python=3.6 -y
conda activate py36-gpu
conda install -y tensorflow-gpu==2.1.0 matplotlib
conda clean -a -y # make space for data

# Set up training repo
git clone https://github.com/j-bernardi/bayesian-label-smoothing.git
cd bayesian-label-smoothing
bash set_python_path_lin.sh

# Fetch and unzip data where expected
mkdir data
wget 'https://storage.googleapis.com/kaggle-data-sets/37151/56512/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201128%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201128T001219Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=a48ea7d1276375093c6492e606a0b05a5b4e95843141ab8285dd3a528dd797f37b6c8afd66804ad1f1a8edd3cb3f783c59f73b537023bac3964072876f565ba38dee432cdeb917caca85a93b7a52503c327963f63e91bd2da5685b02628095bccfbb3589be45f5e188c0c8ec4564f1c46d1d63e530a65f157a7dfdab840dda83e936ac5f01aa97d325c4f29ff9c5f8f6c895fd82dffdfe32f5c5cd48b187fcc122ad332e8ca6b33a3f5ba6dfb2fd5ce637aada2967a55673318f7e08eb6917f6c6d8f691b9e34bfdaae3df79eaaaccaa79457b368bcce9a4d433a2f671c8eb91158b922533dade9664345e6dc2a431d6bcdbd78cdcb48d6652b9e18238fc56d1' -O data/download.zip
unzip data/download.zip -d data/

# Test
python main.py test 50
