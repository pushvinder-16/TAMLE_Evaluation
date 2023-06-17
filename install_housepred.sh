# !/bin/bash
# Usage
# `bash install_housepred.sh -u <relative path to the whl file>` installs house prediction in an env called `mle-dev` (creates new env if an env with this name doesn't exist) and installs TigerNLP and its dependencies in the env. To be used by the end user
# `bash install_housepred.sh -d` installs house prediction in an env called `tmle-dev` (creates new env if an env with this name doesn't exist) and installs TigerNLP and its dependencies (along with the dev dependencies) in the env. To be used by the developer

# conda env remove -n ta-tigernlp
eval "$(conda shell.bash hook)"
if { conda env list | grep 'mle-dev$'; } then
    echo "mle-dev env exists already. house prediction package and dependencies will be installed in the existing env"
    conda activate mle-dev
else
    echo "Creating a new env mle-dev where house prediction package and dependencies will be installed"
    conda create -n mle-dev python=3.8 -y
    conda activate mle-dev
fi

# Conda install HDBSCAN package as pip install hdbscan has gcc incompatibility issues
conda install -c conda-forge hdbscan==0.8.29 -y

# If user options are provided
while [[ $# -gt 0 ]]; do
option=$1
case "$option" in
    -u|--user)
        # If user option, then install using whl file
        pip install $2
        shift
        shift
        ;;
    -d|--dev)
        # If dev option, install using setup.py and also install developer dependency packages
        pip install -e .
        pip install -r deploy/requirements.txt
    shift
    ;;
    esac
done