# LFADS
# """ Warning : THIS CODE WILL RUN ONLY ON A LINUX SYSTEM AS THIS CODE USES JAX """ 
## Install Conda 
```
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh
```
## Create Conda Enviorment 
```
conda create --name LFADS
conda activate LFADS
```

## Install Modules 

```
conda install numpy 
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax[cpu]==0.2.27
pip install h5py
pip install -U scikit-learn
pip install opencv-python
```

## Follow Demo Video to change Paramter in all Files
## If you want to use the default data set then run the following 

### Training
```
python lfads.py
```
### Inference 
```
python lfads_inference.py
```
