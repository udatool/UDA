# UDA

This repo is the official code of **UDA: Unified Pre-training for Multi-Architecture Binary Disassembly**. 

## Get Started
### Requirements
- Linux
- Python 3.12.3
- PyTorch 2.3.0
- CUDA 12.1

### Quick Start

#### 1. Create a conda virtual environment, installing PyTorch and activating it.
```
conda create -n uda pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda activate uda
```

#### 2. Install transformers package.
```
pip install transfromers
```

#### 3. Get code and models of HAformer.
```
git clone https://github.com/udatool/UDA.git && cd UDA
```

Download `model_saved.zip` [baidu drive](https://pan.baidu.com/s/1pjWbzjSQJJq7a-fHdlwjGQ?pwd=vf6d) and extract them `./model_saved`. 

```
unzip model_saved.zip
```

