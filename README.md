# Open-WikiTable

Explanation

## Requirements and Installation
If you want to get the same result with the paper, please follow below 


- python == 3.8
- CUDA == 11.3
- pytorch == 1.12
- faiss == 1.7
- transformers == 4.24
- accelerate == 0.14
- SQLite3 == 3.34

```
git clone https://github.com/sean0042/Open-WikiTable.git
cd Open-WikiTable
conda create -n openwikitable python=3.8
conda activate openwikitable
pip install transformers
pip install accelerate
pip install tabulate
```

## Dataset
you can get the dataset by below command
```
cd data
tar -xvzf data.tar.gz
cd ..

```
