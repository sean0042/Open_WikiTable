# Open-WikiTable :Dataset for Open Domain Question Answering with Complex Reasoning over Table

Despite recent interest in open domain question answering (ODQA) over tables, many studies still rely on datasets that are not truly optimal for the task with respect to utilizing structural nature of table. These datasets assume answers reside as a single cell value and do not necessitate exploring over multiple cells such as aggregation, comparison, and sorting. Thus, we release Open-WikiTable, the first ODQA dataset that requires complex reasoning over tables. Open-WikiTable is built upon [WikiSQL](https://github.com/salesforce/WikiSQL) and [WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) to be applicable in the open-domain setting. As each question is coupled with both textual answers and SQL queries, Open-WikiTable opens up a wide range of possibilities for future research, as both reader and parser methods can be applied. 

The dataset is released along with our paper titled Open-WikiTable :Dataset for Open Domain Question Answering with Complex Reasoning over Table (2023 ACL Findings). For further details, please refer to our paper.



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

## Retriever Training
```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun \
--nnodes=1 \
--nproc_per_node=4 \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:1111 \
./src/retriever_train.py \
--config_path ./config/retriever/reader_bert_bert.yaml \
--fp16
```

## Retriever Evaluating
```
CUDA_VISIBLE_DEVICES=0 \
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:1111 \
./src/retriever_train.py \
--config_path ./config/retriever/reader_bert_bert.yaml \
--fp16
```

## Generator Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--nnodes=1 \
--nproc_per_node=4 \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:1111 \
./src/generator_train.py \
--config_path ./config/generator/parser_t5_5.yaml \
```

## Generator Evaluating
```
CUDA_VISIBLE_DEVICES=0 \
torchrun \
--nnodes=1 \
--nproc_per_node=1 \
--rdzv_backend=c10d \
--rdzv_endpoint=localhost:1111 \
./src/generator_eval.py \
--config_path ./config/generator/parser_t5_5.yaml \
```