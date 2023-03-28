#Python Library
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import yaml
import faiss

#Pytorch
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

#Huggingface
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from dataloader import get_dataloader, get_table_dataloader
from model import BiEncoder
from utils import set_seed, parser_eval, reader_eval


def build_faiss_index(config, args, accelerator):

    seed = int(config["seed"])
    set_seed(seed)

    question_tokenizer = AutoTokenizer.from_pretrained(config["question_encoder"])
    table_tokenizer = AutoTokenizer.from_pretrained(config["table_encoder"])

    if "bert" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]", "[Table name]", "[Header]","[Rows]","[Row]","[sep]"]
        table_tokenizer.add_tokens(special_tokens)
    
    elif "tapas" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]"]
        table_tokenizer.add_tokens(special_tokens)

    model = BiEncoder(config["question_encoder"], config["table_encoder"], question_tokenizer, table_tokenizer)
    model.to(accelerator.device)

    model_path = os.path.join(args.checkpoint_dir,"retriever",config["exp_name"],"pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location = accelerator.device))   

    #question_encoder = model.question_enc
    table_encoder = model.table_enc

    index = faiss.IndexFlatIP(table_encoder.config.hidden_size)

    table_dataloader = get_table_dataloader(config)


    table_encoder, table_dataloader = accelerator.prepare(
        table_encoder, table_dataloader
    ) 

    table_encoder.eval()

    accelerator.print("building faiss index")

    with tqdm(table_dataloader, desc=f"Building Faiss Index", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
        for batch in batch_iterator:
            batch["input_ids"] = batch["input_ids"].to(accelerator.device)
            batch["attention_mask"] = batch["attention_mask"].to(accelerator.device)
            batch["token_type_ids"] = batch["token_type_ids"].to(accelerator.device)

            with torch.no_grad():
                output = table_encoder(**batch).pooler_output
            
            index.add(output.type(torch.float32).cpu().numpy())
    
    return index



def evaluate(config, args, index,accelerator):

    seed = int(config["seed"])
    set_seed(seed)

    question_tokenizer = AutoTokenizer.from_pretrained(config["question_encoder"])
    table_tokenizer = AutoTokenizer.from_pretrained(config["table_encoder"])

    if "bert" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]", "[Table name]", "[Header]","[Rows]","[Row]","[sep]"]
        table_tokenizer.add_tokens(special_tokens)
    
    elif "tapas" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]"]
        table_tokenizer.add_tokens(special_tokens)



    model = BiEncoder(config["question_encoder"], config["table_encoder"], question_tokenizer, table_tokenizer)
    model.to(accelerator.device)


    model_path = os.path.join(args.checkpoint_dir,"retriever",config["exp_name"],"pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location = accelerator.device))   

    question_encoder = model.question_enc
    table_encoder = model.table_enc

    # index = faiss.IndexFlatIP(table_encoder.config.hidden_size)

    _, valid_dataloader, test_dataloader = get_dataloader(question_tokenizer, config)
    table_dataloader = get_table_dataloader(config)

    

    question_encoder, table_encoder, valid_dataloader, test_dataloader,table_dataloader = accelerator.prepare(
        question_encoder, table_encoder, valid_dataloader, test_dataloader ,table_dataloader
    )       
    question_encoder.eval()
    table_encoder.eval()


    # accelerator.print("building faiss index")
    # with tqdm(table_dataloader, desc=f"Building Faiss Index", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
    #     for batch in batch_iterator:
    #         batch["input_ids"] = batch["input_ids"].to(accelerator.device)
    #         batch["attention_mask"] = batch["attention_mask"].to(accelerator.device)
    #         batch["token_type_ids"] = batch["token_type_ids"].to(accelerator.device)

    #         with torch.no_grad():
    #             output = table_encoder(**batch).pooler_output
            
    #         index.add(output.type(torch.float32).cpu().numpy())
    
    acc = [0,0,0,0]
    with tqdm(valid_dataloader, desc=f"Validating", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
        for batch in batch_iterator:
            question_tokenized = batch["question_tokenized"]
            num_batch = question_tokenized["input_ids"].size(0)
            question_tokenized = question_tokenized.to(accelerator.device)
            hard_positive_idx = batch["hard_positive_idx"]
            positive_idx = batch["positive_idx"]
            
            with torch.no_grad():
                output = question_encoder(**question_tokenized).pooler_output
            
            _, indicies = index.search(output.type(torch.float32).cpu().numpy().reshape(num_batch,-1),50)

            for i in range(num_batch):
                if "parser" == config["method"]:
                    acc = parser_eval([a+1 for a in indicies[i]], hard_positive_idx[i], positive_idx[i], acc)
                  
                elif "reader" == config["method"]:
                    acc = reader_eval([a+1 for a in indicies[i]], hard_positive_idx[i], positive_idx[i], acc)
            batch_iterator.set_postfix(top5 = acc[0], top10 = acc[1], top20 = acc[2], top50 = acc[3])
    
    accelerator.print(config["exp_name"])
    total = len(valid_dataloader.dataset)
    acc = [i/total for i in acc]
    accelerator.print(acc)    




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type = str
    )
    parser.add_argument(
        "--fp16",
        action = "store_true"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type = str,
        default = "./checkpoint/"
    )
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    if args.fp16:
        mixed_precision = "fp16"
    else:
        mixed_precision = "no"
    accelerator = Accelerator(cpu = False, mixed_precision = mixed_precision)



    index_file_path = os.path.join(config["path"],"augmented",config["method"]+"_"+config["table_encoder"]+".index")

    if not os.path.exists(index_file_path):
        index = build_faiss_index(config, args, accelerator)
        faiss.write_index(index, index_file_path)
    else:
        index = faiss.read_index(index_file_path)


    evaluate(config, args, index, accelerator)


if __name__=="__main__":
    main()