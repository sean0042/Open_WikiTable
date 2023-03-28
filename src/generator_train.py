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
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from model import FiDT5, FiDBart, BiEncoder
from utils import set_seed
from dataloader import get_dataloader




def trainer(config, args, index, accelerator):

    seed = int(config["seed"])
    lr = float(config["lr"])
    num_epochs = int(config["num_epochs"])

    set_seed(seed)

    best_loss = 10000

    flattened_table = pd.read_json(os.path.join(config["path"],"splitted_tables.json"))
    flattened_table = np.char.array(flattened_table["flattened"].values,unicode=True)

    #tokenizer
    question_tokenizer = AutoTokenizer.from_pretrained(config["question_encoder"])
    table_tokenizer = AutoTokenizer.from_pretrained(config["table_encoder"])
    generator_tokenizer = AutoTokenizer.from_pretrained(config["language_model"])

    
    if "bert" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]", "[Table name]", "[Header]","[Rows]","[Row]","[sep]"]
        table_tokenizer.add_tokens(special_tokens)
        special_tokens = special_tokens+["[Question]"]
        generator_tokenizer.add_tokens(special_tokens,special_tokens=True)
    
    elif "tapas" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]"]
        table_tokenizer.add_tokens(special_tokens)
        special_tokens = special_tokens+["[Question]"]
        generator_tokenizer.add_tokens(special_tokens,special_tokens=True)
    
    model = BiEncoder(config["question_encoder"], config["table_encoder"], question_tokenizer, table_tokenizer)
    model.to(accelerator.device)

    if "t5" in config["language_model"]:
        t5 = AutoModelForSeq2SeqLM.from_pretrained(config["language_model"])
        t5.resize_token_embeddings(len(generator_tokenizer))
        FiD = FiDT5(t5.config)
        FiD.load_t5(t5.state_dict())
        FiD.to(accelerator.device)
    else:
        bart = AutoModelForSeq2SeqLM.from_pretrained(config["language_model"])
        bart.resize_token_embeddings(len(generator_tokenizer))
        FiD = FiDBart(bart.config)
        FiD.load_bart(bart.model.state_dict())
        FiD.to(accelerator.device)



    

    model_path = os.path.join(args.checkpoint_dir,"retriever",config["retriever_exp"],"pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location = accelerator.device))  

    question_encoder = model.question_enc

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(question_tokenizer, config)

    optimizer = AdamW(params = FiD.parameters(), lr = lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps = (len(train_dataloader) * num_epochs)
    )

    question_encoder, FiD, optimizer ,train_dataloader, valid_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        question_encoder, FiD, optimizer ,train_dataloader, valid_dataloader, test_dataloader, lr_scheduler
    )       

    accelerator.wait_for_everyone()


    question_encoder.eval()
    
    for epoch in range(num_epochs):
        FiD.train()
        with tqdm(train_dataloader, desc=f"Epoch {str(epoch+1).rjust(2)} Training", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
            for batch in batch_iterator:
                question_tokenized = batch["question_tokenized"]
                question_tokenized = question_tokenized.to(accelerator.device)

                num_batch = question_tokenized["input_ids"].size(0)
                top_k = config["top_k"]

                with torch.no_grad():
                    output = question_encoder(**question_tokenized).pooler_output
                
                _, indicies = index.search(output.type(torch.float32).cpu().numpy().reshape(num_batch,-1),top_k)

                question = np.char.array(["[Question]"+i for i in batch["question"]], unicode = True)
                documents = flattened_table[indicies]
                input_ = np.char.add(np.repeat(question.reshape(num_batch,-1), top_k, axis=1),documents)
                #print(input_)
                #print(input_.shape)

                question_tokenized = generator_tokenizer(
                    list(input_.reshape(-1,)), return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512
                )

                answer = batch["answer"]
                sql = batch["sql"]
                if config["method"] == "parser":
                    label_tokenized = generator_tokenizer(
                        sql, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 64
                    )
                else:
                    label_tokenized = generator_tokenizer(
                        answer, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 64
                    )                
                
                #question_tokenized = question_tokenized.to(accelerator.device)
                #label_tokenized = label_tokenized.to(accelerator.device)
                #print(question_tokenized["input_ids"].reshape(num_batch,top_k,-1).size())

                input_ids = question_tokenized["input_ids"].reshape(num_batch,top_k,-1)
                attn_mask = question_tokenized["attention_mask"].reshape(num_batch,top_k,-1)
                labels = label_tokenized["input_ids"]
                

                loss = FiD(input_ids = input_ids.to(accelerator.device),
                        attention_mask = attn_mask.to(accelerator.device),
                        labels = labels.to(accelerator.device)).loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                batch_iterator.set_postfix(loss=loss.item())
        FiD.eval()
        valid_loss=0
        with tqdm(valid_dataloader, desc=f"Epoch {str(epoch+1).rjust(2)} Validating", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
            for batch in batch_iterator:
                question_tokenized = batch["question_tokenized"]
                question_tokenized = question_tokenized.to(accelerator.device)

                num_batch = question_tokenized["input_ids"].size(0)
                top_k = config["top_k"]

                with torch.no_grad():
                    output = question_encoder(**question_tokenized).pooler_output
                
                _, indicies = index.search(output.type(torch.float32).cpu().numpy().reshape(num_batch,-1),top_k)
                question = np.char.array(["[Question]"+i for i in batch["question"]], unicode = True)
                documents = flattened_table[indicies]
                input_ = np.char.add(np.repeat(question.reshape(num_batch,-1), top_k, axis=1),documents)


                question_tokenized = generator_tokenizer(
                    list(input_.reshape(-1,)), return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512
                )
                answer = batch["answer"]
                sql = batch["sql"]
                if config["method"] == "parser":
                    label_tokenized = generator_tokenizer(
                        sql, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 64
                    )
                else:
                    label_tokenized = generator_tokenizer(
                        answer, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 64
                    )    
                input_ids = question_tokenized["input_ids"].reshape(num_batch,top_k,-1)
                attn_mask = question_tokenized["attention_mask"].reshape(num_batch,top_k,-1)
                labels = label_tokenized["input_ids"]

                with torch.no_grad():
                    loss = FiD(input_ids = input_ids.to(accelerator.device),
                               attention_mask = attn_mask.to(accelerator.device),
                                labels = labels.to(accelerator.device)).loss
                loss = accelerator.gather_for_metrics(loss)
                valid_loss += torch.sum(loss).item()
        
        valid_loss = valid_loss / len(valid_dataloader.dataset)
        accelerator.print(f"Epoch {str(epoch+1).rjust(2)} valid loss : ",valid_loss)

        if best_loss > valid_loss:
            best_loss = valid_loss
            out_dir = os.path.join(args.checkpoint_dir,"generator",config["exp_name"])
            accelerator.save_state(out_dir)
            accelerator.print(f"Epoch {str(epoch+1).rjust(2)} saved")



            


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

    #index_file_path = os.path.join(config["path"],"augmented",config["method"]+"_"+config["table_encoder"]+".index")
    index_file_path = os.path.join(config["path"],"augmented","reader_"+config["table_encoder"]+".index")

    index = faiss.read_index(index_file_path)

    trainer(config, args, index, accelerator)




if __name__=="__main__":
    main()