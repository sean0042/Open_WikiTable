#Python Library
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import yaml

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

from dataloader import get_retriever_dataloader
from model import BiEncoder
from utils import set_seed


def trainer(config, args):

    if args.fp16:
        mixed_precision = "fp16"
    else:
        mixed_precision = "no"
    accelerator = Accelerator(cpu = False, mixed_precision = mixed_precision)

    exp_name = config["exp_name"]
    seed = int(config["seed"])
    lr = float(config["lr"])
    num_epochs = int(config["num_epochs"])
    warmup_step = config["warm_up_step"]

    set_seed(seed)

    # if args.logging_steps > 0 :
    #     if accelerator.is_main_process:
    #         tb_writer = SummaryWriter("runs/retriever/"+exp_name) 
    
    question_tokenizer = AutoTokenizer.from_pretrained(config["question_encoder"])
    table_tokenizer = AutoTokenizer.from_pretrained(config["table_encoder"])

    if "bert" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]", "[Table name]", "[Header]","[Rows]","[Row]","[sep]"]
        table_tokenizer.add_tokens(special_tokens)
    
    elif "tapas" in config["table_encoder"]:
        special_tokens = ["[Title]", "[Section title]", "[Caption]"]
        table_tokenizer.add_tokens(special_tokens)

     
    train_dataloader = get_retriever_dataloader(question_tokenizer, config)
    model = BiEncoder(config["question_encoder"], config["table_encoder"], question_tokenizer, table_tokenizer)
    model = model.to(accelerator.device)
    optimizer = AdamW(params = model.parameters(), lr = lr)
    cross_entropy_loss = nn.CrossEntropyLoss()
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    ) 


    accelerator.wait_for_everyone()

    train_loss, logging_loss = 0.0, 0.0
    global_step = 1

    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {str(epoch+1).rjust(2)} training", disable=args.local_rank not in [-1, 0], leave=False) as train_batch_iterator:
            for step, batch in enumerate(train_batch_iterator):
                question_tokenized = batch["question_tokenized"]
                table_tokenized = batch["table_tokenized"]
                num_batch = question_tokenized["input_ids"].size(0)
                label = torch.arange(num_batch, dtype = torch.long)

                question_tokenized = question_tokenized.to(accelerator.device)
                table_tokenized["input_ids"] = table_tokenized["input_ids"].to(accelerator.device)
                table_tokenized["attention_mask"] = table_tokenized["attention_mask"].to(accelerator.device)
                table_tokenized["token_type_ids"] = table_tokenized["token_type_ids"].to(accelerator.device)
                #table_tokenized = table_tokenized.to(accelerator.device)
                label = label.to(accelerator.device) 

                question_encoded, document_encoded = model(question_tokenized, table_tokenized)
                sim_scores = torch.matmul(question_encoded,torch.transpose(document_encoded,0,1))

                loss = cross_entropy_loss(sim_scores,label)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()

                optimizer.zero_grad()

                train_batch_iterator.set_postfix(loss=loss.item())
                train_loss += loss.item()
                global_step += 1

                if accelerator.is_main_process and args.logging_steps > 0 and (global_step % args.logging_steps == 0):
                    #tb_writer.add_scalar("train_loss", (train_loss-logging_loss) / args.logging_steps, global_step)
                    logging_loss = train_loss


    out_dir = os.path.join(args.checkpoint_dir,"retriever",exp_name)
    accelerator.save_state(out_dir)




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
        "--logging_steps",
        type = int,
        default = 10
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
    
    trainer(config,args)


if __name__=="__main__":
    main()