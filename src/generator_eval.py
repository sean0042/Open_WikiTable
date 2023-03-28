#Python Library
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import yaml
import faiss
import pickle

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
from utils import set_seed, run_db
from dataloader import get_dataloader


def eval(config, args, index, accelerator):

    seed = int(config["seed"])
    set_seed(seed)    

    flattened_table = pd.read_json(os.path.join(config["path"],"splitted_tables.json"))
    flattened_table = np.char.array(flattened_table["flattened"].values,unicode=True)

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
        fid_path = os.path.join(args.checkpoint_dir,"generator",config["exp_name"],"pytorch_model_1.bin")
        state = torch.load(fid_path, map_location = accelerator.device)
        state = {(k[8:] if k[:8]=="encoder." else k):v for k,v in state.items()}
        state = {(k.replace(".module","") if ".module" in k else k):v for k,v in state.items()}
        FiD.load_t5(state)
        #FiD.load_t5(torch.load(fid_path, map_location = accelerator.device ))
        #FiD.load_t5(t5.state_dict())
        FiD.to(accelerator.device)
    else:
        bart = AutoModelForSeq2SeqLM.from_pretrained(config["language_model"])
        bart.resize_token_embeddings(len(generator_tokenizer))
        FiD = FiDBart(bart.config)
        FiD.load_bart(bart.model.state_dict())
        FiD.to(accelerator.device)
    
    model_path = os.path.join(args.checkpoint_dir,"retriever",config["retriever_exp"],"pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location = accelerator.device))  

    #fid_path = os.path.join(args.checkpoint_dir,"generator",config["exp_name"],"pytorch_model.bin")
    #FiD.load_t5(torch.load(fid_path, map_location = accelerator.device ))

    question_encoder = model.question_enc

    _, valid_dataloader, test_dataloader = get_dataloader(question_tokenizer, config)

    question_encoder, FiD, valid_dataloader, test_dataloader = accelerator.prepare(
        question_encoder, FiD, valid_dataloader, test_dataloader
    )     

    accelerator.wait_for_everyone()  

    question_encoder.eval()
    FiD.eval()

    # valid_exact_match = 0
    # with tqdm(valid_dataloader, desc=f"validate", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
    #     for batch in batch_iterator:
    #         question_tokenized = batch["question_tokenized"]
    #         question_tokenized = question_tokenized.to(accelerator.device)

    #         num_batch = question_tokenized["input_ids"].size(0)
    #         top_k = config["top_k"]

    #         with torch.no_grad():
    #             output = question_encoder(**question_tokenized).pooler_output
            
    #         _, indicies = index.search(output.type(torch.float32).cpu().numpy().reshape(num_batch,-1),top_k)


    #         question = np.char.array(["[Question]"+i for i in batch["question"]], unicode = True)
    #         documents = flattened_table[indicies]
    #         input_ = np.char.add(np.repeat(question.reshape(num_batch,-1), top_k, axis=1),documents)

    #         question_tokenized = generator_tokenizer(
    #             list(input_.reshape(-1,)), return_tensors = "pt", padding = "max_length", truncation = True, max_length = 512
    #         )

    #         input_ids = question_tokenized["input_ids"].reshape(num_batch,top_k,-1)
    #         attn_mask = question_tokenized["attention_mask"].reshape(num_batch,top_k,-1)

    #         with torch.no_grad():
    #             generated = accelerator.unwrap_model(FiD).generate(input_ids = input_ids.to(accelerator.device),
    #                                      attention_mask = attn_mask.to(accelerator.device),
    #                                      max_length = 100)

    #         decoded = generator_tokenizer.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=True)


    #         answer = batch["answer"]


    #         if config["method"] == "parser":
    #             for i in range(num_batch):
    #                 executed = run_db(config,decoded[i])
    #                 executed = " , ".join(executed)
    #                 if executed == answer[i]:
    #                     valid_exact_match+=1    
            
    #         else:
    #             for i in range(num_batch):
    #                 if decoded[i]==answer[i]:
    #                     valid_exact_match+=1

                
    # accelerator.print("valid EM score : ",valid_exact_match/len(valid_dataloader.dataset))
    

    tmp_answer = list()
    test_exact_match = 0
    with tqdm(test_dataloader, desc=f"test", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
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

            input_ids = question_tokenized["input_ids"].reshape(num_batch,top_k,-1)
            attn_mask = question_tokenized["attention_mask"].reshape(num_batch,top_k,-1)

            with torch.no_grad():
                generated = accelerator.unwrap_model(FiD).generate(input_ids = input_ids.to(accelerator.device),
                                         attention_mask = attn_mask.to(accelerator.device),
                                         max_length = 100)

            decoded = generator_tokenizer.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=True)


            answer = batch["answer"]




            if config["method"] == "parser":
                for i in range(num_batch):
                    executed = run_db(config,decoded[i])
                    executed = " , ".join(executed)
                    tmp_answer.append(executed)
                    if executed == answer[i]:
                        test_exact_match+=1    
            
            else:
                for i in range(num_batch):
                    
                    if decoded[i]==answer[i]:
                        test_exact_match+=1

    with open("/home/sjkweon/Open-WikiTable/parser_test.pickle","wb") as f:
        pickle.dump(tmp_answer,f)


    accelerator.print("test EM score : ",test_exact_match/len(test_dataloader.dataset))





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

    index_file_path = os.path.join(config["path"],"augmented","reader_"+config["table_encoder"]+".index")

    index = faiss.read_index(index_file_path)

    eval(config, args, index, accelerator)




if __name__=="__main__":
    main()