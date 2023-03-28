import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import transformers
import os
import sqlite3

def set_seed(seed : int):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # multi-gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed) #huggingface

def parser_eval(sorted, hard_positive_idx, positive_idx, acc):

    if len(hard_positive_idx) == 0:
        if all(elem in sorted[:5] for elem in positive_idx):
            acc[0]+=1
        if all(elem in sorted[:10] for elem in positive_idx):
            acc[1]+=1
        if all(elem in sorted[:20] for elem in positive_idx):
            acc[2]+=1
        if all(elem in sorted[:50] for elem in positive_idx):
            acc[3]+=1

        
    elif len(hard_positive_idx) == 1:
        if hard_positive_idx[0] in sorted[:5]:
            acc[0]+=1
            acc[1]+=1
            acc[2]+=1
            acc[3]+=1

        elif hard_positive_idx[0] in sorted[:10]:
            acc[1]+=1
            acc[2]+=1
            acc[3]+=1

        elif hard_positive_idx[0] in sorted[:20]:
            acc[2]+=1
            acc[3]+=1

        elif hard_positive_idx[0] in sorted[:50]:
            acc[3]+=1

    else:
        if any(elem in sorted[:5] for elem in hard_positive_idx):
            acc[0]+=1
        if any(elem in sorted[:10] for elem in hard_positive_idx):
             acc[1]+=1
        if any(elem in sorted[:20] for elem in hard_positive_idx):
            acc[2]+=1
        if any(elem in sorted[:50] for elem in hard_positive_idx):
            acc[3]+=1
    
    return acc


def reader_eval(sorted, hard_positive_index, positive_index, acc):

    index = hard_positive_index + positive_index

    if all(elem in sorted[:5] for elem in index):
        acc[0]+=1
    if all(elem in sorted[:10] for elem in index):
        acc[1]+=1
    if all(elem in sorted[:20] for elem in index):
        acc[2]+=1
    if all(elem in sorted[:50] for elem in index):
        acc[3]+=1


    return acc


def run_db(config, sql):
    con = sqlite3.connect(os.path.join(config["path"],"table.db"))
    cur = con.cursor()


    try:
        cur.execute(sql)
        answers = cur.fetchall()
        if not answers:
            executed_answer = ["none"]
        else:
            executed_answer = []
            for answer in answers:
                executed_answer.append(str(answer[0]))
    except:
        executed_answer = ["execution error"]
    con.close()
    
    return executed_answer





