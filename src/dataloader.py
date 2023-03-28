import pandas as pd
import random
import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from accelerate import Accelerator


def transform_dataframe(dataframe : pd.DataFrame, option : str):

    assert option == "parser" or option == "reader"

    if option == "parser":

        tmp = pd.DataFrame(columns = ["question_id","original_table_id","question","sql","answer","positive_idx","negative_idx","dataset"])
        for row in dataframe.iterrows():
            question_id = row[1]["question_id"]
            original_table_id = row[1]["original_table_id"]
            question = row[1]["question"]
            sql = row[1]["sql"]
            answer = row[1]["answer"]
            
            hard_positive_idx = row[1]["hard_positive_idx"]
            positive_idx = row[1]["positive_idx"]
            negative_idx = row[1]["negative_idx"]

            dataset = row[1]["dataset"]
            
            if len(hard_positive_idx) == 0:
                for i in range(len(positive_idx)):
                    random.shuffle(negative_idx)
                    add = [question_id,original_table_id,question,sql,answer,positive_idx[i],negative_idx[0],dataset]
                    tmp.loc[len(tmp)] = add
            elif len(hard_positive_idx) == 1:
                add = [question_id,original_table_id,question,sql,answer,hard_positive_idx[0],negative_idx[0],dataset]
                tmp.loc[len(tmp)] = add
            else:
                random.shuffle(hard_positive_idx)
                add = [question_id,original_table_id,question,sql,answer,hard_positive_idx[0],negative_idx[0],dataset]
                tmp.loc[len(tmp)] = add
        return tmp

    elif option == "reader":
        tmp = pd.DataFrame(columns = ["question_id","original_table_id","question","sql","answer","positive_idx","negative_idx","dataset"])
        for row in dataframe.iterrows():
            question_id = row[1]["question_id"]
            original_table_id = row[1]["original_table_id"]
            question = row[1]["question"]
            sql = row[1]["sql"]
            answer = row[1]["answer"]
            
            hard_positive_idx = row[1]["hard_positive_idx"]
            positive_idx = row[1]["positive_idx"]
            negative_idx = row[1]["negative_idx"]

            dataset = row[1]["dataset"]

            pos = hard_positive_idx + positive_idx


            for i in range(len(pos)):
                random.shuffle(negative_idx)
                add = [question_id,original_table_id,question,sql,answer,pos[i],negative_idx[0],dataset]
                tmp.loc[len(tmp)] = add
        return tmp

         


class OpenWikiTable(Dataset):

    def __init__(self, df : pd.DataFrame, train : bool):
        self.train = train
        self.data = df
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question = self.data.iloc[idx].question
        sql = self.data.iloc[idx].sql
        answer = self.data.iloc[idx].answer
         
        if self.train:
            positive_idx = self.data.iloc[idx].positive_idx
            negative_idx = self.data.iloc[idx].negative_idx

            return {
                "question" : question,
                "sql" : sql,
                "answer" : answer,
                "positive_idx" : positive_idx,
                "negative_idx" : negative_idx
            }

        
        else:
            hard_positive_idx = self.data.iloc[idx].hard_positive_idx
            positive_idx = self.data.iloc[idx].positive_idx

            return {
                "question" : question,
                "sql" : sql,
                "answer" : answer,
                "hard_positive_idx" : hard_positive_idx,
                "positive_idx" : positive_idx
            }


 
def tokenize_tables(tables : pd.DataFrame, config):

    table_encoder = config["table_encoder"]

    if "bert" in table_encoder:
        table_encoder_name = "bert"
        tokenizer = AutoTokenizer.from_pretrained(table_encoder)
        special_tokens = ["[Title]", "[Section title]", "[Caption]", "[Table name]", "[Header]","[Rows]","[Row]","[sep]"]
        tokenizer.add_tokens(special_tokens)

        for idx, i in enumerate(tables["flattened"]):
            if idx == 0:
                tmp_dict = tokenizer(i, return_tensors="pt", padding= "max_length", truncation = True)
            else:
                tokenized = tokenizer(i, return_tensors="pt", padding= "max_length", truncation = True)
                tmp_dict["input_ids"] = torch.cat((tmp_dict["input_ids"],tokenized["input_ids"]),0)
                tmp_dict["token_type_ids"] = torch.cat((tmp_dict["token_type_ids"],tokenized["token_type_ids"]),0)
                tmp_dict['attention_mask'] = torch.cat((tmp_dict['attention_mask'],tokenized['attention_mask']),0)

        torch.save(tmp_dict,os.path.join(config["path"],"augmented",table_encoder_name+"_table_tokenized.pkl"))
        return tmp_dict


    elif "tapas" in table_encoder:
        table_encoder_name = "tapas"
        tokenizer = AutoTokenizer.from_pretrained(table_encoder)
        special_tokens = ["[Title]", "[Section title]", "[Caption]"]
        tokenizer.add_tokens(special_tokens)


        for row in tables.iterrows():
            page_title = row[1]["page_title"]
            section_title = row[1]["section_title"]
            caption = row[1]["caption"]

            title_description = "[Title]:"+page_title+"[Section title]"+section_title+"[Caption]"+caption


            header = row[1]["header"]
            rows = row[1]["rows"]
            dictionary = dict(zip(header,[list(x) for x in zip(*rows)]))
            table = pd.DataFrame.from_dict(dictionary)

            if row[0] == 0:
                tmp_dict = tokenizer(table = table, queries = title_description , return_tensors = "pt", padding = "max_length", truncation = True)
            else:
                tokenized = tokenizer(table = table, queries = title_description , return_tensors = "pt", padding = "max_length", truncation = True)
                tmp_dict["input_ids"] = torch.cat((tmp_dict["input_ids"],tokenized["input_ids"]),0)
                tmp_dict["token_type_ids"] = torch.cat((tmp_dict["token_type_ids"],tokenized["token_type_ids"]),0)
                tmp_dict['attention_mask'] = torch.cat((tmp_dict['attention_mask'],tokenized['attention_mask']),0)

        torch.save(tmp_dict,os.path.join(config["path"],"augmented",table_encoder_name+"_table_tokenized.pkl"))
        return tmp_dict
    
    else:
        print("we only offer bert and tapas for table retriever")


def get_retriever_dataloader(tokenizer, config):
    
    method = config["method"]

    path = os.path.join(config["path"],"augmented",method+"_train.json")

    if not os.path.exists(path):
        dataframe = pd.read_json(os.path.join(config["path"],"train.json"))
        print(f"building and saving {method} data")
        train_dataset = transform_dataframe(dataframe, method)
        train_dataset.to_json(path)
    else:
        train_dataset = pd.read_json(path)


    tables = pd.read_json(os.path.join(config["path"],"splitted_tables.json"))
    table_encoder = config["table_encoder"]

    if "bert" in table_encoder:
        table_encoder_name = "bert"
    else:
        table_encoder_name = "tapas"

    tokenized_table_path = os.path.join(config["path"],"augmented",table_encoder_name+"_table_tokenized.pkl")

    if not os.path.exists(tokenized_table_path):
        print(f"building and saving {table_encoder_name} tokenized table")
        tokenized_table = tokenize_tables(tables, config)
    else:
        tokenized_table = torch.load(tokenized_table_path)
    
        


    train_dataset = OpenWikiTable(train_dataset, train = True)


    batch_size = config["batch_size"]
    question_max_len = config["question_max_len"]


    def collate_fn(samples):
        questions = [sample["question"] for sample in samples]
        #answer = [sample["answer"] for sample in samples]
        positive_idx = [sample["positive_idx"] for sample in samples]
        negative_idx = [sample["negative_idx"] for sample in samples]

        if config["hard_negative"]:
            index = positive_idx + negative_idx
        else:
            index = positive_idx

        """
        이거 다시
        """
        index = [i-1 for i in index]

        table_tokenized = {"input_ids" : tokenized_table["input_ids"][index],
                              "attention_mask" : tokenized_table["attention_mask"][index],
                              "token_type_ids" : tokenized_table["token_type_ids"][index]}   
        
        
        question_tokenized = tokenizer(
            questions, return_tensors = "pt", padding = "max_length", truncation = True, max_length = question_max_len
        )

        return {"question" : questions,
                "question_tokenized" : question_tokenized,
                "table_tokenized" : table_tokenized}
    
    train_dataloader = DataLoader(
        train_dataset, shuffle = True, collate_fn = collate_fn, batch_size = batch_size, num_workers = 4
    )

    return train_dataloader




def get_dataloader(tokenizer, config):

    train_dataset = pd.read_json(os.path.join(config["path"],"train.json"))
    valid_dataset = pd.read_json(os.path.join(config["path"],"valid.json"))
    test_dataset = pd.read_json(os.path.join(config["path"],"test.json"))

    train_dataset = OpenWikiTable(train_dataset, train = False)
    valid_dataset = OpenWikiTable(valid_dataset, train = False)
    test_dataset = OpenWikiTable(test_dataset, train = False)

    batch_size = config["batch_size"]
    question_max_len = config["question_max_len"]

    def collate_fn(samples):
        questions = [sample["question"] for sample in samples]
        answer = [" , ".join(sample["answer"]) for sample in samples]
        hard_positive_idx = [sample["hard_positive_idx"] for sample in samples]
        positive_idx = [sample["positive_idx"] for sample in samples]
        sql = [sample["sql"] for sample in samples]
 
        
        
        question_tokenized = tokenizer(
            questions, return_tensors = "pt", padding = "max_length", truncation = True, max_length = question_max_len
        )

        return {"question" : questions,
                "question_tokenized" : question_tokenized,
                "hard_positive_idx" : hard_positive_idx,
                "positive_idx" : positive_idx,
                "answer" : answer,
                "sql" : sql}
    
    train_dataloader = DataLoader(
        train_dataset, shuffle = True, collate_fn = collate_fn, batch_size = batch_size, num_workers = 4
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, shuffle = False, collate_fn = collate_fn, batch_size = batch_size, num_workers = 4
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle = False, collate_fn = collate_fn, batch_size = batch_size, num_workers = 4
    )

    return train_dataloader, valid_dataloader, test_dataloader



class tokenized_table(Dataset):
    def __init__(self, config):
        if "bert" in config["table_encoder"]:
            path = os.path.join(config["path"],"augmented","bert_table_tokenized.pkl")
        else:
            path = os.path.join(config["path"],"augmented","tapas_table_tokenized.pkl")
        self.tokenized = torch.load(path)
    
    def __len__(self):
        return len(self.tokenized["input_ids"])
    
    def __getitem__(self,idx):
        return {
            "input_ids" : self.tokenized["input_ids"][idx],
            "token_type_ids" : self.tokenized["token_type_ids"][idx],
            "attention_mask" : self.tokenized["attention_mask"][idx]
        }


def get_table_dataloader(config):

    data = tokenized_table(config)
    
    table_dataloader = DataLoader(data, batch_size = 1024, shuffle= False ,num_workers =4)

    return table_dataloader




