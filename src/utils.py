import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class BBCNewsDataset(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer:DistilBertTokenizer, max_length:int):
        """
        Dataset class for tokenization and load BBC News data
        Inputs 
            - df : data
            - tokenizer : distilbert tokenizer to tokenize text 
            - max_length : 
        """
        self.df = df 
        self.tokenizer = tokenizer 
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        text = self.df["text"].iloc[idx]
        label = self.df["label"].iloc[idx]
        encordings = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True, # We want the embeddings of the CLS token
            max_length = self.max_length,
            return_token_type_ids = False, 
            padding = "max_length",
            return_attention_mask = True,
            return_tensors = "pt",
            truncation = True
        )
        return {
            "text" : text, 
            "label" : label, 
            "input_ids" : encordings["input_ids"].flatten(), #(L,)
            "attention_mask" : encordings["attention_mask"].flatten() #(L,)
        }



