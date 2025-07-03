# ---------------------------------------------------------------------------- #
#                       Script to train Distil-BERT model                      #
# ---------------------------------------------------------------------------- #
import argparse 
import os
import string 
import pandas as pd
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from utils import BBCNewsDataset
from datasets import load_dataset 
import time
from typing import Tuple
import sys

class CustomDistiledBertModel(nn.Module):
    def __init__(self, num_classes:int = 5):
        super(CustomDistiledBertModel, self).__init__()
        """
        Customized Distil Bert Model - Classification layers at end
        Inputs 
            - num_classes : Number of classes in dataset
        """
        # Load pretrained Distiled Bert Modle
        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc1 = nn.Linear(768,768) # Bert model outputs 768 dim
        self.fc2 = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, input_ids, attention_mask):
        backbone_out = self.backbone(
            input_ids = input_ids, attention_mask = attention_mask
            )["last_hidden_state"] #(N,L,H) i.e (64,512,768)
        cls_out = backbone_out[:,0] # (N,H), Take CLS hidden state as it represents the entire text
        out = F.relu(self.fc1(cls_out)) # (N,H) 
        out = self.dropout(out)
        out = self.fc2(out) # (N, num_classes)
        return out

def train_one_epoch(
    model:CustomDistiledBertModel,
    criterion:nn.modules.loss.CrossEntropyLoss,
    optimizer:optim.AdamW,
    trainloader:DataLoader,
    validloader:DataLoader,
    device:torch.device
    )->Tuple[float,float]:
    """
    Train model for a single epoch 
    Inputs
        - model : Customized DistilBertModel
        - criterion : loss function 
        - optimizer : optimizer 
        - trainloader : train dataloader 
        - validloader : validation dataloader 
        - device : to put tensors to compute resources
    """
    model.train()
    train_loss = 0
    for batch in trainloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        yhat = model(input_ids = input_ids, attention_mask = attention_mask)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    valid_loss = 0
    for batch in validloader:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_valid = batch["label"].to(device)
            yhat_valid = model(input_ids = input_ids, attention_mask = attention_mask)
            loss_valid = criterion(yhat_valid, y_valid)

            valid_loss += loss_valid.item()
        
    return train_loss/len(trainloader),valid_loss/len(validloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train DistilBERT model")
    parser.add_argument("--epochs", type = int, default = 3, help = "number of epochs")
    parser.add_argument("--freeze_bb", action = "store_true", help = "whether to freeze backbone model")
    parser.add_argument("--output_p", type = str, help = "location to save weights")
    args = parser.parse_args()

    # Load data
    ds = load_dataset("SetFit/bbc-news")
    df = pd.DataFrame(ds["train"])
    df_train,df_valid = train_test_split(df, test_size = 0.2)
    num_classes = df["label"].unique()

    # Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Instantiate datasets and dataloaders
    bbc_train_dataset = BBCNewsDataset(df_train,tokenizer,512)
    bbc_valid_dataset = BBCNewsDataset(df_valid,tokenizer,512)
    trainloader = DataLoader(bbc_train_dataset, batch_size = 16, shuffle = False, drop_last = True)
    validloader = DataLoader(bbc_valid_dataset, batch_size = 16, shuffle = False, drop_last = True)

    # Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDistiledBertModel()
    model = model.to(device)

    # Freeze distilbert model weights to speed up training
    if args.freeze_bb:
        print("Freezing backbone")
        for params in model.backbone.parameters():
            params.requires_grad = False

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training 
    out_file_name = f"{args.output_p}/custom_distilbert_epochs{args.epochs}.pth"
    start = time.time()
    print("Beginning training ...")
    for epoch in range(args.epochs):
        train_loss, valid_loss = train_one_epoch(model, criterion, optimizer, trainloader, validloader, device)
        print(f"train batch loss : {train_loss:.3f}, valid batch loss : {valid_loss:.3f}")
    end = time.time()
    print(f"time taken : {(end - start)/60:.2f}")

    # Save model weights for inference 
    print(f"saving weights in {out_file_name}")
    torch.save(model.state_dict(), out_file_name)