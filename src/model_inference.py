# ---------------------------------------------------------------------------- #
#           Script to evaluate trained Distil-Bert Model on test data          #
# ---------------------------------------------------------------------------- #
import os
from datasets import load_dataset 
import numpy as np
import pandas as pd 
import argparse 
from transformers import DistilBertTokenizerFast
from model_train import CustomDistiledBertModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import  roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from utils import BBCNewsDataset

def get_predictions(model:CustomDistiledBertModel,testloader:DataLoader,device:str):
    """
    Gets prediction from model
    Inputs 
        - model : Trained Distil-BERT model 
        - testloader : dataloader
        - device : compute resource
    """
    model.eval()
    true_ys = []
    pred_ys = []
    prob_ys_b = []
    for batch in testloader:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device) 
            yhat_probs = F.softmax(model(input_ids = input_ids, attention_mask = attention_mask),dim = 1)
            yhat_preds = torch.argmax(yhat_probs, dim = 1)
            true_ys.extend([x.item() for x in y])
            pred_ys.extend([x.item() for x in yhat_preds])
            prob_ys_b.append(yhat_probs)

    prob_ys = torch.cat(prob_ys_b, dim = 0)

    return true_ys, pred_ys, prob_ys

def compute_acc_auc(y_true:np.array,y_preds:np.array,y_probs:np.array):
    """Computes accuracy and auc scores"""
    acc = accuracy_score(y_true, y_preds)
    y_true_bin = label_binarize(y_true, classes = np.unique(y_true))
    auc = roc_auc_score(
        y_true_bin,
        y_probs,
        average = "weighted",
        multi_class = "ovo"
    )
    return acc, auc

if __name__ == "__main__":
    # Argeparse
    parser = argparse.ArgumentParser(description = "Infer DistilBert")
    parser.add_argument("--weights_p", type = str, help = "path to model weights", required = True)
    args = parser.parse_args()
    # Testing dataset
    ds = load_dataset("SetFit/bbc-news")
    df_test = pd.DataFrame(ds["test"])
    num_classes = df_test["label"].unique()
    # Datsets and Dataloaders
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bbc_test_dataset = BBCNewsDataset(df_test,tokenizer,512)
    testloader = DataLoader(bbc_test_dataset, batch_size = 16, shuffle = False)
    # Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDistiledBertModel()
    model = model.to(device)
    # Load model weights
    print("Loading model weights ...")
    model.load_state_dict(torch.load(args.weights_p))
    # Inference 
    print("Getting predictions ...")
    true_ys, pred_ys, prob_ys = get_predictions(model, testloader, device)
    # Metrics
    # Convert to numpy
    true_ys = np.array(true_ys)
    pred_ys = np.array(pred_ys)
    prob_ys = prob_ys.numpy()
    acc, auc = compute_acc_auc(true_ys,pred_ys,prob_ys)

    # Save results
    print("saving metrics in results folder ...")
    print(f"accuracy : {acc:.3f}, auc : {auc:.3f}")
    if not os.path.exists("results"):
        print("no results folder creating folder ...")
        os.mkdir("results")
    pd.DataFrame({"acc" : np.round(np.array([acc]),3), "auc" : np.round(np.array([auc]),3)}).to_csv("results/distilbert_acc_auc.csv", index = False)





