# ---------------------------------------------------------------------------- #
#                  Script To train and evaluate Baseline Model                 #
# ---------------------------------------------------------------------------- #
import argparse 
import os
import numpy as np
from datasets import load_dataset 
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import label_binarize 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, make_scorer
from typing import Tuple, Union, List


def get_X_y()->Tuple[pd.Series, pd.Series]:
    """Load bbc-news dataset and return features and target"""
    # Load data
    ds = load_dataset("SetFit/bbc-news")
    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])
    # Get features and targets
    X = train_df["text"]
    y = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]
    return X,y,X_test,y_test

def train_vectorizer(X:pd.Series,method:str)->Union[TfidfVectorizer]:
    """
    Train Vectorizer
    Inputs 
        - X : feature set
    Note for Tfidf-vectorizer we will learn from entire training corpus (training + validation)
    """
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
    return vectorizer

def compute_auc(y_true:pd.Series,y_hat:pd.Series)->float:
    """
    Custom auc score desinged for cross_val_score func
    Inputs 
        - y_true : true y labels
        - y_hat : predictions
    """
    y_true_binarize = label_binarize(y_true, classes = num_classes)
    auc = roc_auc_score(
        y_true_binarize,
        y_hat,
        average = "weighted",
        multi_class = "ovo"
    )
    return auc

def log_results(scores:dict, folder:str, model_name:str, emb_name:str):
    """
    Save Results
    Inputs
        - scores : accuracy scores (auc)
        - folder : store location 
        - model_name : model used for training and prediction
        - emb_name : method of embedding
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    f_name = "base_" + model_name + "_" + emb_name +  "_auc" + ".csv"
    pd.DataFrame(scores).to_csv(os.path.join(folder,f_name))

    
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description = "Arguments for baseline model")
    parser.add_argument("--model", type = str, choices = ["logistic","randomforest","nbayes"], required = True, default = "logisitc")
    parser.add_argument("--emb", type = str, choices = ["tfidf"])
    args = parser.parse_args()

    #Get data
    X, y, X_test, y_test = get_X_y()
    num_classes = y.unique()

    # Train - Validation splits
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3)
    
    # Embeddings from text
    vectorizer = train_vectorizer(X, method = "tfidf")
    X_train_emb = vectorizer.transform(X_train) # Note embeddings are trained on both train + val as tfidf are very sensitive to the available data
    X_val_emb = vectorizer.transform(X_val) # Note embeddings are trained on both train + val as tfidf are very sensitive to the available data
    X_test_emb = vectorizer.transform(X_test)

    # Select model
    if args.model == "logistic":
        model = LogisticRegression()
    if args.model == "randomforest":
        model = RandomForestClassifier()
    if args.model == "nbayes":
        model = MultinomialNB()
    
    # Train baseline model
    model.fit(X_train_emb, y_train)

    # Prediction AUC + Cross Validation
    auc_scorer = make_scorer(
        score_func =compute_auc, 
        response_method = "predict_proba"
    )

    train_scores = cross_val_score(model,X_train_emb,y_train,cv = 5,scoring = auc_scorer)
    val_scores = cross_val_score(model,X_val_emb,y_val,cv = 5,scoring = auc_scorer)
    test_scores = cross_val_score(model,X_test_emb, y_test, cv = 5, scoring = auc_scorer)
    
    # Reduce decimals
    train_scores = np.round(np.array(train_scores),3)
    val_scores = np.round(np.array(val_scores),3)
    test_scores = np.round(np.array(test_scores),3)

    print(f"train scores : {train_scores}")
    print(f"val scores : {val_scores}")
    print(f"test scores : {test_scores}")

    # Save results
    scores = {
        "train" : train_scores,
        "val" : val_scores,
        "test" : test_scores
    }

    log_results(scores, "results", args.model, args.emb)



