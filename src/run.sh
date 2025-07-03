#!/bin/bash
# baseline model with logistic regression and tfidf embeddings, uncomment to run
#python ./src/baseline.py --model logistic --emb tfidf

# train custom distil-bert model
mkdir -p ./src/model_weights 
EPOCHS=1 # Define number of epochs
python ./src/model_train.py --epochs $EPOCHS --freeze_bb --output_p "$PWD/src/model_weights"

# inference on test set 
WEIGHTS_PATH="./src/model_weights/custom_distilbert_epochs${EPOCHS}.pth"
python ./src/model_inference.py --weights_p $WEIGHTS_PATH
