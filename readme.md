# running with docker
- docker build -t iconic . && docker run -v $PWD/results:/usr/src/app/results iconic
- Note running distil-bert exeeds 10 mins (~15 mins) using docker on an M1 mac. Faster results can be achieved using  virtual env approach. 

# running in virtual env
- To Install Conda visit : https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- To install required packages : `conda env create -f iconic-osx.yml`
    - Note conda environment was created in a mac osx computer and may not work in linux or windows operating system due to dependency issues. (If so install packages manually - numpy, pandas, datasets, pytorch, transformers)
- To activate conda environment : `conda activate iconic-osx`
- To run baseline model : `python src/baseline.py --model logistic --emb tfidf`
    -  model options : logisitic regression (`logistic`), random forest (`randomforest`) , naive bayes (`nbayes`)
- To train custom distil-bert model : `python src/model_train.py --epochs 3 --output_p src/model_weights --freeze_bb`
    - `epochs` : number of epochs to train
    - `output_p` : path to store model weights
    - `freeze_bb` : freezes weights of distil-bert and only trains final classification layer (recommended for faster training)
- To infer custom distil-bert model on test data : `python src/model_inference.py --weights_p model_weights/custom_distilbert_epochs5.pth`
    - `weights_p` : path to saved weights (during training)
- Note running distil-bert exeeds 10 mins on CPU. I have tested the model on an M1 macbook on cpu. 

# Information on problem

- I opted to perform multiclass classification with the aims of text representations to predict the relevant topic (e.g sports, business, entertainment)
- Intitially a baseline model was explored using tf-idf vectors and classical machine learning algorithms (e.g Logistic regression, Naive Bayes etc). Although it had reasonable accuracy, the auc was relatively low (~0.65).
    - Note that during training, Tf-IDF vectors were trained both on training and validations sets (not testing), as the vector representation is very sensitive to vocabulary in these dataset splits, leading to widely varying results if only training set was used.
- The low auc is likely due to the slight imbalance in the dataset and more importantly due limited training examples (~250 per class).
- To counteract the data limitation, I explored finetuning the dataset on pretrained Distil-Bert model.
- The choice of Distil-Bert was due to its lower parameter size compared to its other variants such as RoBERTa or BERT itself, which would lead to lengthier training times.
- A  classification layer was placed (head) to the existing Distil-BERT backbone which was finetuned. Distil-BERT backbone parameters were frozen to speed up training.
- Finetuning the model for single epochs allowed to improve the auc to 0.99 and an accuracy of 0.96.
- As next steps, I would like to explore topic modelling, to leverage representations to understand topics in an unsupervised fashion. 

