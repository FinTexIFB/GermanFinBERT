from datasets import Dataset, load_dataset
import os
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import pickle 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import gc

#%%
    
def Encoder(df,columnToEncode="label",label_encoder=None):
    if label_encoder:
        le=label_encoder
        df[columnToEncode] = le.transform(df[columnToEncode])
        return df
    else:
        le = LabelEncoder()
        df[columnToEncode] = le.fit_transform(df[columnToEncode])
        classes = pd.Series(le.classes_)
        return df,le,classes
    
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def prepare_data(data):
    tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.remove_columns(["sentence"])
    tokenized_data = tokenized_data.rename_column("label", "labels")
    tokenized_data.set_format("torch")
    return tokenized_data

def finetuneBertSenitment1Cycle(lr,num_epochs,seed,pathBase,train_dataloader,eval_dataloader, modelPath, numLabels):
    # Store checkpoint
    path = pathBase+"\\seed_"+str(seed)
    if not os.path.exists(path):
        os.makedirs(path)
    # Train
    model = AutoModelForSequenceClassification.from_pretrained(modelPath,
        num_labels=numLabels)

    optimizer = AdamW(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, div_factor=10, steps_per_epoch=len(train_dataloader), epochs=num_epochs, pct_start=0.5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    actualStartTime = time.time()
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    # Evaluate on test
    model.eval()
    predictions = []
    targets = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictionsAct = torch.argmax(logits, dim=-1).tolist()
        predictions += predictionsAct
        targetAct = batch["labels"].tolist()
        targets += targetAct
        batch = {k: v.cpu() for k, v in batch.items()}

    # Confusion matrix: Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    cm = confusion_matrix(targets, predictions)

    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}

    # Calculate the accuracy for each one of our classes
    for idx, classAct in enumerate(classes):
        # True negatives are all the samples that are not our current GT class (not the current row) 
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        
        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]

        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[classAct] = (true_positives + true_negatives) / np.sum(cm)
    per_class_accuracies["macro"] =np.mean(list(per_class_accuracies.values()))
    per_class_accuracies["micro"] =accuracy_score(targets, predictions)
    accuracy = pd.Series(per_class_accuracies)
    #precision_score
    precision = precision_score(targets, predictions, average=None)
    precision = np.append(precision,precision_score(targets, predictions, average="macro"))
    precision = np.append(precision,precision_score(targets, predictions, average="micro"))
    precision = pd.Series(precision,index = accuracy.index)
    #recall_score
    recall = recall_score(targets, predictions, average=None)
    recall = np.append(recall,recall_score(targets, predictions, average="macro"))
    recall = np.append(recall,recall_score(targets, predictions, average="micro"))
    recall = pd.Series(recall,index = accuracy.index)
    #F1
    f1 = f1_score(targets, predictions, average=None)
    f1 = np.append(f1,f1_score(targets, predictions, average="macro"))
    f1 = np.append(f1,f1_score(targets, predictions, average="micro"))
    f1 = pd.Series(f1,index = accuracy.index)
    actF1Score = f1["macro"]
    # Store checkpoint
    model.save_pretrained(path)
    # Print statement
    hoursFromBeginning = (time.time() - overallStartTime) / 3600
    hoursFromTrainingStartSetup = (time.time() - actualStartTime) / 3600
    print(f"Hours Passed From Beginning: {hoursFromBeginning}, Hours Passed For Actual Setup: {hoursFromTrainingStartSetup}, Act Test F1: {actF1Score}")

    outputDict = {}
    outputDict["LearningRate"] = lr
    outputDict["BatchSize"] = batch_size
    outputDict["NumEpochs"] = num_epochs
    outputDict["Seed"] = seed
    outputDict["Model"] = "huggingface"
    outputDict["Accuracy"] = accuracy
    outputDict["Precision"] = precision
    outputDict["Recall"] = recall
    outputDict["F1"] = f1

    with open(path+"\\checkpoint_dictionary.pkl", 'wb') as f:
        pickle.dump(outputDict, f)
    
#%% Hyperparameters
labelType="coarse" # "coarse" (Binary), "fine" (Multiclass)
seeds=5
learning_rates = [1e-5,2e-5,3e-5,4e-5,5e-5]
batch_size=8
numEpochs = [3,4,5]
overallStartTime = time.time()
modelPath = "..\\BaseModels\\german-fin-hf-bert-optimized-small-lr-ba87500-converted-hf"#"bert-base-german-cased"
if "gbert" in modelPath:
    modelName = modelPath.split("\\")[-1]
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
elif "\\" in modelPath:
    modelName = modelPath.split("\\")[-1]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
elif "/" in modelPath:
    modelName = modelPath.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
else:
    modelName = modelPath
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
if "-converted-hf" in modelName:
    modelName = modelName.replace("-converted-hf","")
# Load data
#Train
train_data = load_dataset("gwlms/germeval2018", split="train")
train_data = train_data.to_pandas()
if labelType == "coarse":
    train_data = train_data[[ 'text', 'coarse-grained']]
    train_data = train_data.rename(columns={'coarse-grained':"label"})
else:
    train_data = train_data[[ 'text', 'fine-grained']]
    train_data = train_data.rename(columns={'fine-grained':"label"})
train_data = train_data.rename(columns={"text":"sentence"})
train_data,encoder,classes = Encoder(train_data)
train_dataset = Dataset.from_pandas(train_data[:-500])
# Val
val_dataset = Dataset.from_pandas(train_data[-500:])
# Test
test_data = load_dataset("gwlms/germeval2018", split="test")
if labelType == "coarse":
    test_data = test_data[['text','coarse-grained']]
    test_data = test_data.rename(columns={'coarse-grained':"label"})
else:
    test_data = test_data[[ 'text', 'fine-grained']]
    test_data = test_data.rename(columns={'fine-grained':"label"})
test_data = test_data.rename(columns={"text":"sentence"})
test_data = Encoder(test_data,label_encoder=encoder)
test_dataset = Dataset.from_pandas(test_data)    
# Tokenize
tokenized_train_dataset = prepare_data(train_dataset)
tokenized_val_dataset = prepare_data(val_dataset)
tokenized_test_dataset = prepare_data(test_dataset)
numLabels = len(classes)
#%% Finetune
if "german-fin" in modelPath:
    pathRoot = "GermanFinBERT"
else:
    pathRoot = "Benchmarks"
for lr in learning_rates:
    for num_epochs in numEpochs:
        pathBase = pathRoot + "\\"+modelName+"\\Sentiment_General\\lr_"+str(lr)+"num_epochs_"+str(num_epochs)
        if not os.path.exists(pathBase):
            os.makedirs(pathBase)
        for seed in range(seeds):
            # Create Dataloader
            train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=batch_size)
            eval_dataloader = DataLoader(tokenized_val_dataset, batch_size=batch_size)
            #test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size)
            # Finetune
            finetuneBertSenitment1Cycle(lr,num_epochs,seed,pathBase,train_dataloader,eval_dataloader, modelPath,numLabels)
            # Delete all variables but Hyperparameters
            #torch._C._cuda_clearCublasWorkspaces()
            #torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            
#%% Get finetuning results (evaluation set)
modelDirs = os.listdir(pathRoot + "\\"+modelName+"\\Sentiment_General")
modelDirs = [x for x in modelDirs if "num_epochs" in x]
results =[]
for mdl in modelDirs:
    pathAct = pathRoot + "\\"+modelName+"\\Sentiment_General\\"+mdl
    seedsAct = os.listdir(pathAct)
    accuracy = []
    f1 = []
    for seedIdx,seedPath in enumerate(seedsAct):
        with open(pathAct+"\\"+seedPath+"\\checkpoint_dictionary.pkl", 'rb') as f:
            d = pickle.load( f)        
        accuracy.append(d["Accuracy"]["macro"])
        f1.append(d["F1"]["macro"])
    resultAct = {}
    resultAct["Learning Rate"] = d["LearningRate"]
    resultAct["Num. Epochs"] = d["NumEpochs"]
    resultAct["Mean Accuracy"] = np.mean(np.array(accuracy))
    resultAct["Std. Accuracy"] = np.std(np.array(accuracy))   
    resultAct["Mean F1"] = np.mean(np.array(f1))
    resultAct["Std. F1"] = np.std(np.array(f1))    
    results.append(resultAct)
results = pd.DataFrame(results)
        
        
#%% Get finetuning results (test set)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size)
maxId = results["Mean F1"].idxmax()
bestLr = results.loc[maxId,"Learning Rate"]
bestNumEpoch = results.loc[maxId,"Num. Epochs"]
modelPath = pathRoot + "\\"+modelName+"\\Sentiment_General\\lr_"+str(bestLr)+"num_epochs_"+str(bestNumEpoch)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
accuracies = {}
precisions = {}
recalls = {}
f1s = {}
for seed in range(seeds):
    actModelPath = modelPath +"\\seed_"+str(seed)
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(actModelPath,
        num_labels=numLabels)
    model.to(device)
    # Evaluate on test
    model.eval()
    predictions = []
    targets = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictionsAct = torch.argmax(logits, dim=-1).tolist()
        predictions += predictionsAct
        targetAct = batch["labels"].tolist()
        targets += targetAct
    
    # Confusion matrix: Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    cm = confusion_matrix(targets, predictions)
    
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    
    # Calculate the accuracy for each one of our classes
    for idx, classAct in enumerate(classes):
        # True negatives are all the samples that are not our current GT class (not the current row) 
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        
        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]
    
        # The accuracy for the current class is the ratio between correct predictions to all predictions
        per_class_accuracies[classAct] = (true_positives + true_negatives) / np.sum(cm)
    per_class_accuracies["macro"] =np.mean(list(per_class_accuracies.values()))
    per_class_accuracies["micro"] =accuracy_score(targets, predictions)
    accuracy = pd.Series(per_class_accuracies)
    accuracies[seed] = accuracy
    #precision_score
    precision = precision_score(targets, predictions, average=None)
    precision = np.append(precision,precision_score(targets, predictions, average="macro"))
    precision = np.append(precision,precision_score(targets, predictions, average="micro"))
    precision = pd.Series(precision,index = accuracy.index)
    precisions[seed] = precision
    #recall_score
    recall = recall_score(targets, predictions, average=None)
    recall = np.append(recall,recall_score(targets, predictions, average="macro"))
    recall = np.append(recall,recall_score(targets, predictions, average="micro"))
    recall = pd.Series(recall,index = accuracy.index)
    recalls[seed] = recall
    #F1
    f1 = f1_score(targets, predictions, average=None)
    f1 = np.append(f1,f1_score(targets, predictions, average="macro"))
    f1 = np.append(f1,f1_score(targets, predictions, average="micro"))
    f1 = pd.Series(f1,index = accuracy.index)
    f1s[seed] = f1
meanTestAccuracy = pd.DataFrame(accuracies).mean(axis=1)
stdTestAccuracy = pd.DataFrame(accuracies).std(axis=1)
meanTestPrecision = pd.DataFrame(precisions).mean(axis=1)
stdTestPrecision = pd.DataFrame(precisions).std(axis=1)
meanTestRecall = pd.DataFrame(recalls).mean(axis=1)
stdTestRecall = pd.DataFrame(recalls).std(axis=1)
meanTestF1 = pd.DataFrame(f1s).mean(axis=1)
stdTestF1 = pd.DataFrame(f1s).std(axis=1)
           
# Store Results for Accuracy and F1
# Create Multiindex for Accuracy
iterables = [["Accuracy"], pd.Series(meanTestAccuracy.index).str.capitalize().to_list()]
mi = pd.MultiIndex.from_product(iterables)
meanTestAccuracy.index = mi
stdTestAccuracy.index = mi
# Create Multiindex for F1
iterables = [["F1"], pd.Series(meanTestF1.index).str.capitalize().to_list()]
mi = pd.MultiIndex.from_product(iterables)
meanTestF1.index = mi
stdTestF1.index = mi
# Combine 
meanTest = pd.concat([meanTestAccuracy,meanTestF1])
stdTest = pd.concat([stdTestAccuracy,stdTestF1])
resultDict = {}
resultDict["Mean"] = meanTest
resultDict["Std"] = stdTest
resultDict["Learning Rate"] = bestLr
resultDict["Num. Epochs"] = bestNumEpoch
resultDict["Dataset"] = "Sentiment"
resultDict["Num. Seeds"] = seeds
# Store
with open(pathRoot + "\\"+modelName+"\\Sentiment_General\\test_results.pkl", 'wb') as f:
    pickle.dump(resultDict, f)