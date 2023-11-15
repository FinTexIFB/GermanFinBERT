from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import pandas as pd
from sklearn.metrics import f1_score
import os
import time

modelPath = "deepset/gbert-base"#"bert-base-german-cased"
if "gbert" and "\\" in modelPath:
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
#%% Begin by loading the ad hoc dataset dataset:
    
train_data = pd.read_parquet(r"..\..\FinetuneData\AdHocMultilabel\train.parquet")
train_dataset = Dataset.from_pandas(train_data)

val_data = pd.read_parquet(r"..\..\FinetuneData\AdHocMultilabel\testFull.parquet")
val_dataset = Dataset.from_pandas(val_data)
#%% Tokenize
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def prepare_data(data):
    tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data = tokenized_data.remove_columns(["sentence"])
    tokenized_data.set_format("torch")
    return tokenized_data

tokenized_train_dataset = prepare_data(train_dataset)

tokenized_val_dataset = prepare_data(val_dataset)
numSeeds=5
if "german-fin" in modelPath:
    pathBase = "GermanFinBERT\\"+modelName+"\\AdhocMultilabel"
else:
    pathBase = "Benchmarks\\"+modelName+"\\AdhocMultilabel"
if not os.path.exists(pathBase):
    os.makedirs(pathBase)
overallStartTime = time.time()
for seed in range(numSeeds):
    actualStartTime = time.time()
    path = pathBase+"\\"+str(seed)
    if not os.path.exists(path):
        os.makedirs(path)
    #%% Create Dataloader
    train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(tokenized_val_dataset, batch_size=4)
    #%% Train
    topics = ["Earnings","SEO","Management","Guidance","Gewinnwarnung","Beteiligung","Dividende","Restructuring","Debt","Law","Großauftrag","Squeeze","Insolvenzantrag","Insolvenzplan","Delay","Split","Pharma_Good","Rückkauf","Real_Invest","Delisting"]
    model = AutoModelForSequenceClassification.from_pretrained(modelPath,
        num_labels=len(topics),
        problem_type="multi_label_classification")
    
    # model.state_dict()["bert.encoder.layer.0.attention.output.dense.bias"]
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 4
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, div_factor=10, steps_per_epoch=len(train_dataloader), epochs=num_epochs, pct_start=0.5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k!="Hashs"}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    #%% Evaluate 
    
    model.eval()
    predictions = []
    hashs = []
    targets = []
    for batch in eval_dataloader:
        hashs += batch["Hashs"]
        batch = {k: v.to(device) for k, v in batch.items() if k!="Hashs"}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictionsAct = (torch.sigmoid(logits)>0.6).float()
        predictions += predictionsAct.tolist()
        targets+= batch["labels"].tolist()
    
    # Aggregate on document level
    predictionsDf= pd.DataFrame(predictions)
    predictionsDf["Hashs"] = hashs
    predictionsDf = (predictionsDf.groupby("Hashs").sum()>0).to_numpy()
    
    targetsDf= pd.DataFrame(targets)
    targetsDf["Hashs"] = hashs
    targetsDf = (targetsDf.groupby("Hashs").sum()>0).to_numpy()
    
    # Compute scores
    f1Scores=pd.Series(f1_score(targetsDf,predictionsDf, average=None),index=topics)
    f1Scores=pd.concat([f1Scores,pd.Series([f1_score(targetsDf,predictionsDf, average="macro"),f1_score(targetsDf,predictionsDf, average="micro")],index=["Avg. (Macro)","Avg. (Micro)"])])
    # Store checkpoint
    #os.makedirs(path)
    model.save_pretrained(path)
    f1Scores.to_pickle(path+"\\f1scores.pkl")
    actF1Score = f1Scores["Avg. (Macro)"]
    # Print statement
    hoursFromBeginning = (time.time() - overallStartTime) / 3600
    hoursFromTrainingStartSetup = (time.time() - actualStartTime) / 3600
    print(f"Hours Passed From Beginning: {hoursFromBeginning}, Hours Passed For Actual Setup: {hoursFromTrainingStartSetup}, Act Test F1: {actF1Score}")

#%% Get finetuning results
modelDirs = os.listdir(pathBase)
modelDirs =[x for x in modelDirs if "test" not in x]
results ={}
for mdl in modelDirs:
    pathAct = pathBase+"\\"+mdl
    resultsAct = pd.read_pickle(pathAct+"\\f1scores.pkl")
    results[mdl] = resultsAct
results = pd.DataFrame(results)
results["Mean"] = results.mean(axis=1)
results["Median"] = results.median(axis=1)
results["Std"] = results.std(axis=1)
results.to_pickle(pathBase+"\\test_results.pkl")

