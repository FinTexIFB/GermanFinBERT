# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:28:56 2023

@author: scherrmann
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb#scrollTo=Dw-jG9KT-gBK
"""
from datasets import load_metric
from transformers import AutoTokenizer
from squadUtil import prepare_train_features, prepare_validation_features,  postprocess_qa_predictions
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import collections
import os
import pandas as pd
import gc
import torch
import numpy as np
import shutil
import sys
sys.path.append('..\\..\\FinetuneData\\AdHocQuad')
from loadAdHocData import loadAdHocData

#%% Load and prepare data
modelPath = "deepset/gbert-base"#"..\\BaseModels\\german-fin-gbert-optimized-3-fp-512-small-lr-ba10400-converted-hf"#"bert-base-german-cased"
if "fin-gbert" in modelPath:
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
batch_size = 4
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
numSeeds = 5
pad_on_right = tokenizer.padding_side == "right"

#%% Finteune
if "german-fin" in modelPath:
    pathBase = "GermanFinBERT\\"+modelName+"\\AdhocQuAD"
else:
    pathBase = "Benchmarks\\"+modelName+"\\AdhocQuAD"
if not os.path.exists(pathBase):
    os.makedirs(pathBase)
for seed in range(numSeeds):
    datasets = loadAdHocData()
    tokenized_datasets = datasets.map(prepare_train_features, 
                                      batched=True, 
                                      remove_columns=datasets["train"].column_names, 
                                      fn_kwargs={"tokenizer":tokenizer, "pad_on_right":pad_on_right, 
                                                 "max_length":max_length, "doc_stride":doc_stride})
    path = pathBase+"\\seed_"+str(seed)
    if not os.path.exists(path):
        os.makedirs(path)
    model = AutoModelForQuestionAnswering.from_pretrained(modelPath)
    args = TrainingArguments(
        f"{modelName}-finetuned-adhocquad",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=4,
        weight_decay=0.01,
        report_to="none",
        seed = np.random.randint(1,1000000,size=1)[0]
    )
    data_collator = default_data_collator
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    shutil.rmtree(f"{modelName}-finetuned-adhocquad")
    trainer.save_model(path)
    #%% Evaluate
    validation_features = datasets["test"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["test"].column_names, 
        fn_kwargs={"tokenizer":tokenizer, "pad_on_right":pad_on_right, 
                   "max_length":max_length, "doc_stride":doc_stride}
    )
    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    
    examples = datasets["test"]
    features = validation_features
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        
    final_predictions = postprocess_qa_predictions(datasets["test"], validation_features, raw_predictions.predictions, tokenizer)
    
    metric = load_metric("squad")
    formatted_predictions = [{"id": k, "prediction_text": v.lower()} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["test"]]
    for idx in range(len(references)):
        dictAct = references[idx]
        references[idx] = {'id': dictAct["id"],
                           'answers': {'answer_start': dictAct["answers"]['answer_start'], 
                                       'text': [dictAct["answers"]['text'][0].lower()]}}
    results = pd.Series(metric.compute(predictions=formatted_predictions, references=references))
    results.to_pickle(path + "\\eval_performance.pkl")
    print(results)
    # free gpu memory
    del model, trainer, results, raw_predictions, validation_features, examples, features, final_predictions,formatted_predictions, references,tokenized_datasets, datasets
    gc.collect()
    torch.cuda.empty_cache()

#%% Store eval results
seedPaths = os.listdir(pathBase)
results = {}
for seedPath in seedPaths:
    pathAct = pathBase + "\\"+seedPath
    results[seedPath] = pd.read_pickle(pathAct+"\\eval_performance.pkl")
results = pd.DataFrame(results)
results["Mean"] = results.mean(axis=1)
results["Median"] = results.median(axis=1)
results["Std"] = results.std(axis=1)
results.to_pickle(pathBase+"\\test_results.pkl")
