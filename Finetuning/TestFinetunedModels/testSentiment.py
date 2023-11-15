# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:58:01 2023

@author: scherrmann
"""

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datasets import Dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
classes = {0 : "negative",
           1 : "neutral",
           2 : "positive"}

sentences = ["Die Bruttomarge des Unternehmens verbesserte sich im dritten Quartal um 0,2 Prozentpunkte auf 49,3 % (2022: 49,1 %).",
             "Unter Berücksichtigung der positiven Auswirkungen der Verkäufe von Yeezy Produkten im zweiten und dritten Quartal, der potenziellen Abschreibung des übrigen Yeezy Bestands in Höhe von nun etwa 300 Mio. € (bisherige Prognose: 400 Mio. €) sowie von Einmalkosten im Zusammenhang mit der strategischen Prüfung von bis zu 200 Mio. € (unverändert), geht das Unternehmen nun davon aus, für das Geschäftsjahr 2023 ein negatives Betriebsergebnis in Höhe von etwa 100 Mio. € zu berichten (bisherige Prognose: negatives Betriebsergebnis in Höhe von 450 Mio. €).",
             "Vectron setzt operativen Aufwärtstrend mit einem starken 3. Quartal fort",
             "INFICON verzeichnete im zweiten Quartal sowohl in allen Zielmärkten als auch in allen Regionen deutliche Umsatzsteigerungen und ist mit der gesamthaften Entwicklung grösstenteils sehr zufrieden",
             "STS Group AG erhält Großauftrag von führendem Nutzfahrzeughersteller in Nordamerika und plant Bau eines ersten US-Werks",
            "Der Vorstand der Consulting Team Holding Aktiengesellschaft hat am 29. August 2023 mit Zustimmung des Aufsichtsrats der Gesellschaft am 06. September 2023 beschlossen, das Grundkapital der Gesellschaft von zurzeit EUR 9. 648.000,- gegen Bareinlage um bis zu EUR 603.000,- auf bis zu EUR 10.251.000,- durch Ausgabe von bis zu 603.000 neuen, auf den Inhaber lautenden Stückaktien zu erhöhen.",
            "Der Vorstand der The Social Chain AG hat heute ein von allen drei Mitgliedern des Aufsichtsrates, Herrn Stephan Brunke, Herrn Sebastian Stietzel und Frau Henrike Luszick, unterzeichnetes Schreiben erhalten, in welchem diese ihr jeweiliges Amt als Mitglieder des Aufsichtsrates mit sofortiger Wirkung niederlegen. ",
            "Die Stabilus SE hat eine Vereinbarung zum Erwerb des Industrial-Automation-Spezialisten DESTACO von dem amerikanischen Industriegüterkonzern Dover Corporation unterzeichnet.",
            "Zukünftig soll jedoch je Geschäftsjahr eine Mindestdividende in Höhe von EUR 2,00 je dividendenberechtigter Aktie an die Aktionärinnen und Aktionäre ausgeschüttet werden.",
            "Die Geschäftsführung der Neue ZWL Zahnradwerk Leipzig GmbH hat heute die Emission einer neuen Unternehmensanleihe (ISIN: DE000A351XF8) mit einem Zinssatz von 9,5 % p.a. und einem Volumen von bis zu 15 Mio. Euro beschlossen.",
            "Die Steuler Fliesengruppe AG (ISIN DE0006770001) muss die für das laufende Geschäftsjahr 2023 im Rahmen des Geschäftsberichts 2022 abgegebene Prognose der Erreichung eines Konzernjahresüberschusses in einer Bandbreite von 3,2 bis 3,7 Mio. € zurücknehmen.",
            "Dem Vorstand der Deutsche Konsum REIT-AG ist heute bekannt geworden, dass das Finanzgericht Berlin-Brandenburg den Antrag der DKR auf Aussetzung der Vollziehung der Körperschaftsteuerbescheide für die Jahre 2016 bis 2021, der Gewerbesteuermessbetragsbescheide für die Jahre 2018 bis 2021 sowie die Bescheide für Körperschaftsteuervorauszahlungen und Gewerbesteuermessbeträge für Zwecke der Vorauszahlungen für 2022 und ab 2023 abgewiesen hat. ",
            "Der Vorstand der Gigaset AG hat heute beschlossen, wegen Zahlungsunfähigkeit einen Antrag auf Eröffnung eines Regelinsolvenzverfahrens für die Gigaset AG sowie einen Antrag auf Eröffnung eines Insolvenzverfahrens in Eigenverwaltung für deren mittelbare Tochtergesellschaft Gigaset Communications GmbH beim zuständigen Amtsgericht Münster zu stellen.",
            "Comet passt Jahresprognose nach Q3 unter Erwartungen an",
            "Maternus-Kliniken AG von Cyberangriff betroffen",
            ]
labels = 5*["positive"] + 5*["neutral"] + 5*["negative"]
testData = pd.DataFrame({"sentence":sentences,"label":labels})
# testData = testData.reset_index()
# testData = testData.rename(columns = {0:"Sentences","index":"True Label"})
# testData = testData[["Sentences","True Label"]]

#%% Run models

def runModel(modelName, modelPath, tokenizerName, data):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"],max_length=512, padding="max_length", truncation=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
    model = AutoModelForSequenceClassification.from_pretrained(modelPath,
        num_labels=3)
    tokenizedSentences = Dataset.from_pandas(data).map(tokenize_function, batched=True)
    tokenizedSentences = tokenizedSentences.remove_columns(["sentence"])
    tokenizedSentences.set_format("torch")
    
    model.to(device)
    model.eval()
    tokenizedSentences = {k: tokenizedSentences[k].to(device) for k in tokenizedSentences.features.keys()}
    with torch.no_grad():
        outputs = model(**tokenizedSentences)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()
    return predictions
    

#%% Model 1
# Find best model
#t = pd.read_pickle(r"Q:\Forschung\AA_TextanalysisTools\GermanFinBERT\FinetuneModels\Benchmarks\gottbert-base\AdHocMultilabel\test_results.pkl")
modelName = "gottbert-base"
modelPath = r"..\Benchmarks\gottbert-base\Sentiment\lr_2e-05num_epochs_5\seed_0"
tokenizerName = "uklfr/gottbert-base"
predictions = runModel(modelName, modelPath, tokenizerName, testData["sentence"].to_frame())
testData[modelName] = [ classes[x] for x in predictions]

#%% Model 2 
modelName = "GermanFinBertSC"
modelPath = r"..\GermanFinBERT\german-fin-hf-bert-optimized-continue-ba174000\Sentiment\lr_3e-05num_epochs_5\seed_0"
tokenizerName = "bert-base-german-cased"
predictions = runModel(modelName, modelPath, tokenizerName, testData["sentence"].to_frame())
testData[modelName] = [ classes[x] for x in predictions]

#%% 
testData.to_excel("adHoSentimentTestResults.xlsx")