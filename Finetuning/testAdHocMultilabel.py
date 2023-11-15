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

topics = np.array(["Earnings","SEO","Management","Guidance","Gewinnwarnung","Beteiligung","Dividende","Restructuring","Debt","Law","Großauftrag","Squeeze","Insolvenzantrag","Insolvenzplan","Delay","Split","Pharma_Good","Rückkauf","Real_Invest","Delisting"])
testData = {"Earnings":" Die Bruttomarge des Unternehmens verbesserte sich im dritten Quartal um 0,2 Prozentpunkte auf 49,3 % (2022: 49,1 %).",
            "SEO":"Der Vorstand der Consulting Team Holding Aktiengesellschaft hat am 29. August 2023 mit Zustimmung des Aufsichtsrats der Gesellschaft am 06. September 2023 beschlossen, das Grundkapital der Gesellschaft von zurzeit EUR 9. 648.000,- gegen Bareinlage um bis zu EUR 603.000,- auf bis zu EUR 10.251.000,- durch Ausgabe von bis zu 603.000 neuen, auf den Inhaber lautenden Stückaktien zu erhöhen.",
            "Management":"Der Vorstand der The Social Chain AG hat heute ein von allen drei Mitgliedern des Aufsichtsrates, Herrn Stephan Brunke, Herrn Sebastian Stietzel und Frau Henrike Luszick, unterzeichnetes Schreiben erhalten, in welchem diese ihr jeweiliges Amt als Mitglieder des Aufsichtsrates mit sofortiger Wirkung niederlegen. ",
            "Guidance":"Unter Berücksichtigung der positiven Auswirkungen der Verkäufe von Yeezy Produkten im zweiten und dritten Quartal, der potenziellen Abschreibung des übrigen Yeezy Bestands in Höhe von nun etwa 300 Mio. € (bisherige Prognose: 400 Mio. €) sowie von Einmalkosten im Zusammenhang mit der strategischen Prüfung von bis zu 200 Mio. € (unverändert), geht das Unternehmen nun davon aus, für das Geschäftsjahr 2023 ein negatives Betriebsergebnis in Höhe von etwa 100 Mio. € zu berichten (bisherige Prognose: negatives Betriebsergebnis in Höhe von 450 Mio. €).",
            "Gewinnwarnung":"Die Steuler Fliesengruppe AG (ISIN DE0006770001) muss die für das laufende Geschäftsjahr 2023 im Rahmen des Geschäftsberichts 2022 abgegebene Prognose der Erreichung eines Konzernjahresüberschusses in einer Bandbreite von 3,2 bis 3,7 Mio. € zurücknehmen.",
            "Beteiligung":"Die Stabilus SE hat eine Vereinbarung zum Erwerb des Industrial-Automation-Spezialisten DESTACO von dem amerikanischen Industriegüterkonzern Dover Corporation unterzeichnet.",
            "Dividende":"Zukünftig soll jedoch je Geschäftsjahr eine Mindestdividende in Höhe von EUR 2,00 je dividendenberechtigter Aktie an die Aktionärinnen und Aktionäre ausgeschüttet werden.",
            "Restructuring":" Die HÖRMANN lndustries GmbH gibt bekannt, dass die HÖRMANN Automotive GmbH am Vorabend einen Vertrag zum Verkauf von 100 % der Anteile an der HÖRMANN Automotive Eislingen GmbH unterzeichnet hat.",
            "Debt":"Die Geschäftsführung der Neue ZWL Zahnradwerk Leipzig GmbH hat heute die Emission einer neuen Unternehmensanleihe (ISIN: DE000A351XF8) mit einem Zinssatz von 9,5 % p.a. und einem Volumen von bis zu 15 Mio. Euro beschlossen.",
            "Law":"Dem Vorstand der Deutsche Konsum REIT-AG ist heute bekannt geworden, dass das Finanzgericht Berlin-Brandenburg den Antrag der DKR auf Aussetzung der Vollziehung der Körperschaftsteuerbescheide für die Jahre 2016 bis 2021, der Gewerbesteuermessbetragsbescheide für die Jahre 2018 bis 2021 sowie die Bescheide für Körperschaftsteuervorauszahlungen und Gewerbesteuermessbeträge für Zwecke der Vorauszahlungen für 2022 und ab 2023 abgewiesen hat. ",
            "Großauftrag":"Der Auftrag hat ein Gesamtvolumen von rund 4,9 Mio. € und wird größtenteils erst im Geschäftsjahr 2024 umsatzwirksam.",
            "Squeeze":"Die Adler Group S.A. („Adler Group”) hat heute gegenüber der ADLER Real Estate Aktiengesellschaft ihr förmliches Verlangen vom 23. Juni 2022 hinsichtlich der Übertragung der Aktien der Minderheitsaktionäre der ADLER Real Estate Aktiengesellschaft auf die Adler Group gemäß § 327a Abs. 1 Satz 1 AktG bestätigt und konkretisierend mitgeteilt, dass sie die den Minderheitsaktionären als Gegenleistung für die Übertragung ihrer Aktien zu zahlende Barabfindung auf EUR 8,76 je Aktie der ADLER Real Estate Aktiengesellschaft festgelegt hat.",
            "Insolvenzantrag":"Der Vorstand der Gigaset AG hat heute beschlossen, wegen Zahlungsunfähigkeit einen Antrag auf Eröffnung eines Regelinsolvenzverfahrens für die Gigaset AG sowie einen Antrag auf Eröffnung eines Insolvenzverfahrens in Eigenverwaltung für deren mittelbare Tochtergesellschaft Gigaset Communications GmbH beim zuständigen Amtsgericht Münster zu stellen.",
            "Insolvenzplan":"Die MagForce AG (Frankfurt, Scale, Xetra: MF6, ISIN: DE000A0HGQF5) ) informiert, dass Herr Rüdiger Wienberg von der Kanzlei hww hermann wienberg wilhelm Insolvenzverwalter Partnerschaft vom Amtsgericht Berlin-Charlottenburg als dem zuständigen Insolvenzgericht als vorläufiger Insolvenzverwalter für die MagForce AG bestellt worden ist.",
            "Delay":"Das Unternehmen hat eine Verlängerung der Frist für die Veröffentlichung des Halbjahresberichts 2022 beantragt.",
            "Split":"Die ordentliche Hauptversammlung der Alexanderwerk AG hat heute unter anderem beschlossen, das Grundkapital der Gesellschaft i.H.v. EUR 4.680.000, zerlegt in 1.800.000 Stückaktien ohne Nennbetrag, neu einzuteilen und die Satzung entsprechend zu ändern.",
            "Pharma_Good":"Newron gibt positive Topline-Ergebnisse von allen Patienten der klinischen Phase-II-Studie 014 mit Evenamide als Zusatztherapie bei behandlungsresistenter Schizophrenie bekannt.",
            "Rückkauf":"Der Vorstand der Scherzer & Co. AG hat heute mit Zustimmung des Aufsichtsrats beschlossen, von der auf der ordentlichen Hauptversammlung vom 27. Mai 2021 beschlossenen Ermächtigung zum Erwerb eigener Aktien Gebrauch zu machen und im Zeitraum vom 16. Oktober 2023 bis längstens zum 29. März 2024 bis zu 500.000 Aktien im Gegenwert von bis zu EUR 1 Mio. zu erwerben.",
            "Real_Invest":"Die Gesellschaft hat heute mit der Stadtsparkasse Augsburg einen notariellen Kaufvertrag über das sog. Ziegeleigrundstück abgeschlossen.",
            "Delisting":"Der Vorstand der Ekotechnika AG hat heute mit Zustimmung des Aufsichtsrats beschlossen, die Einbeziehung der Aktien (ISIN: DE000A161234) in den Primärmarkt der Börse Düsseldorf zu widerrufen und die endgültige Notierungseinstellung zu beantragen."}
testData = pd.Series(testData).to_frame()
testData = testData.reset_index()
testData = testData.rename(columns = {0:"Sentences","index":"True Label"})
testData = testData[["Sentences","True Label"]]

#%% Run models

def runModel(modelName, modelPath, tokenizerName, data):
    def tokenize_function(examples):
        return tokenizer(examples["Sentences"],max_length=512, padding="max_length", truncation=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
    model = AutoModelForSequenceClassification.from_pretrained(modelPath,
        num_labels=len(topics),
        problem_type="multi_label_classification")
    tokenizedSentences = Dataset.from_pandas(data).map(tokenize_function, batched=True)
    tokenizedSentences = tokenizedSentences.remove_columns(["Sentences"])
    tokenizedSentences.set_format("torch")
    
    model.to(device)
    model.eval()
    tokenizedSentences = {k: tokenizedSentences[k].to(device) for k in tokenizedSentences.features.keys()}
    with torch.no_grad():
        outputs = model(**tokenizedSentences)
    logits = outputs.logits
    predictions = (torch.sigmoid(logits)>0.6).cpu().numpy() # predictions = nobs x ntopics
    predictions = [",".join(list(topics[predictions[idx,:]])) for idx in range(len(predictions))]
    predictions = pd.Series(predictions, name = modelName)
    return predictions
    

# Model 1
# Find best model
#t = pd.read_pickle(r"Q:\Forschung\AA_TextanalysisTools\GermanFinBERT\FinetuneModels\Benchmarks\gottbert-base\AdHocMultilabel\test_results.pkl")
modelName = "gottbert-base"
modelPath = r"..\Benchmarks\gottbert-base\AdHocMultilabel\1"
tokenizerName = "uklfr/gottbert-base"
predictions = runModel(modelName, modelPath, tokenizerName, testData["Sentences"].to_frame())
testData[modelName] = predictions

# Model 2 
modelName = "GermanFinBertFP"
modelPath = r"..\GermanFinBERT\german-fin-gbert-optimized-3-fp-512-small-lr-ba10400\AdHocMultilabel\2"
tokenizerName = "deepset/gbert-base"
predictions = runModel(modelName, modelPath, tokenizerName, testData["Sentences"].to_frame())
testData[modelName] = predictions

#%% 
testData.to_excel("adHocMultilabelTestResults.xlsx")