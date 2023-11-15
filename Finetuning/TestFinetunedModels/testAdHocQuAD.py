# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:58:01 2023

@author: scherrmann
"""
from transformers import AutoTokenizer
from transformers import default_data_collator
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import sys
sys.path.append("..")
from squadUtil import prepare_validation_features,  postprocess_qa_predictions

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

topics = np.array(["Earnings","SEO","Management","Guidance","Gewinnwarnung","Beteiligung","Dividende","Restructuring","Debt","Law","Großauftrag","Squeeze","Insolvenzantrag","Insolvenzplan","Delay","Split","Pharma_Good","Rückkauf","Real_Invest","Delisting"])
context = ["Die Bruttomarge des Unternehmens verbesserte sich im dritten Quartal um 0,2 Prozentpunkte auf 49,3 % (2022: 49,1 %).",
            "Der Vorstand der Consulting Team Holding Aktiengesellschaft hat am 29. August 2023 mit Zustimmung des Aufsichtsrats der Gesellschaft am 06. September 2023 beschlossen, das Grundkapital der Gesellschaft von zurzeit EUR 9. 648.000,- gegen Bareinlage um bis zu EUR 603.000,- auf bis zu EUR 10.251.000,- durch Ausgabe von bis zu 603.000 neuen, auf den Inhaber lautenden Stückaktien zu erhöhen.",
            "Der Vorstand der The Social Chain AG hat heute ein von allen drei Mitgliedern des Aufsichtsrates, Herrn Stephan Brunke, Herrn Sebastian Stietzel und Frau Henrike Luszick, unterzeichnetes Schreiben erhalten, in welchem diese ihr jeweiliges Amt als Mitglieder des Aufsichtsrates mit sofortiger Wirkung niederlegen. ",
            "Unter Berücksichtigung der positiven Auswirkungen der Verkäufe von Yeezy Produkten im zweiten und dritten Quartal, der potenziellen Abschreibung des übrigen Yeezy Bestands in Höhe von nun etwa 300 Mio. € (bisherige Prognose: 400 Mio. €) sowie von Einmalkosten im Zusammenhang mit der strategischen Prüfung von bis zu 200 Mio. € (unverändert), geht das Unternehmen nun davon aus, für das Geschäftsjahr 2023 ein negatives Betriebsergebnis in Höhe von etwa 100 Mio. € zu berichten (bisherige Prognose: negatives Betriebsergebnis in Höhe von 450 Mio. €).",
            "Die Steuler Fliesengruppe AG (ISIN DE0006770001) muss die für das laufende Geschäftsjahr 2023 im Rahmen des Geschäftsberichts 2022 abgegebene Prognose der Erreichung eines Konzernjahresüberschusses in einer Bandbreite von 3,2 bis 3,7 Mio. € zurücknehmen.",
            "Die Stabilus SE hat eine Vereinbarung zum Erwerb des Industrial-Automation-Spezialisten DESTACO von dem amerikanischen Industriegüterkonzern Dover Corporation unterzeichnet.",
            "Zukünftig soll jedoch je Geschäftsjahr eine Mindestdividende in Höhe von EUR 2,00 je dividendenberechtigter Aktie an die Aktionärinnen und Aktionäre ausgeschüttet werden.",
            "Die HÖRMANN Industries GmbH gibt bekannt, dass die HÖRMANN Automotive GmbH am Vorabend einen Vertrag zum Verkauf von 100 % der Anteile an der HÖRMANN Automotive Eislingen GmbH unterzeichnet hat.",
            "Die Geschäftsführung der Neue ZWL Zahnradwerk Leipzig GmbH hat heute die Emission einer neuen Unternehmensanleihe (ISIN: DE000A351XF8) mit einem Zinssatz von 9,5 % p.a. und einem Volumen von bis zu 15 Mio. Euro beschlossen.",
            "Dem Vorstand der Deutsche Konsum REIT-AG ist heute bekannt geworden, dass das Finanzgericht Berlin-Brandenburg den Antrag der DKR auf Aussetzung der Vollziehung der Körperschaftsteuerbescheide für die Jahre 2016 bis 2021, der Gewerbesteuermessbetragsbescheide für die Jahre 2018 bis 2021 sowie die Bescheide für Körperschaftsteuervorauszahlungen und Gewerbesteuermessbeträge für Zwecke der Vorauszahlungen für 2022 und ab 2023 abgewiesen hat.",
            "Der Auftrag hat ein Gesamtvolumen von rund 4,9 Mio. € und wird größtenteils erst im Geschäftsjahr 2024 umsatzwirksam.",
            "Die Adler Group S.A. („Adler Group”) hat heute gegenüber der ADLER Real Estate Aktiengesellschaft ihr förmliches Verlangen vom 23. Juni 2022 hinsichtlich der Übertragung der Aktien der Minderheitsaktionäre der ADLER Real Estate Aktiengesellschaft auf die Adler Group gemäß § 327a Abs. 1 Satz 1 AktG bestätigt und konkretisierend mitgeteilt, dass sie die den Minderheitsaktionären als Gegenleistung für die Übertragung ihrer Aktien zu zahlende Barabfindung auf EUR 8,76 je Aktie der ADLER Real Estate Aktiengesellschaft festgelegt hat.",
            "Der Vorstand der Gigaset AG hat heute beschlossen, wegen Zahlungsunfähigkeit einen Antrag auf Eröffnung eines Regelinsolvenzverfahrens für die Gigaset AG sowie einen Antrag auf Eröffnung eines Insolvenzverfahrens in Eigenverwaltung für deren mittelbare Tochtergesellschaft Gigaset Communications GmbH beim zuständigen Amtsgericht Münster zu stellen.",
            "Die MagForce AG (Frankfurt, Scale, Xetra: MF6, ISIN: DE000A0HGQF5) ) informiert, dass Herr Rüdiger Wienberg von der Kanzlei hww hermann wienberg wilhelm Insolvenzverwalter Partnerschaft vom Amtsgericht Berlin-Charlottenburg als dem zuständigen Insolvenzgericht als vorläufiger Insolvenzverwalter für die MagForce AG bestellt worden ist.",
            "Das Unternehmen hat eine Verlängerung der Frist für die Veröffentlichung des Halbjahresberichts 2022 beantragt.",
            "Die ordentliche Hauptversammlung der Alexanderwerk AG hat heute unter anderem beschlossen, das Grundkapital der Gesellschaft i.H.v. EUR 4.680.000, zerlegt in 1.800.000 Stückaktien ohne Nennbetrag, neu einzuteilen und die Satzung entsprechend zu ändern.",
            "Newron gibt positive Topline-Ergebnisse von allen Patienten der klinischen Phase-II-Studie 014 mit Evenamide als Zusatztherapie bei behandlungsresistenter Schizophrenie bekannt.",
            "Der Vorstand der Scherzer & Co. AG hat heute mit Zustimmung des Aufsichtsrats beschlossen, von der auf der ordentlichen Hauptversammlung vom 27. Mai 2021 beschlossenen Ermächtigung zum Erwerb eigener Aktien Gebrauch zu machen und im Zeitraum vom 16. Oktober 2023 bis längstens zum 29. März 2024 bis zu 500.000 Aktien im Gegenwert von bis zu EUR 1 Mio. zu erwerben.",
            "Die Gesellschaft hat heute mit der Stadtsparkasse Augsburg einen notariellen Kaufvertrag über das sog. Ziegeleigrundstück abgeschlossen.",
            "Der Vorstand der Ekotechnika AG hat heute mit Zustimmung des Aufsichtsrats beschlossen, die Einbeziehung der Aktien (ISIN: DE000A161234) in den Primärmarkt der Börse Düsseldorf zu widerrufen und die endgültige Notierungseinstellung zu beantragen."]
question = ["Um wie viele Prozentpunkte verbesserte sich die Bruttomarge des Unternehmens?",
            "Wie viele neue Aktien werden maximal ausgegeben?",
            "Wann haben die drei Personen das schreiben eingereicht?",
            "Wie hoch sind die möglichen Abschreibungen des übrigen Yeezy Bestands?",
            "Wer hat die Prognose zurückgezogen?",
            "Wer heißt der Industrial-Automation-Spezialist?",
            "Wie hoch ist die Dividendenausschüttung in Zukunft mindestens?",
            "Wo befindet sich die HÖRMANN Automotive?",
            "Wie lautet die Wertpapierkennnummer der genannten Anleihe?",
            "Für welchen Zeitraum wurde die Aussetzung der Vollziehung der Gewerbesteuermessbetragsbescheide beantragt?",
            "Wie hoch ist das Auftragsvolumen insgesamt?",
            "Wie hoch ist die Abfindung der Minderheitsaktionäre pro Aktie?",
            "Bei wem wurde ein Antrag auf Insolvenz eingereicht?",
            "Wie heißt die Person, die die Insolvenz verwalten soll?",
            "Was wurde beantragt?",
            "Wie viele Aktien werden neu eingeteilt?",
            "Welche Krankheit kann mit Evenamide behandelt werden?",
            "Was ist das Datum der Hauptversammlung?",
            "Mit wem hat die Gesellschaft einen Kaufvertrag geschlossen?",
            "Auf welchen Handelsplatz bezieht sich das Delisting?"]
answers = [{"answer_start":[72],'text':["0,2 Prozentpunkte"]},
           {"answer_start":[262],'text':["603.000"]},
           {"answer_start":[41],'text':["heute"]},
           {"answer_start":[193],'text':["300 Mio. €"]},
           {"answer_start":[4],'text':["Steuler Fliesengruppe AG"]},
           {"answer_start":[88],'text':["DESTACO"]},
           {"answer_start":[73],'text':["EUR 2,00"]},
           {"answer_start":[23],'text':["Eislingen"]},
           {"answer_start":[121],'text':["DE000A351XF8"]},
           {"answer_start":[274],'text':["2018 bis 2021"]},
           {"answer_start":[38],'text':["rund 4,9 Mio. €"]},
           {"answer_start":[460],'text':["EUR 8,76"]},
           {"answer_start":[318],'text':["Amtsgericht Münster"]},
           {"answer_start":[91],'text':["Rüdiger Wienberg"]},
           {"answer_start":[25],'text':["Verlängerung der Frist für die Veröffentlichung des Halbjahresberichts 2022"]},
           {"answer_start":[158],'text':["1.800.000"]},
           {"answer_start":[47],'text':["Schizophrenie"]},
           {"answer_start":[141],'text':["27. Mai 2021"]},
           {"answer_start":[35],'text':["Stadtsparkasse Augsburg"]},
           {"answer_start":[160],'text':["Börse Düsseldorf"]}]
idx = list(range(len(context)))

testData = {"id":idx,"context":context, "question": question, "answer":answers}
data = Dataset.from_dict(testData)
testData = pd.DataFrame(testData)
testData["answer"] = [x["text"][0] for x in testData["answer"]]
testData = testData.rename(columns={"answer" : "True Answer"})
testData = testData.drop(["id"],axis=1)
#%% Run models

def runModel(modelName, modelPath, tokenizerName, data):
    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
    pad_on_right = tokenizer.padding_side == "right"
    test_features = data.map(
        prepare_validation_features,
        batched=True,
        remove_columns=data.column_names, 
        fn_kwargs={"tokenizer":tokenizer, "pad_on_right":pad_on_right, 
                   "max_length":max_length, "doc_stride":doc_stride}
    )
    model =  AutoModelForQuestionAnswering.from_pretrained(modelPath)
    args = TrainingArguments("tmp")
    data_collator = default_data_collator
    trainer = Trainer(
        model,
        args,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    raw_predictions = trainer.predict(test_features)
    test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys()))        
    final_predictions = postprocess_qa_predictions(data, test_features, raw_predictions.predictions, tokenizer)
    return final_predictions
    

#%% Model 1
modelName = "gottbert-base"
modelPath = r"..\Benchmarks\gottbert-base\AdhocQuAD\seed_0"
tokenizerName = "uklfr/gottbert-base"
predictions = runModel(modelName, modelPath, tokenizerName,data)
predictions = list(predictions.values())
testData[modelName] = predictions

#%% Model 2 
modelName = "GermanFinBertFP"
modelPath = r"..\GermanFinBERT\german-fin-gbert-optimized-3-fp-512-small-lr-ba10400\AdhocQuAD\seed_3"
tokenizerName = "deepset/gbert-base"
predictions = runModel(modelName, modelPath, tokenizerName,data)
predictions = list(predictions.values())
testData[modelName] = predictions

#%% 
testData.to_excel("adHocQuadTestResults.xlsx")