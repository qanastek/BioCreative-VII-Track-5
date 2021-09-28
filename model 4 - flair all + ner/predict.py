import os
import re
import json
import random
import argparse
from collections import Counter
from tqdm import tqdm

from datetime import datetime

import pandas as pd

from flair.models import TARSClassifier
# from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence

parser = argparse.ArgumentParser(description='BioCreative 7 Task 5')
parser.add_argument('-output', type=str, default="./", help='The output root directory for the model')
args = parser.parse_args()

classes = ['Treatment', 'Mechanism', 'Prevention', 'Case Report', 'Diagnosis', 'Transmission', 'Epidemic Forecasting']
classes_output = ['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report']

tars = TARSClassifier.load(args.output + "10_2021-09-11-13-55-12/final-model.pt")
tars.switch_to_task("task_v")
tars.multi_label_threshold = 0.0
tars.multi_label = True

# # Treatment
# sentence_1 = "Studies have shown that infection, excessive coagulation, cytokine storm, leukopenia, lymphopenia, hypoxemia and oxidative stress have also been observed in critically ill Severe Acute Respiratory Syndrome coronavirus 2 (SARS-CoV-2) patients in addition to the onset symptoms. There are still no approved drugs or vaccines. Dietary supplements could possibly improve the patient's recovery. Omega-3 fatty acids, specifically eicosapentaenoic acid (EPA) and docosahexaenoic acid (DHA), present an anti-inflammatory effect that could ameliorate some patients need for intensive care unit (ICU) admission. EPA and DHA replace arachidonic acid (ARA) in the phospholipid membranes. When oxidized by enzymes, EPA and DHA contribute to the synthesis of less inflammatory eicosanoids and specialized pro-resolving lipid mediators (SPMs), such as resolvins, maresins and protectins. This reduces inflammation. In contrast, some studies have reported that EPA and DHA can make cell membranes more susceptible to non-enzymatic oxidation mediated by reactive oxygen species, leading to the formation of potentially toxic oxidation products and increasing the oxidative stress. Although the inflammatory resolution improved by EPA and DHA could contribute to the recovery of patients infected with SARS-CoV-2, Omega-3 fatty acids supplementation cannot be recommended before randomized and controlled trials are carried out."
# sentences = [sentence_1 for i in range(2551)]

sentences = pd.read_csv("../BC7-LitCovid-Test.csv")

index = 0
output = []

# Classes Tokens
raw_per_class_keywords = json.loads(open("threshold_0-65.json", "r").read())
per_class_keywords = {}
for class_key in raw_per_class_keywords:    
    if class_key not in per_class_keywords:
        per_class_keywords[class_key] = []    
    per_class_keywords[class_key] = "(" + "|".join([" " + re.escape(raw.lstrip().strip().lower()) + " " for raw in raw_per_class_keywords[class_key]]) + ")"
per_class_replacement = {
    "Treatment":            " [{\g<0>}] ",
    "Mechanism":            " [|\g<0>|] ",
    "Prevention":           " |{\g<0>}| ",
    "Case Report":          " |[\g<0>]| ",
    "Diagnosis":            " {|\g<0>|} ",
    "Transmission":         " [#\g<0>#] ",
    "Epidemic Forecasting": " #{\g<0>}# ",
}
print(per_class_keywords.keys())
print(per_class_replacement.values())

# Tokenize the input text
def tokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    
    # # Remove years
    # text = re.sub(years, " ", text)
    
    # For each label
    for current_label in per_class_replacement.keys():

        text = re.sub(
            per_class_keywords[current_label],
            per_class_replacement[current_label],
            text
        )

    return text

for s in tqdm(sentences.itertuples()):

    if str(s.keywords) != "nan":
        keywords = [k for k in str(s.keywords).split(";")]
        keywords = " . ".join(keywords) + " . "
    else:
        keywords = ""

    tokenized = tokenizer(keywords + str(s.title) + " . " + str(s.abstract))
    
    # my_labels = list(per_class_replacement.keys())
    res = Sentence(tokenized)
    tars.predict(res)

    # print("res")
    # print(res)

    # print("res.to_dict()['labels']")
    # print(res.to_dict()["labels"])

    sequence = {}

    # Fill with 0.0
    for c in classes_output:
        sequence[c] = "0.0"

    # For each expected class
    for current_class in classes_output:

        # print(res.to_dict()["labels"])

        # Check in the labels for it
        for label in res.to_dict()["labels"]:

            # If found
            if label["value"] == current_class:

                # Add it to the output sequence
                sequence[current_class] = '{:.10f}'.format(label["confidence"])
    
        # print(str(index) + "," + ",".join(sequence))

    # print(sequence.values())
    # print(list(sequence.values()))

    output.append(str(index) + "," + ",".join(list(sequence.values())))
    index += 1

# Add header
content = "PMID,Treatment,Diagnosis,Prevention,Mechanism,Transmission,Epidemic Forecasting,Case Report\n"
# Merge lines
# print(content)
content += "\n".join(output)
# Print lines
# print(content)

# Current date formatted
CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

f = open(str(CURRENT_DATE) + ".csv", "w")
f.write(content)
f.close()

# tars.predict_zero_shot(res, classes)
# print(res)