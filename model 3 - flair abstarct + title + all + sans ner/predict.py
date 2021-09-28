import os
import re
import json
import random
import argparse
from collections import Counter
from tqdm import tqdm

from datetime import datetime

import pandas as pd

# Before Flair 0.9
#   Refer to https://github.com/flairNLP/flair/issues/2426 for upgrading this script to Flair 0.9
from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence

parser = argparse.ArgumentParser(description='BioCreative 7 Task 5')
parser.add_argument('-output', type=str, default="./10_2021-09-11-13-55-01/final-model.pt", help='The output root directory for the model')
args = parser.parse_args()

classes = ['Treatment', 'Mechanism', 'Prevention', 'Case Report', 'Diagnosis', 'Transmission', 'Epidemic Forecasting']
classes_output = ['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report']

tars = TARSClassifier.load(args.output)
tars.switch_to_task("task_v")
tars.multi_label_threshold = 0.0
tars.multi_label = True

sentences = pd.read_csv("../BC7-LitCovid-Test.csv")

index = 0
output = []

# Tokenize the input text
def tokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    
    return text

for s in tqdm(sentences.itertuples()):

    if str(s.keywords) != "nan":
        keywords = [k for k in str(s.keywords).split(";")]
        keywords = " . ".join(keywords) + " . "
    else:
        keywords = ""

    tokenized = tokenizer(keywords + str(s.title) + " . " + str(s.abstract))
    
    res = Sentence(tokenized)
    tars.predict(res)

    sequence = []

    # For each expected class
    for current_class in classes_output:

        # print(res.to_dict()["labels"])

        # Check in the labels for it
        for label in res.to_dict()["labels"]:

            # If found
            if label["value"] == current_class:

                # Add it to the output sequence
                sequence.append('{:.10f}'.format(label["confidence"]))
    
    output.append(str(index) + "," + ",".join(sequence))
    index += 1

# Add header
content = "PMID,Treatment,Diagnosis,Prevention,Mechanism,Transmission,Epidemic Forecasting,Case Report\n"
content += "\n".join(output)

# Current date formatted
CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

f = open(str(CURRENT_DATE) + ".csv", "w")
f.write(content)
f.close()