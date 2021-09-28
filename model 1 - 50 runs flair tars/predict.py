import os
import re
import json
import random
import argparse
from collections import Counter
from tqdm import tqdm

import pandas as pd

from datetime import datetime

from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence

parser = argparse.ArgumentParser(description='BioCreative 7 Task 5')
parser.add_argument('-output', type=str, default="./", help='The output root directory for the model')
args = parser.parse_args()

classes = ['Treatment', 'Mechanism', 'Prevention', 'Case Report', 'Diagnosis', 'Transmission', 'Epidemic Forecasting']
classes_output = ['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report']

tars = TARSClassifier.load(args.output + "50_2021-07-15-14-47-08-87.15/final-model.pt")
tars.switch_to_task("task_v")
tars.multi_label_threshold = 0.0
tars.multi_label = True

sentences = pd.read_csv("../BC7-LitCovid-Test.csv", usecols=['abstract']).T.values.tolist()[0]

i = 0
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

for s in tqdm(sentences):
    
    res = Sentence(
        tokenizer(str(s))
    )
    tars.predict(res)

    sequence = []

    # For each expected class
    for current_class in classes_output:

        # Check in the labels for it
        for label in res.to_dict()["labels"]:

            # If found
            if label["value"] == current_class:

                # Add it to the output sequence
                sequence.append('{:.10f}'.format(label["confidence"]))
    
    output.append(str(i) + "," + ",".join(sequence))
    i += 1

# Add header
content = "PMID,Treatment,Diagnosis,Prevention,Mechanism,Transmission,Epidemic Forecasting,Case Report\n"
content += "\n".join(output)

f = open("predictions.txt", "w")
f.write(content)
f.close()