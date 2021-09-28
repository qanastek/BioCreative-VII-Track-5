import re
import os
import json
import argparse
from datetime import datetime

import pandas as pd

from flair.models import TARSClassifier
from flair.embeddings import *
from flair.data import Corpus, Sentence
from flair.trainers import ModelTrainer
from flair.datasets import SentenceDataset

parser = argparse.ArgumentParser(description='Next Drug Prediction (NDP)')
parser.add_argument('--input', type=str, default="data/BC7-LitCovid-", help='The input file')
parser.add_argument('--output', type=str, default="./out/models/", help='The output root directory for the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args()

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

# Ratios
RATIO_DEV = 0.90
RATIO_TEST = 0.10

# Training Configuration
LR = 0.02
MIN_BATCH_SIZE = 1
EPOCHS = args.epochs
TRAIN_WITH_DEV = True

# Current date formatted
CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

output = os.path.join(args.output, "Flair_Classifier".upper() + "_BC7-LitCovid_Joined+NER_" + "en".upper() + "/" + str(args.epochs) + "_" + CURRENT_DATE)
print(output)

contentTrainTest = pd.read_csv(args.input + "Train.csv", usecols=['journal','title','abstract','keywords','pub_type','label'])
print("CSV file Train/Test loaded!")

contentDev = pd.read_csv(args.input + "Dev.csv", usecols=['journal','title','abstract','keywords','pub_type','label'])
print("CSV file Dev loaded!")

# Tokenize the input text
def tokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    
    return text

def getCorpora(data,mode):

    sentences = []

    # For each row
    for i in data.itertuples():

        if str(i.keywords) != "nan":
            keywords = [k for k in str(i.keywords).split(";")]
            keywords = " . ".join(keywords) + " . "
        else:
            keywords = ""

        tokenized = tokenizer(keywords + str(i.title) + " . " + str(i.abstract))
        token_custom = tokenized

        if mode == "train":

            # For each label
            for current_label in i.label.split(";"):

                token_custom = re.sub(
                    per_class_keywords[current_label],
                    per_class_replacement[current_label],
                    token_custom
                )

        else:

            # For each label
            for current_label in per_class_replacement.keys():

                token_custom = re.sub(
                    per_class_keywords[current_label],
                    per_class_replacement[current_label],
                    token_custom
                )

        # Create a sentence
        s = Sentence(
            tokenizer(tokenized)
        )

        if mode != "test":
            
            # For each label
            for label in i.label.split(";"):
                s.add_label("task_v",label)

        sentences.append(s)                    

    print(len(sentences))

    return sentences

all = getCorpora(contentTrainTest,"train")
print("Corpora processed for Train!")

allDev = getCorpora(contentDev,"dev")
print("Corpora processed for Dev!")

# Indexes
CORPORA_SIZE = len(allDev)
DEV_INDEX = int(CORPORA_SIZE * RATIO_DEV)

# Both Corpora
train = all
dev   = allDev[:DEV_INDEX]
test  = allDev[DEV_INDEX:]

# Split corpora
train, dev, test = SentenceDataset(train), SentenceDataset(dev), SentenceDataset(test)

# Make a corpus with train and test split
corpus = Corpus(train=train, dev=dev, test=test)

# Make the dictionnary for the corpus
label_dict = corpus.make_label_dictionary(label_type='task_v')

# Load base TARS and setup
tars = TARSClassifier.load("tars-base")
tars.multi_label = True
tars.add_and_switch_to_new_task(
    "task_v",
    label_dictionary=label_dict,
    label_type="task_v",
)

# Train model
trainer = ModelTrainer(tars, corpus)
trainer.train(
    base_path=output,
    learning_rate=LR,
    mini_batch_size=MIN_BATCH_SIZE,
    max_epochs=EPOCHS,
    train_with_dev=TRAIN_WITH_DEV,
)