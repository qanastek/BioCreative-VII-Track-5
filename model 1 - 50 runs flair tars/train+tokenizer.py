import re
import os
import json
import random
import argparse

from datetime import datetime

import pandas as pd

from flair.models.text_classification_model import TARSClassifier
from flair.data import Corpus
from flair.data import Sentence
from flair.trainers import ModelTrainer
from flair.datasets import SentenceDataset

parser = argparse.ArgumentParser(description='Next Drug Prediction (NDP)')
parser.add_argument('-input', type=str, default="data/BC7-LitCovid-", help='The input file')
parser.add_argument('-output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('-epochs', type=int, default=1, help='Number of Epochs')
parser.add_argument('-stats', action='store_true', help='Enable statistics monitoring')
args = parser.parse_args()

# Ratios
RATIO_TRAIN = 0.85
RATIO_TEST = 0.15

# Training Configuration
LR = 0.02
MIN_BATCH_SIZE = 1
EPOCHS = args.epochs
TRAIN_WITH_DEV = True

# Current date formatted
CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

output = os.path.join(args.output, "Flair_Classifier".upper() + "_BC7-LitCovid_" + "en".upper() + "/" + str(args.epochs) + "_" + CURRENT_DATE)
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

def getCorpora(data):

    sentences = []

    # For each row
    for i in data.itertuples():

        # Create a sentence from abstract
        s = Sentence(
            tokenizer(i.abstract)
        )

        # For each label
        for label in i.label.split(";"):
            s.add_label("task_v",label)

        sentences.append(s)                    

    print(len(sentences))

    return sentences

# Load Train
all = getCorpora(contentTrainTest)
# Load Dev
allDev = getCorpora(contentDev)

# Indexes
CORPORA_SIZE = len(all)
TRAIN_INDEX = int(CORPORA_SIZE * RATIO_TRAIN)
TEST_INDEX = int(TRAIN_INDEX + CORPORA_SIZE * RATIO_TEST)

# Both Corpora
train = all[:TRAIN_INDEX]
dev   = allDev
test  = all[TRAIN_INDEX:]

# Split corpora
train, dev, test = SentenceDataset(train), SentenceDataset(dev), SentenceDataset(test)

# Make a corpus with train and test split
corpus = Corpus(train=train, dev=dev, test=test)

# Load base TARS
tars = TARSClassifier.load('tars-base')
tars.multi_label = True

# Make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task(
    "task_v",
    label_dictionary=corpus.make_label_dictionary()
)

# Initialize the text classifier trainer with your corpus
trainer = ModelTrainer(tars, corpus)

# Train model
trainer.train(
    base_path=output,
    learning_rate=LR,
    mini_batch_size=MIN_BATCH_SIZE,
    max_epochs=EPOCHS,
    train_with_dev=TRAIN_WITH_DEV,
)