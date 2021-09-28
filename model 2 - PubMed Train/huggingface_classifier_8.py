# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from datetime import datetime
from threading import current_thread

import torch
import pandas as pd

import transformers
from transformers import (BertForSequenceClassification, TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertTokenizerFast, BertForSequenceClassification

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

parser = argparse.ArgumentParser(description='BC7 LitCovid')
parser.add_argument('--train', type=str, default="./data/BC7-LitCovid-Train.csv", help='The training input files')
parser.add_argument('--name', type=str, default="BC7-LitCovid", help='The custom name of the experience')
parser.add_argument('--output', type=str, default="./out/models/", help='The output root directory for the model')
parser.add_argument('--model_ckpt', type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", help='The model checkpoint')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
parser.add_argument('--batch_size', type=int, default=12, help='Size of the batch size')
args = parser.parse_args()

CURRENT_DATE = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

model_name_short = args.model_ckpt.replace("/","-").upper()
epochs_nbr = args.epochs
output_model = os.path.join(args.output, model_name_short + "_" + args.name + "_" + "en".upper() + "/" + str(epochs_nbr) + "_" + CURRENT_DATE)
print(output_model)

# Classes keywords
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

mlb = MultiLabelBinarizer()

def textTokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    return text

def getBow(documents, mode):

    docs = [text for text, labels in documents]
    all_tags = [labels for text, labels in documents]

    # Apply Transformation
    if mode == "train":
        bow_tags = mlb.fit_transform(all_tags)
    else:
        bow_tags = mlb.transform(all_tags)

    return docs, bow_tags, list(mlb.classes_)

def getCorpora(data, mode):

    sentences = []

    # For each row
    for i in data.itertuples():

        if str(i.keywords) != "nan":
            keywords = [k for k in str(i.keywords).split(";")]
            keywords = " . ".join(keywords) + " . "
        else:
            keywords = ""

        full = keywords + str(i.title) + " . " + str(i.abstract)

        tokenized = textTokenizer(full.lower())
        token_custom = tokenized

        anns = i.label.split(";")

        if mode == "train":

            # For each label
            for current_label in anns:

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
        s = (token_custom, [])

        # For each label, add it to the list of labels
        for label in i.label.split(";"):
            s[1].append(label)

        sentences.append(s)                    

    return sentences, mode

def getCorporaBow(data,mode):
    sentences, mode = getCorpora(data,mode)
    print(len(sentences))
    docs, tags, labels = getBow(sentences, mode)
    return docs, tags, labels, sentences

contentTrain = pd.read_csv(args.train, usecols=['journal','title','abstract','keywords','pub_type','label'])
print("CSV file Train loaded!")

# Load the test csv with fake labels to keep dimension consistency
contentTest = pd.read_csv("./data/BC7-LitCovid-TestFull.csv", usecols=['journal','title','abstract','keywords','pub_type','label'])
print("CSV file TestFull loaded!")

# Load the Train corpora
docs, bow_tags, labels, sentences_train = getCorporaBow(contentTrain,"train")

# Load the Test corpora
docs_test, bow_tags_test, labels_test, sentences_test = getCorporaBow(contentTest,"test")

# create labels column
label_cols = labels

contentTrain["labels"] = list(bow_tags)
contentTrain["comment_text"] = list(docs)

contentTest["labels"] = list(bow_tags_test)
contentTest["comment_text"] = list(docs_test)

df_train = contentTrain
df_test = contentTest

model_ckpt = args.model_ckpt
tokenizer = BertTokenizerFast.from_pretrained(model_ckpt, do_lower_case=True)

max_length = 512
train_encodings = tokenizer(df_train["comment_text"].values.tolist(), truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(df_test["comment_text"].values.tolist(), truncation=True, padding=True, max_length=max_length)

train_labels = df_train["labels"].values.tolist()
test_labels = df_test["labels"].values.tolist()

class BC7Dataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BC7Dataset(train_encodings, train_labels)
test_dataset = BC7Dataset(test_encodings, test_labels)

class BertForMultilabelSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
      super().__init__(config)

    def forward(self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

# Number of labels
num_labels=len(label_cols)

# Load the label of the GPU
model = BertForMultilabelSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to('cuda')

# Multilabel Accuracy
def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True): 
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid: 
      y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true.bool()).float().mean().item()

# Compute Metrics
def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    acc = accuracy_thresh(predictions, labels)

    y_pred, y_true = eval_pred
    
    thresh=0.5
    sigmoid=True

    y_pred = torch.from_numpy(y_pred)
    print(y_pred)

    indexes = [nbr for nbr in range(len(y_pred))]

    custom_df = pd.DataFrame(columns=label_cols)
    custom_df.index.name = "PMID"

    # Insert all lines
    for ind, yp in zip(indexes, y_pred):
        custom_df.loc[ind] = [float(elem) for elem in list(yp)]
    
    # Reorder
    custom_df = custom_df[["Treatment","Diagnosis","Prevention","Mechanism","Transmission","Epidemic Forecasting","Case Report"]]

    output_csv_dir = "./out/csv/" + model_name_short + "_" + "BC7-LitCovid" + "_" + "en".upper() + "/"

    # Check if the directory exist
    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)

    # Build the output path
    output_csv = os.path.join(output_csv_dir + str(epochs_nbr) + "_" + str(CURRENT_DATE) + ".csv")
    custom_df.to_csv(output_csv, index=True) 

    if sigmoid: 
      y_pred = y_pred.sigmoid()

    y_pred = (y_pred > thresh)
    y_pred = 1*y_pred

    return {
        'accuracy_thresh': 0,
    }

batch_size = args.batch_size
logging_steps = len(train_dataset) // batch_size

args = TrainingArguments(
    output_dir=output_model,
    learning_rate=5e-5,
    # learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs_nbr,
    weight_decay=0.01,
    logging_steps=logging_steps
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.evaluate()

# python huggingface_classifier_8.py --name="TestFinalTrain+Dev" --batch_size=8 --epochs=14 --train="./data/BC7-LitCovid-Train+Dev.csv"