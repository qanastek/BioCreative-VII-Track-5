import re
import argparse

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Multilabel Classification')
parser.add_argument('--input', type=str, default="../data/BC7-LitCovid-", help='The input file')
parser.add_argument('--output', type=str, default="out/models/", help='The output root directory for the model')
parser.add_argument('--mode', type=str, default="tf-idf", help='bow / tf_idf')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args()

# SVM Setup
num_samples_total = 10000
cluster_centers = [(5,5), (3,3)]
num_classes = len(cluster_centers)

mlb = MultiLabelBinarizer()

contentTrain = pd.read_csv(args.input + "Train+Dev.csv", usecols=['journal','title','abstract','keywords','pub_type','label'])
print("CSV file Train+Dev loaded!")
print(len(contentTrain))

# The test file with random labels to keep the same dimension
contentTest = pd.read_csv(args.input + "TestFull.csv", usecols=['journal','title','abstract','keywords','pub_type','label'])
print("CSV file Test loaded!")
print(len(contentTest))

vectorizer = None

if args.mode == "bow":

    vectorizer = CountVectorizer(
        max_features = 3200,
        min_df       = 5,
        max_df       = 0.7,
        stop_words   = stopwords.words('english')
    )
    
else:

    vectorizer = TfidfVectorizer(
        max_df=1.0,
        max_features=20000,
        min_df=0.0,
        stop_words=stopwords.words('english'),
        use_idf=True,
        ngram_range=(1,3)
    )

def tokenizer(text):
    
    # Lower + Remove date
    text = re.sub("\d\d/\d\d/\d\d\d\d", " ", text.lower())

    # Tokenize
    text = text.replace("•"," | ").replace("#"," ").replace("."," . ").replace(","," , ").replace(":"," : ").replace(";"," ; ").replace("/"," / ").replace("'"," ' ").replace('"',' " ').replace("+"," + ")
    text = text.replace("("," ( ").replace(")"," ) ").replace("["," [ ").replace("]"," ] ").replace("{"," { ").replace("}"," } ")
    text = text.replace("!"," ! ").replace("?"," ? ").replace("*"," * ").replace("@"," @ ").replace("|"," ")
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    return text

def getBow(documents, mode):

    # Documents
    docs = [text for text, labels in documents]
    # Labels
    all_tags = [labels for text, labels in documents]

    bow = None

    # Apply Transformation
    if mode == "train":
        bow = vectorizer.fit_transform(docs).toarray()
        bow_tags = mlb.fit_transform(all_tags)
    else:
        bow = vectorizer.transform(docs).toarray()
        bow_tags = mlb.transform(all_tags)

    print("Vector size:", str(len(bow[0])))
    print(list(mlb.classes_))

    return bow, bow_tags, list(mlb.classes_)

def getCorpora(data, mode):

    sentences = []

    # For each row
    for i in data.itertuples():

        if str(i.keywords) != "nan":
            keywords = [k for k in str(i.keywords).split(";")]
            keywords = " . ".join(keywords) + " . "
        else:
            keywords = ""

        # Concat keywords, title and abstract
        full = keywords + str(i.title) + " . " + str(i.abstract)

        # Lowercase and tokenize
        tokenized = tokenizer(full.lower())

        # Create a sentence
        s = (tokenized, [])

        # For each label
        for label in i.label.split(";"):
            s[1].append(label)

        sentences.append(s)                    

    print(len(sentences))
    return getBow(sentences, mode)

# TF-IDF and labels (text, [labels]) for Train
bows, bow_tags, labels = getCorpora(contentTrain,"train")
X, Y = bows, bow_tags

# TF-IDF and labels (text, [labels]) for Test
X_test, y_test, labels_test = getCorpora(contentTest,"test")

# Split into training and testing data
X_train, y_train = X, Y

# Create the SVM
svm = LinearSVC(
    max_iter     = 10000000,
    random_state = 42,
)

# Make it an Multilabel classifier
multilabel_classifier = MultiOutputClassifier(svm, n_jobs=-1)

# Fit the data to the Multilabel classifier
multilabel_classifier = multilabel_classifier.fit(X_train, y_train)

# Get predictions for test data
y_test_pred = multilabel_classifier.predict(X_test)

# Create a empty dataframe
df_output = pd.DataFrame(columns=labels)

# Fill up the dataframe
for index in range(len(y_test_pred)):
    df_output.loc[df_output.shape[0]] = [float(d) for d in y_test_pred[index]]

# Define the header
real_head = ["Treatment","Diagnosis","Prevention","Mechanism","Transmission","Epidemic Forecasting","Case Report"]

# Rename the index column
df_output.index.name = "PMID"

# Reorder columns
df_output = df_output[["Treatment","Diagnosis","Prevention","Mechanism","Transmission","Epidemic Forecasting","Case Report"]]

# Save as CSV
df_output.to_csv('TF-IDF__20K-features__1-2-3-grams__stopwords-EN__Keywords+Title+Abstract__Train+Dev.index.csv', index=True)
