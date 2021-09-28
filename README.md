# BioCreative-VII-Track-5

**Author**:

[LABRAK Yanis](https://www.linkedin.com/in/yanis-labrak-8a7412145/)
Master 2 – Computer Science

**Affiliation**:

[Laboratoire Informatique d’Avignon (LIA)](https://lia.univ-avignon.fr/)
Natural Language Processing Department

## Compatibility issues between Flair 0.8 and 0.9 scripts

Refer to [this GitHub issue](https://github.com/flairNLP/flair/issues/2426) to solve the compatibility issues or go back to Flair 0.8.

## Descriptions

|submit_id        |label-based micro avg precision|label-based micro avg recall|label-based micro avg f1|label-based macro avg precision|label-based macro avg recall|label-based macro avg f1|instance-based precision|instance-based recall|instance-based f1|
|-----------------|-------------------------------|----------------------------|------------------------|-------------------------------|----------------------------|------------------------|------------------------|---------------------|-----------------|
|BC7_submission_39|0.5130                         |0.8598                      |0.6426                  |0.5240                         |0.7391                      |0.5614                  |0.5965                  |0.8597               |0.7043           |
|BC7_submission_40|0.8760                         |0.8659                      |0.8709                  |0.8498                         |0.8138                      |0.8231                  |0.8981                  |0.8942               |0.8961           |
|BC7_submission_62|0.8699                         |0.8966                      |0.8830                  |0.8298                         |0.8570                      |0.8366                  |0.8993                  |0.9198               |0.9094           |
|BC7_submission_61|0.8951                         |0.8280                      |0.8602                  |0.8814                         |0.7723                      |0.8174                  |0.8787                  |0.8610               |0.8698           |

### BC7 Submission 39

I used class specific keywords extracted from the training dataset (keywords + title + abstract) with a TF-IDF to enhance a HuggingFace PubMedBERT model (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) adapted to the task by changing the loss function to a BCE one for multi-label classification and running it during 28 epochs with a learning rate of 5e-5.

**Folder**: `model 2 - PubMed Train`

### BC7 Submission 40

I used a pretrained model called TARS based on the paper "Task-Aware Representation of Sentences for Generic Text Classification" available in the framework Flair to classify documents based only on their abstracts during 50 epochs with a learning rate of 0.02 and with only 85% of the training corpus.

**Folder**: `model 1 - 50 runs flair tars`

### BC7 Submission 42

I used class specific keywords extracted from the training dataset (keywords + title + abstract) with a TF-IDF to enhance a HuggingFace PubMedBERT model (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) adapted to the task by changing the loss function to a BCE one for multi-label classification and running it during 28 epochs on **Train and Dev** with a learning rate of 5e-5.

**Folder**: `model 5 - PubMed Train+Dev`

### BC7 Submission 61

I trained a 1-2-3 gram TF-IDF on both Train and Dev datasets to compute df vectors (dimension 20K) which will represents documents (keywords + title + abstract) in the multi-label SVM classifier.

**Folder**: `model 6 - TF-IDF 1-2-3 gram`

### BC7 Submission 62

I used class specific keywords extracted from the training dataset (keywords + title + abstract) with a TF-IDF to enhance a pretrained model called TARS based on the paper "Task-Aware Representation of Sentences for Generic Text Classification" available in the framework Flair adapted to the task for multi-label classification and running it during 10 epochs on Train only with a learning rate of 0.02.

**Folder**: `model 4 - flair all + ner`
