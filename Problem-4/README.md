# Problem-4: Sports vs Politics Text Classification

## Overview
This repository documents the work carried out for Problem-4 of Assignment-1
in the Natural Language Understanding course. The task focuses on binary text
classification, where news articles are categorized as either Sports or Politics.

The objective of this problem is not only to obtain high classification accuracy,
but also to understand how feature representations and classical machine learning
models behave on structured news text.

## Dataset
The experiments were conducted using the BBC News dataset, which consists of
professionally written news articles across multiple categories. For this task,
only the Sports and Politics categories were selected to construct a binary
classification problem.

Dataset statistics:
- Sports articles: 511  
- Politics articles: 417  
- Total documents used: 928  

An 80% training and 20% testing split was applied using stratified sampling to
preserve class proportions.

The dataset exhibits strong lexical separation between the two categories,
which simplifies classification but may lead to optimistic performance estimates.

## Feature Representation
The following feature extraction techniques were explored:
- Bag of Words (BoW)
- TF-IDF
- TF-IDF with n-grams (unigrams and bigrams)

Standard preprocessing steps such as lowercasing, stopword removal, and minimum
document frequency filtering were applied to reduce noise and sparsity.

## Machine Learning Models
The following supervised learning models were evaluated:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

To ensure a fair comparison, all models were trained and evaluated using the same
TF-IDF feature representation and train-test split.

## Results
All three models achieved perfect classification accuracy on the test set. This
result is primarily attributed to the clean and well-structured nature of the
dataset, where Sports and Politics articles use largely distinct vocabularies.

Although the numerical performance is identical, the models differ in their
assumptions and learning mechanisms. These differences are discussed in detail
in the accompanying report.

## Limitations
The dataset contains professionally edited articles and does not include
ambiguous or mixed-topic documents. As a result, the classification task is
simpler than many real-world scenarios. The study is also limited to English
news text and a binary classification setting.

## Note
The code and data included in this repository are provided for reference and
documentation purposes only. The primary deliverable for this problem is the
PDF report, which contains the complete experimental setup, analysis, and
discussion as required by the assignment.
