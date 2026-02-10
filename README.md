# Problem-4: Sports vs Politics Text Classification

## Overview
This repository contains the implementation and experimental results for
Problem-4 of Assignment-1 in the Natural Language Understanding course.

The task involves binary text classification, where news articles are classified
into two categories: **Sports** and **Politics**. The primary goal of this work
is not only to achieve good classification performance, but also to analyze how
different feature representations and classical machine learning models behave
on structured news data.

## Dataset
The experiments were conducted using the **BBC News dataset**, which consists of
professionally written news articles from multiple categories. For this problem,
only the *Sports* and *Politics* categories were selected to form a binary
classification dataset.

Dataset statistics:
- Sports articles: 511  
- Politics articles: 417  
- Total documents: 928  

An **80% training and 20% testing split** was applied using **stratified sampling**
to preserve class proportions across splits.

It is important to note that the dataset exhibits strong lexical separation
between the two categories, which makes the classification task relatively
easier compared to real-world, noisy datasets.

## Feature Representation
The following feature extraction techniques were explored:
- **Bag of Words (BoW)**
- **TF-IDF**
- **TF-IDF with n-grams (unigrams and bigrams)**

Standard preprocessing steps such as lowercasing, stopword removal, and minimum
document frequency thresholds were applied to reduce noise and sparsity.

## Machine Learning Models
The following supervised learning models were evaluated:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

All models were trained and evaluated using the same experimental pipeline to
ensure fair comparison across feature representations.

## Experimental Setup
To assess the stability of results, experiments were conducted using **multiple
random seeds** for the train-test split. Performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Mean accuracy and standard deviation across different seeds were also reported
to capture variability in model performance.

## Results
Across most experimental settings, the models achieved **very high accuracy**,
with some configurations reaching near-perfect performance. This behavior is
primarily attributed to the clean and well-structured nature of the dataset, as
well as the strong vocabulary separation between Sports and Politics articles.

While different models achieved similar accuracy values, they differ in their
learning assumptions and sensitivity to feature representations. These aspects
are analyzed in detail in the accompanying report.

## Limitations
The dataset consists of professionally edited news articles and lacks ambiguous
or multi-topic documents. As a result, the classification task does not fully
reflect the complexity of real-world news classification problems.

Additionally, the study is limited to:
- English language text
- Binary classification
- Classical machine learning models

These limitations should be considered when interpreting the reported results.

## Note
The code and data included in this repository are provided for reference and
documentation purposes. The **primary deliverable** for this problem is the
PDF report submitted as part of the assignment, which contains the complete
experimental design, results, and analysis as required.
