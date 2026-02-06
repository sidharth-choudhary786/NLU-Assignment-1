# Problem-4: Sports vs Politics Text Classification

## Overview
This project addresses a binary text classification problem where news articles are classified into two categories: Sports and Politics. The objective is to compare different machine learning techniques and feature representations for document classification.

## Dataset
The BBC News dataset was used for this task. The dataset was obtained in CSV format and contains news articles from multiple categories. For this problem, only the sports and politics categories were selected to form a binary classification dataset.

- Sports articles: 511
- Politics articles: 417
- Total documents used: 928

An 80% training and 20% testing split was applied with stratified sampling.

## Feature Representation
The following feature extraction techniques were explored:
- Bag of Words (BoW)
- TF-IDF
- TF-IDF with n-grams (unigrams and bigrams)

Stopword removal and minimum document frequency thresholds were applied to reduce noise.

## Machine Learning Models
Three supervised learning models were trained and evaluated:
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

All models were trained using TF-IDF features for fair comparison.

## Results
All three models achieved very high accuracy on the test set. This is attributed to the clean and well-separated nature of the dataset, where sports and politics articles exhibit distinct vocabularies.

Despite identical accuracy scores, the models differ in their learning behavior and assumptions.

## Limitations
The dataset is relatively clean and does not include ambiguous or overlapping documents. Real-world news articles may contain mixed topics or informal language, which could reduce performance. Additionally, the study is limited to English news text.

## How to Run
1. Install required libraries:
   - pandas
   - scikit-learn
2. Run the classification script:
''' python sports_politics_classification.py'''
