import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report



# Load and prepare dataset

'''
This script aims at understanding how traditional machine learning models fare at 
solving a simple binary text classification problem where we try to classify Sports news or Politics news. 
Instead of concentrating solely on achieving high accuracy, 
we are looking at how different feature representations fare, 
how biases from our data set might work, or how robust our tests are. 
The tests are designed to be simple.
A study on the impact of various text characteristics on classification is conducted by 
employing three modes of representation: Bag of Words, TF-IDF, TF-IDF with n-grams. 
A number of traditional supervised classifiers—Multinomial Naive Bayes, Logistic Regression, 
Linear SVM—are examined with metrics used to measure classification performance: accuracy, precision, recall, F1-score; the experiment is performed for a number of 
random splits to avoid making decisions based on a potentially fluked result.
'''

df = pd.read_csv("bbc_news.csv")

# Keep only required classes
df = df[df["category"].isin(["sport", "politics"])]

texts = df["text"].values
labels = df["category"].values

# Sanity check: class distribution
print("Class distribution:")
print(df["category"].value_counts())



# Storage for results across seeds


results = []



# Training and evaluation function


def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name, seed):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\nModel:", model_name)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, preds))

    results.append((seed, model_name, acc))



# Experiments across multiple random seeds

'''

To avoid unwittingly hanging our findings on the specific choice of a random seed that 
just happens to work well, we repeat the experiments with a variety of different random seeds. 
This provides some assurance that the performance of the model 
does not vary wildly with a specific choice of data split. 
Repeating the evaluation in this way is particularly important in the face of 
a small data set and a significant degree of lexical separation, such as in our case.

'''


for seed in [1, 7, 21]:

    print("\n==============================")
    print("Random Seed:", seed)
    print("==============================")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels
    )

    # ---------------- Feature Extraction ----------------

    bow = CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5
    )

    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,
        max_df=0.9
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    tfidf_ngram = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2)
    )

    X_train_ngram = tfidf_ngram.fit_transform(X_train)
    X_test_ngram = tfidf_ngram.transform(X_test)

    # ---------------- Models ----------------

    # Naive Bayes
    train_and_evaluate(
        X_train_bow, X_test_bow, y_train, y_test,
        MultinomialNB(),
        "Naive Bayes + BoW",
        seed
    )

    train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        MultinomialNB(),
        "Naive Bayes + TF-IDF",
        seed
    )

    train_and_evaluate(
        X_train_ngram, X_test_ngram, y_train, y_test,
        MultinomialNB(),
        "Naive Bayes + N-gram",
        seed
    )

    # Logistic Regression
    train_and_evaluate(
        X_train_bow, X_test_bow, y_train, y_test,
        LogisticRegression(max_iter=1000),
        "Logistic Regression + BoW",
        seed
    )

    train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        LogisticRegression(max_iter=1000),
        "Logistic Regression + TF-IDF",
        seed
    )

    # Linear SVM
    train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        LinearSVC(),
        "Linear SVM + TF-IDF",
        seed
    )



# Final averaged results (robust evaluation)

print("\n========== FINAL AVERAGED RESULTS ==========")

models = sorted(set([r[1] for r in results]))

for model in models:
    accs = [r[2] for r in results if r[1] == model]
    print(
        model,
        "-> Mean Accuracy:", round(np.mean(accs), 4),
        "| Std Dev:", round(np.std(accs), 4)
    )


'''

To avoid unwittingly hanging our findings on the specific choice of a random seed that just happens to work well, 
we repeat the experiments with a variety of different random seeds. 
This provides some assurance that the performance of the model 
does not vary wildly with a specific choice of data split. 
Repeating the evaluation in this way is particularly important in the face of a 
small data set and a significant degree of lexical separation, such as in our case.


'''

