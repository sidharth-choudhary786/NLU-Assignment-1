import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Problem 4: Sports vs Politics Text Classification

# This script carries out binary text classification on news articles using classical machine learning approaches.
# The aim is to classify Sports and Politics news articles using various feature spaces and classifiers.

# The code consists of a typical NLP workflow: data loading, processing, feature extraction, model training, and evaluation.
# The libraries used are restricted to pandas and scikit-learn, which are standard libraries for text classification.

# load csv
df = pd.read_csv("bbc_news.csv")

# filter classes
df = df[df["category"].isin(["sport", "politics"])]

texts = df["text"].values
labels = df["category"].values


# The data is read from a CSV file and filtered to retain only two categories: sport and politics.
# This reduces the problem to a binary classification problem.
# The data is split into a training set and a testing set.
# Stratified sampling is employed to ensure that the proportion of classes in both sets is the same.
# Text data is represented as numerical data using various vectorization methods like Bag of Words and TF-IDF.
# Stop words are removed and words with low document frequency are ignored using a minimum document frequency threshold.
# Three different classifiers are trained and tested:
# 1. Multinomial Naive Bayes
# 2. Logistic Regression
# 3. Linear Support Vector Machine

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Bag of Words
bow = CountVectorizer(
    lowercase=True,
    stop_words='english',
    min_df=2
)

X_train_bow = bow.fit_transform(X_train)
X_test_bow  = bow.transform(X_test)


tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)



tfidf_ngram = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    min_df=2,
    ngram_range=(1, 2)
)

X_train_ngram = tfidf_ngram.fit_transform(X_train)
X_test_ngram  = tfidf_ngram.transform(X_test)


# The results indicate that the accuracy achieved by all models is very high.
# This is primarily because the BBC News dataset is clean and well-separated, with sports and politics articles having their own vocabularies.
# However, such high accuracy may not be applicable in real-world datasets, which may have overlapping topics and noisy language.


nb = MultinomialNB()

nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))



lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))



svm = LinearSVC()

svm.fit(X_train_tfidf, y_train)

y_pred_svm = svm.predict(X_test_tfidf)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


# The fact that all models have achieved perfect accuracy on the dataset means that the dataset is very separable. Although this is a proof of
# the efficiency of the classifiers, it also indicates that the dataset does not capture the complexity of the real world.

