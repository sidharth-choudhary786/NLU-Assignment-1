import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# load csv
df = pd.read_csv("bbc_news.csv")

# filter classes
df = df[df["category"].isin(["sport", "politics"])]

texts = df["text"].values
labels = df["category"].values



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

