import pandas as pd
import numpy as np
import re, string

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from helpers import clean_text, tokenize, get_stopwords

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# benchmark: NB con tfidf
# train
nb_clf = MultinomialNB(alpha=1).fit(X_train, y_train)
# predict
nb_class_pred = nb_clf.predict(X_test)
nb_prob_pred = nb_clf.predict_proba(X_test)[:,0]
# metrics
print(classification_report(y_test, nb_class_pred, target_names=nb_clf.classes_))
pd.Series([
    f1_score(y_test, nb_class_pred, pos_label=nb_clf.classes_[0])
    ,accuracy_score(y_test, nb_class_pred)
    ]
    ,index=["f1-score","accuracy"]
)
# roc_auc_score(y_test, nb_prob_pred)
# # y_test tiene que ser 0-1
confusion_matrix(y_test, nb_class_pred)

# NB con CV
nb_scores = cross_val_score(nb_clf, X_train, y_train, cv=10, scoring='f1_macro')
print("f-score=",round(nb_scores.mean(),4),"(sd =",round(nb_scores.std(),4),")")
