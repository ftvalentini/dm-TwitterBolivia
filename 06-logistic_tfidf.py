import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score

from helpers import clean_text, tokenize, get_stopwords

# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score

#%% read data
# tagged tweets dframe
dat = pd.read_pickle('data/working/tweets_tagged.p')
# get stopwords
sw = get_stopwords()

#%% get textos y clase de tuits
# delete textos duplicados
datf = dat.drop_duplicates('texto', keep='first').reset_index(drop=True)
X_texto = X.texto
y = datf.clase

#%% pasos de los modelos
# extraccion de features
tfidf = TfidfVectorizer(preprocessor=clean_text, tokenizer=tokenize
                        , min_df=0.01, ngram_range=(1,4), stop_words=sw, binary=True)
# clasificador logistico (ridge)
clf = LogisticRegression(random_state=1993, penalty='l2')

#%% Pipeline - only tfidf (se aplica sobre X_texto)
pipe_a = Pipeline([('vect', tfidf), ('clf', clf)])

#%% Performance CV
scores_a = cross_val_score(pipe_a, X_texto, y, cv=5, scoring='f1_micro')
print("f-score=",round(scores_a.mean(),4),"(sd =",round(scores_a.std(),4),")")

#%% Feature importance
# fit on all data
mod = pipe_a.fit(X_texto, y)
# get feature names
features = mod.named_steps['vect'].get_feature_names()
# get regression coefficientes
weights = mod.named_steps['clf'].coef_[0]
# me parece que esta al reves! (no se por que):
important = {
mod.named_steps['clf'].classes_[0]: pd.Series(weights,index=features).sort_values(ascending=False)[:10]
,mod.named_steps['clf'].classes_[1]: pd.Series(weights,index=features).sort_values(ascending=False)[-10:]
}
# plots (revisar)
# important['AE']
important['AE'].plot(kind="bar",figsize=(15,5),color="darkgreen")
plt.ylabel("Feature importance",size=20);plt.xticks(size = 20);plt.yticks(size = 20)
# important['PE'].sort_values()
important['PE'].plot(kind="bar",figsize=(15,5),color="darkgreen")
plt.ylabel("Feature importance",size=20);plt.xticks(size = 20);plt.yticks(size = 20)





# OTROS:
#
# pipe_a_fitted = pipe_a.fit(X_texto, y)
# preds = pipe_a_fitted.predict(X_texto)
# probs = pipe_a_fitted.predict_proba(X_texto)
# print(classification_report(y, preds, target_names=clf.classes_))
# # metrics
# pd.Series([
# f1_score(y_test, log_class_pred, pos_label=log_clf.classes_[0])
# ,accuracy_score(y_test, log_class_pred)
# ]
# ,index=["f1-score","accuracy"]
# )
# # roc_auc_score(y_test, nb_prob_pred)
# # # y_test tiene que ser 0-1
# confusion_matrix(y_test, log_class_pred)
#
# train-test
# X_train_text, X_test_text, y_train, y_test = train_test_split(X,y,stratify=y, test_size=0.20, random_state=2011)
