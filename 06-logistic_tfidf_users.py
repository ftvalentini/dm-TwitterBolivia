import pandas as pd
import numpy as np
import re, string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_validate, StratifiedKFold

from helpers import clean_text, tokenize, get_stopwords

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score

#%% read data
# tagged tweets dframe
dat = pd.read_pickle('data/working/tweets_tagged.p')
# get stopwords
sw = get_stopwords()

#%% get textos y clase de tuits
# delete textos duplicados
datf = dat.drop_duplicates('texto', keep='first').reset_index(drop=True)
X = datf.drop(columns=['clase'])
y = datf.clase

#%% generacion de features de users
# ACA GENERAR FEATURES DE USERS en X, GUARDAR NOMBRES EN UNA LISTA
# QUE EMPIECEN CON "abt_" !!!!!!!!
user_features = ['abt_user_followers','abt_user_friends']

#%% pasos de los modelos
semilla = 1993
# extraccion de features
tfidf = TfidfVectorizer(preprocessor=clean_text, tokenizer=tokenize
                        , min_df=0.01, ngram_range=(1,4), stop_words=sw, binary=True)
# feature selector (para seleccionar user_features ya generados)
class FeatSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        return df[self.variables]
    def get_feature_names(self):
        return self.variables
# scaling de features (minmax no funciona con sparse data)
# scaler = StandardScaler(with_mean=False)
scaler = MaxAbsScaler()
# clasificador logistico (ridge)
clf = LogisticRegression(random_state=semilla, penalty='l2')

#%% Pipeline - tfidf + user_features (se aplica sobre X)
tfidf_features = Pipeline([('selector', FeatSelector(variables='texto'))
                          ,('tfidf', tfidf)])
pipe_b = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', tfidf_features)
        ,('others', FeatSelector(variables=user_features))
    ]))
    ,('scaler', scaler)
    ,('clf', clf)
])

#%% Performance CV
cv = StratifiedKFold(n_splits=5, random_state=semilla)
metrics = {'acc':'accuracy','prec':'precision_macro','rec':'recall_macro','f1':'f1_macro'}
scores_b = cross_validate(pipe_b, X, y, cv=cv, scoring=metrics, return_train_score=False)
# save mean and std to csv
pd.DataFrame(scores_b).agg(["mean","std"]).round(4).T.to_csv('output/cv_scores_tfidf_featusers.csv')

#%% Feature importance
# fit on all data
mod = pipe_b.fit(X, y)
# get feature names
vars_tfidf = dict(mod.named_steps['features'].transformer_list).get('tfidf').named_steps['tfidf'].get_feature_names()
vars_user = dict(mod.named_steps['features'].transformer_list).get('others').get_feature_names()
features = vars_tfidf + vars_user
# get regression coefficientes (NO HAY PVALUES!!!)
weights = pd.Series(mod.named_steps['clf'].coef_[0], index=features)

#%% export as csv para ggplot
weights.to_csv("data/working/weights_tfidf_featusers.csv")



# OTROS:
# ## importance segun magnitud
# # me parece que esta al reves! (no se por que):
# important = weights.reindex(weights.abs().sort_values(ascending=False).index)[:20]
# # plots (revisar)
# important['AE']
# important['AE'].plot(kind="bar",figsize=(15,5),color="darkgreen")
# plt.ylabel("Feature importance",size=20);plt.xticks(size = 20);plt.yticks(size = 20)
# important['PE'].sort_values()
# important['PE'].plot(kind="bar",figsize=(15,5),color="darkgreen")
# plt.ylabel("Feature importance",size=20);plt.xticks(size = 20);plt.yticks(size = 20)
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
