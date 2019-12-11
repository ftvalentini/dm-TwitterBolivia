import pandas as pd
import numpy as np
import re, string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score

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
X['ln_user_followers'] = np.log(X['user_followers'] + 1)
X['ln_user_friends'] = np.log(X['user_friends'] + 1)
X['ln_user_listed'] = np.log(X['user_listed'] + 1)
X['ln_user_statuses'] = np.log(X['user_statuses'] + 1)
X['tiempo_user'] = X['created'] - X['user_created']
X['tiempo_user'] = X['tiempo_user'].astype('timedelta64[D]')
X['ln_tiempo_user'] = np.log(X['tiempo_user'] + 1)
X['words_tweet'] = X['texto_limpio'].str.split().apply(len)

user_features = ['ln_user_followers',
                 'ln_user_friends',
                 'ln_user_listed',
                 'ln_user_statuses',
                 'ln_tiempo_user',
                 'user_verified',
                 'words_tweet']

#%% pasos de los modelos
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
# clasificador logistico (ridge)
clf = LogisticRegression(random_state=1993, penalty='l2')

#%% Pipeline - tfidf + user_features (se aplica sobre X)
tfidf_features = Pipeline([('selector', FeatSelector(variables='texto'))
                          ,('tfidf', tfidf)])
pipe_b = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', tfidf_features)
        ,('others', FeatSelector(variables= user_features)) # aca va lista de user_features
    ])),
    ('clf', clf)
])

#%% Performance CV
scores_b = cross_val_score(pipe_b, X, y, cv=5, scoring='f1_micro')
print("f-score=",round(scores_b.mean(),4),"(sd =",round(scores_b.std(),4),")")

#%% Feature importance
# fit on all data
mod = pipe_b.fit(X, y)
# get feature names
vars_tfidf = dict(mod.named_steps['features'].transformer_list).get('tfidf').named_steps['tfidf'].get_feature_names()
vars_user = dict(mod.named_steps['features'].transformer_list).get('others').get_feature_names()
features = vars_tfidf + vars_user
# get regression coefficientes (OJO QUE ESTAN EN ESCALA DISTINTA Y NO HAY PVALUES)
weights = mod.named_steps['clf'].coef_[0]
# me parece que esta al reves! (no se por que):
important = {
mod.named_steps['clf'].classes_[0]: pd.Series(weights,index=features).sort_values(ascending=False)[:10]
,mod.named_steps['clf'].classes_[1]: pd.Series(weights,index=features).sort_values(ascending=False)[-10:]
}
# plots (revisar)
important['AE']
important['AE'].plot(kind="bar",figsize=(15,5),color="darkgreen")
plt.ylabel("Feature importance",size=20);plt.xticks(size = 20);plt.yticks(size = 20)
important['PE'].sort_values()
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
