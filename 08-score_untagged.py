import numpy as np, pandas as pd
import pickle
from sklearn.externals import joblib

from helpers import FeatSelector

#%% read
# untagged data
dat = pickle.load(open("data/working/tweets_untagged.p", "rb"))
# models
mod_a = joblib.load('data/working/mod_tfidf.joblib')
mod_b = joblib.load('data/working/mod_tfidf_featusers.joblib')

#%% choose X random rows y scorear
datr = dat.sample(50, random_state=1993)
# para model_a
X_texto = datr.drop(columns=['clase']).texto

df_preds_a = pd.DataFrame({
    'textos': X_texto
    ,'probs': [p[0] for p in mod_a.predict_proba(X_texto)]
    ,'preds': mod_a.predict(X_texto)
})
# para model_b
X = datr.drop(columns=['clase'])
X['abt_ln_user_followers'] = np.log(X['user_followers'] + 1)
X['abt_ln_user_friends'] = np.log(X['user_friends'] + 1)
X['abt_ln_user_listed'] = np.log(X['user_listed'] + 1)
X['abt_ln_user_statuses'] = np.log(X['user_statuses'] + 1)
X['tiempo_user'] = X['created'] - X['user_created']
X['tiempo_user'] = X['tiempo_user'].astype('timedelta64[D]').astype('int')
X['abt_ln_tiempo_user'] = np.log(X['tiempo_user'] + 1)
df_preds_b = pd.DataFrame({
    'textos': X.texto
    ,'probs': [p[0] for p in mod_b.predict_proba(X)]
    ,'preds': mod_b.predict(X)
})

#%% save as csv
df_preds_a.to_excel('output/pred_unknown_tfidf.xlsx', encoding='utf-16')
df_preds_b.to_excel('output/pred_unknown_tfidf_featusers.xlsx', encoding='utf-16')
