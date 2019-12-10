import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from helpers import clean_text, tokenize, get_stopwords

#%% parametros del vectorizer
min_frec = 0.01
max_ngram = 4
binario = True

#%% read data
# tagged tweets dframe
dat = pd.read_pickle('data/working/tweets_tagged.p')
# get stopwords
sw = get_stopwords()

#%% get textos de tuits
# delete textos duplicados
    # duplicados en cada clase:
# dat.loc[dat.clase == 'AE'].texto.duplicated().sum()
# dat.loc[dat.clase == 'PE'].texto.duplicated().sum()
datf = dat.drop_duplicates('texto', keep='first').reset_index(drop=True)
texto_ae = datf.loc[datf.clase == 'AE'].texto
texto_pe = datf.loc[datf.clase == 'PE'].texto
texto_full = datf.texto

#%% tokenizar cada tuit
# vectorizer binario (1 si esta el termino - 0 si no)
    # no son frecuencias de palabras -- en cada tuit vale lo mismo si esta la palabra 1 vez que si esta 10
# se define diccionario en base a todos los tuits
count_vect = CountVectorizer(preprocessor=clean_text, tokenizer=tokenize
            , min_df=min_frec, ngram_range=(1,max_ngram), stop_words=sw, binary=binario)
mat = count_vect.fit_transform(texto_full)
mat_ae = count_vect.transform(texto_ae)
mat_pe = count_vect.transform(texto_pe)

#%% scores de importancia de terminos en cada clase
terminos = np.array(count_vect.get_feature_names())
freqs = np.asarray(mat.sum(axis=0)).ravel()
freqs_ae = np.asarray(mat_ae.sum(axis=0)).ravel()
freqs_pe = np.asarray(mat_pe.sum(axis=0)).ravel()
df = pd.DataFrame({'termino':terminos, 'fabs':freqs, 'fabs_ae':freqs_ae, 'fabs_pe':freqs_pe})
df['frel_ae'] = (df.fabs_ae / len(texto_ae))
df['frel_pe'] = (df.fabs_pe / len(texto_pe))
df['score_ae'] = np.log(df.frel_ae / df.fabs)
df['score_pe'] = np.log(df.frel_pe / df.fabs)
df['dif_score'] = np.abs(df.score_ae - df.score_pe)
# get top 15 diferencias para cada clase (solo de terminos con freq mayor a X)
    # use X=0 (o sea dado por min_frec) y X=5000
min_fabs = 0
temp = df.loc[df.fabs>=min_fabs].sort_values('dif_score',ascending=False)
tokens_ae = temp.loc[temp.score_ae > temp.score_pe].head(10).termino.tolist()
tokens_pe = temp.loc[temp.score_pe > temp.score_ae].head(10).termino.tolist()
tokens = tokens_ae + tokens_pe

#%% frecuencias relativas por hora de los tokens detectados
# agrega tokens al dataframe
i_terminos = [tokens.index(ter) if ter in tokens else None for ter in terminos]
temp2 = pd.DataFrame(mat.tocsr()[:,[i for i in i_terminos if i is not None]].toarray()
                    , columns=tokens)
tot = pd.concat([datf[['created','clase']], temp2], axis=1)
# tokens por intervalos de X horas y por clase
sums_time = tot.groupby([pd.Grouper(key="created", freq='24h', base=0, label='left'),'clase']).sum()
tot_time = tot.groupby([pd.Grouper(key="created", freq='24h', base=0, label='left'),'clase']).count()
# frecuencias relativas
rel_sums_time = sums_time / tot_time * 100

#%% export to csv para ggplot
sums_time.to_csv("data/working/time_tokens_abs_"+str(min_fabs)+".csv")
rel_sums_time.to_csv("data/working/time_tokens_rel_"+str(min_fabs)+".csv")
tot_time.to_csv("data/working/time_tokens_all_"+str(min_fabs)+".csv")



# PARA REVISAR TUITS:
# ts = ['dictador']
# aa = pd.DataFrame(mat[:,np.isin(terminos,ts)].toarray(), columns=ts)
# i_token0 = aa.loc[aa[ts[0]] == 1].index.tolist()
# dae = datf.loc[(datf.index.isin(i_token0)) & (datf.clase=="AE")]
# dpe = datf.loc[(datf.index.isin(i_token0)) & (datf.clase=="PE")]
# dpe[['texto','user_screenname']]
# dae[['texto','user_screenname']]
# datf.loc[datf.user_screenname == 'Fullchavisto'].texto.tolist()
# datf.loc[(datf.user_screenname == 'coloresperanz2') & datf.texto.str.contains("#")].texto.tolist()
# datf.loc[datf.user_screenname == 'CarmenEGonzale2'].texto.tolist()
# datf.loc[(datf.user_screenname == 'CarmenEGonzale2') & datf.texto.str.contains("#")].texto.tolist()
