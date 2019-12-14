# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re, string, pickle

# import nltk
# nltk.download('stopwords')
from unidecode import unidecode
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# read tagged data
dat = pd.read_pickle('tweets_bolivia_tagged.p')
# train-test
X = dat.texto
y = dat.clase
X_train_text, X_test_text, y_train, y_test = train_test_split(X,y,stratify=y, test_size=0.20, random_state=2011)

# stopwords con y sin acentos
    # import nltk
    # nltk.download("stopwords")
sw1 = stopwords.words('spanish')
sw2 = [unidecode(w) for w in sw1]
sw = list(set(sw1 + sw2))
sw.extend(["x","xq","vs","ahi","hoy","q","u","pq"])
# hashtags usados para clasificar (se eliminan)
def clean_ht(ht):
    # saca #
    texto = re.sub(r'#', '', ht)
    # reemplza caracteres con acento por caracteres sin acento
    texto = unidecode(texto)
    # lowercase
    texto = texto.lower()
    return texto
h_proevo = [clean_ht(line.rstrip("\n")) for line in open("hashtags_proevo.txt")]
h_antievo = [clean_ht(line.rstrip("\n")) for line in open("hashtags_antievo.txt")]
hts = h_proevo + h_antievo

# clean_text para ajustar modelo
def clean_text(texto):
    # saca links
    texto = re.sub(r'https?:\/\/.*[\r\n]*', '', texto)
    # saca hashtags
    texto = re.sub(r'#\S+', '', texto)
    # saca #
    texto = re.sub(r'#', ' ', texto)
    # saca mentions
    texto = re.sub(r'@\S+', '', texto)
    # saca punctuation
    texto = "".join([char for char in texto if char not in string.punctuation])
    # texto = re.sub(r'[^\w\s]',' ',texto)
    # saca guiones y algunos caracteres demas
    texto = re.sub(r'[-+_+¡+¿+]',' ',texto)
    # saca numeros
    texto = re.sub(r'\d+',' ',texto)
    # reemplza caracteres con acento por caracteres sin acento
    match_replace = [('á|Á','a'),('é|É','e'),('í|Í','i'),('ó|Ó','o'),('ú|Ú','u')]
    for i in range(len(match_replace)):
        texto = re.sub(match_replace[i][0], match_replace[i][1], texto)
    # saca hashtags usados para clasificar
    texto = re.sub('|'.join(hts), '', texto, flags=re.IGNORECASE)
    # saca whitespace duplicado
    texto = re.sub(' +', ' ', texto)
    return texto.strip()

# tokenizer
def tokenize(texto):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(texto)
    return tokens

# trainset con collocations
trainset = [tokenize(clean_text(i)) for i in X]
    # trainset = [tokenize(clean_text(i)) for i in X_train_text]
collocations = Phrases(sentences=trainset, min_count=100,threshold=0.5,scoring='npmi') # threshold: minimo score aceptado
to_collocations = Phraser(collocations)
trainset_ngrams = to_collocations[trainset]

# Word2Vec
w2v_dim = 25
w2v_model = Word2Vec(trainset_ngrams, workers=4, size=w2v_dim, min_count=10, window=7, sample=1e-3, negative=5, sg=1)
w2v_model.save("word2vec_"+str(w2v_dim)+"dim")  # save model

# w2v_model.most_similar(positive=["oea"], negative=[], topn=15)

# mean WE of each tuit
def get_mean_we(word2vec_model, words):
    words = [word for word in words if word in word2vec_model.wv.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []
mean_wes = []
for doc in trainset_ngrams:
    mean_wes.append(get_mean_we(w2v_model, doc))

# dataframe para matchear WEs con clase y tuit ID (y saca tuits sin WE)
datos = pd.DataFrame({'lens': [len(i) for i in mean_wes], 'clase': y, 'we': mean_wes
,'id':y.index.tolist()})
datosf_pe = datos.loc[datos.lens==w2v_dim,:].loc[datos.clase.isin(["PE"]),:].reset_index()
datosf_ae = datos.loc[datos.lens==w2v_dim,:].loc[datos.clase.isin(["AE"]),:].reset_index()
wes_pe = np.vstack(datosf_pe.we)
wes_ae = np.vstack(datosf_ae.we)

# similitud coseno entre tuits de cada clase
# tomamos muestra porque crashea con todo el dset
# sim_pe = cosine_similarity(sparse.csr_matrix(wes_pe))
# sim_ae = cosine_similarity(sparse.csr_matrix(wes_ae))
idx_pe = np.random.randint(wes_pe.shape[0], size=20000)
idx_ae = np.random.randint(wes_ae.shape[0], size=20000)
sim_pe = cosine_similarity(sparse.csr_matrix(wes_pe)[idx_pe,:])
sim_ae = cosine_similarity(sparse.csr_matrix(wes_ae)[idx_ae,:])
# save matrices de similitud
pickle.dump(sim_pe, open("word_similarities_pe.p", "wb"))
pickle.dump(sim_ae, open("word_similarities_ae.p", "wb"))

# tuits que minimizan distancia con el resto (medoides)
n_top = 5
ids_max_pe = datosf_pe.loc[sim_pe.mean(axis=0).argsort()[::-1][:n_top],:].id
ids_max_ae = datosf_ae.loc[sim_ae.mean(axis=0).argsort()[::-1][:n_top],:].id
ids_min_pe = datosf_pe.loc[sim_pe.mean(axis=0).argsort()[:n_top],:].id
ids_min_ae = datosf_ae.loc[sim_ae.mean(axis=0).argsort()[:n_top],:].id



dat.loc[ids_max_pe,'texto'].tolist()
dat.loc[ids_min_pe,'texto'].tolist()
dat.loc[ids_max_ae,'texto'].tolist()



# from gensim.similarities import Similarity
# from gensim.corpora import Dictionary
# dictionary = word2vec_model.wv.vocab.keys()
# sims = Similarity('temp/', w2v_model.wv, num_features=len(w2v_model.wv.vocab))
