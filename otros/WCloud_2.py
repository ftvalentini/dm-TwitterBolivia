# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:45:34 2019

@author: Luciano
"""

# import json
import re
import pickle
import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
from stop_words import get_stop_words

# from nltk.tokenize.casual import TweetTokenizer
# from nltk.corpus import stopwords
from unidecode import unidecode
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import matplotlib.pyplot as plt
def clean_text(texto):
    # saca links
    texto = re.sub(r'https?:\/\/.*[\r\n]*', '', texto)
    # saca #
    texto = re.sub(r'#', '', texto)
    # saca punctuation
    texto = re.sub(r'[^\w\s]','',texto)
    # saca guiones
    texto = re.sub(r'[-+_+]','',texto)
    # saca numeros
    texto = re.sub(r'\d+','',texto)
    # reemplza caracteres con acento por caracteres sin acento
    # (NOTA: esto saca los emojis -- hay que lograrlo de una manera que no los saque)
    texto = unidecode(texto)
    texto= texto.lower()
    return texto


# read clean datos
# dirWork='C:/mAESTRIA/RMT/Bolivia/data/working/'
# os.chdir(dirWork)
fileTweets= 'data/working/tweets_tagged.p'#'tweets_bolivia.p'
dat = pd.read_pickle(fileTweets)

dat.columns.values.tolist()
dat = dat.sort_values(by='created')

maxDate=dat['created'].max()
minDate=dat['created'].min()
stop_words = get_stop_words('es')
stop_words.extend(["x","xq","vs","ahi","hoy","q","u","evo","morales","bolivia","golpedeestadobolivia","golpedeestadoenbolivia","evopresidentelegitimo","evonoestassolo","evodictador","bolivianohaygolpe"])

textos = dat.texto.tolist()
limpios = []
for i in textos:
    limpios.append(clean_text(i))
dat["textLimpio"]=limpios
print(stop_words)
intervalo=24

ahora = dt.now()
nombre = str(ahora.strftime("%Y%m%d%H%M"))
print(nombre)
ahora.strftime("%d/%m/%Y %H:%M")
i=1
##list=[[False, True, False ], [True, False, False], [False, False, False], [False, False, True] ]
list=[[False, False, True] ]

dat['proevo'] = dat.clase == "PE"
dat['antievo'] = dat.clase == "AE"

CortePrDiaAcum= True;
for pars in list:
    proevo=pars[0]
    antievo=pars[1]
    todos=pars[2]
    dateLoop=minDate+ timedelta(hours=intervalo)
    print (proevo, antievo, todos)
    #proevo=False
    #antievo=True
    #todos=False
    maxInterval=dt(2019,11,5)
    mininterval=minDate
    while dateLoop < maxDate:
        cuales=''
        if (todos):
            cuales='Todos'
        elif ( proevo):
            cuales='ProEvo'
        elif (antievo):
            cuales='AntiEvo'
        elif(proevo==False & antievo ==False ):
            cuales='Sin Clase'
        print (dateLoop)
        print (dateLoop + timedelta(hours=intervalo))
        if(not CortePrDiaAcum):
            dateLoop=dateLoop+ timedelta(hours=intervalo)
            tit=cuales + ' entre ' +dateLoop.strftime("%d/%m/%Y %H:%M") + ' y ' + maxInterval.strftime("%d/%m/%Y %H:%M")
            maxInterval=dateLoop+ timedelta(hours=intervalo)
            datBtDate = dat[(dat['created'] >= dateLoop) & (dat['created'] < maxInterval) & ((dat['proevo'] == proevo) &(dat['antievo'] == antievo)| todos==True)]
        else:
            dateLoop=maxInterval

            maxInterval=maxInterval + timedelta(hours=intervalo)
            tit=cuales + ' entre ' +dateLoop.strftime("%d/%m/%Y %H:%M") + ' y ' + maxInterval.strftime("%d/%m/%Y %H:%M")
            datBtDate = dat[(dat['created'] >= dateLoop) & (dat['created'] < maxInterval) & ((dat['proevo'] == proevo) &(dat['antievo'] == antievo)| todos==True)]
        text=datBtDate.textLimpio
        wordcloud = WordCloud(
        width = 750,
            height = 500,
            background_color = 'white',
            stopwords = stop_words).generate(str(text))
        fig = plt.figure(
            #figsize = (10, 8.5),
            facecolor = 'k',
            edgecolor = 'k'
            )
        fig.suptitle(tit, fontsize=11)
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout(pad=1.5)
        plt.savefig(cuales+nombre +'-' + str(i) +'.png' )
        i= i+1

    print (maxDate  -minDate)
