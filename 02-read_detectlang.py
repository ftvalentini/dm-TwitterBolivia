from __future__ import (absolute_import, division,print_function, unicode_literals)
import json
import sys
import re
import pandas as pd
import pickle
from langdetect import DetectorFactory, detect_langs, detect

# datos input
tweets_files = ['data/raw/tweets_bolivia.txt', 'data/raw/tweets_bolivia_evo.txt', 'data/raw/tweets_bolivia_tokens.txt']

# read as string
data = ''
for file in tweets_files:
    with open(file, 'rb') as f:
        # data.extend(f.read().decode('utf-16le').encode().decode('utf-8-sig'))
        data = data + f.read().decode('utf-16le').encode().decode('utf-8-sig')
# parse as list of tuits
str_list = re.findall(r'{\r\n(?s).*?\r\n}', data)
# keep solo los tweets que tienen los keys correctos (algunos estaban mal -- uno solo je)
str_ok = [i for i in str_list if len(re.findall(r'\r\n\s+"\w+":', i))==15]
# parse as list of jsons
tweets = [json.loads(i) for i in str_ok]
# parse as dataframe
df = pd.DataFrame(tweets)

# transforma a datetime y resta 3 horas
df[['created','user_created']] = df[['created','user_created']].apply(pd.to_datetime)
df.created = df.created - pd.Timedelta(hours=3)
df.user_created = df.user_created - pd.Timedelta(hours=3)

# FILTRO: drop duplicates
dat = df.drop_duplicates()

# columna con idioma estimado del tuit
# (idealmente ver prob de cada idioma con detect_langs pero tarda siglos)
DetectorFactory.seed = 2011
# funcion de limpieza para interpretar idioma
def limpia(str):
    # regex de emoji (no funciona perfectamente bien)
    RE_EMOJI = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    # remove mentions, links, hashtags, emojis, digitos
    # lowercase
    out = re.sub(r'@\S+', '', str)
    out = re.sub(r'https?:\/\/.*[\r\n]*', '', out)
    out = re.sub(r'#\S+', '', out)
    out = RE_EMOJI.sub(r'', out)
    out = re.sub(r'\d+', '', out)
    out = re.sub(r'\n', ' ', out)
    out = out.strip().lower()
    return out
# columna de texto limpio
dat.loc[:,'texto_limpio'] = [limpia(t) for t in dat.texto]
# keep si solo tiene por lo menos una letra
datf = dat.loc[dat.texto_limpio.str.contains(r'\w+')].copy()

# SAVE as pickle
# pickle.dump(datf, open("tweets_bolivia_raw.p", "wb"))

# columna de idioma estimado
langs = []
i = 0
for t in datf.texto_limpio:
    i += 1
    # print("detected lang of {0} tweets of {1}".format(i, datf.shape[0]))
    try:
        langs.append(detect(t))
    except:
        langs.append(str(sys.exc_info()[1]))
datf['lang_detected'] = langs

# write langs to text file
# with open('langs.txt','w') as f:
#     for s in langs:
#         f.write("%s\n" % s)

# temp = [str(i) for i in langs]
# datf['lang_detected'] = temp

# SAVE as pickle
pickle.dump(datf, open("data/working/tweets_bolivia.p", "wb"))
# pickle.dump(langs, open("tweets_langs.p", "wb"))


# # FILTRO: saca retweets
# dat = dat.loc[dat.is_retweet==0]
#
# # FILTRO: usuarios con mas de 10 followers
# dat = dat.loc[dat.user_followers>10]
#
# # FILTRO: keep solo tokens que importan en el texto
# tokens = "eleccionesargentina|argentinavota|argentinadecide|macripresidente|albertopresidente|votoportodos"
# dat = dat.loc[df.texto.str.contains(tokens, case=False)]

# # sort by fecha creacion
# dat = dat.sort_values(by='created')
#
# # reset index
# dat = dat.reset_index(drop=True)

# # SAVE as pickle
# pickle.dump(dat, open("datos_clean.p", "wb"))
