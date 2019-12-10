import pandas as pd
import numpy as np
import re, pickle
from unidecode import unidecode

#%% parametros
# tweets dataframe pickle
f_tweets = 'data/working/tweets_bolivia.p'
# text file hashstags proevo-antievo
f_htpro = "resources/hashtags_proevo.txt"
f_htanti = "resources/hashtags_antievo.txt"
# tokens indicadores de portales de noticias en user description
tokens_portales = ['diario', 'portal', 'noticias', 'radio', 'news', 'actualidad'
                    ,'jornal', 'revista', 'informacion', 'periodismo', 'plataforma'
                    , 'CNN', 'prensa', 'television', 'periodico', 'canal', 'BBC'
                    ,'gazeta', 'medio de', 'agencia']

#%% read data
# read tuits dframe
dat = pd.read_pickle(f_tweets)
# limpiar HT function
def clean_ht(ht):
    # saca #
    texto = re.sub(r'#', '', ht)
    # reemplza caracteres con acento por caracteres sin acento
    texto = unidecode(texto)
    # lowercase
    texto = texto.lower()
    return texto
# read clean hashtags (proevo-antievo)
h_proevo = [clean_ht(line.rstrip("\n")) for line in open(f_htpro)]
h_antievo = [clean_ht(line.rstrip("\n")) for line in open(f_htanti)]

#%% filtros
# FILTER: solo tuits detectados como español
datf = dat.loc[dat.lang_detected=='es']

# FILTER: solo tuits con 4 palabras o mas (considerando texto limpio usado para lang_detect)
datf = datf.loc[datf.texto_limpio.str.split().str.len() >= 4]

# FILTER: drop portales de noticias
user_desc_temp = datf.user_desc.apply(clean_ht)
flag_portal = user_desc_temp.str.contains('|'.join(tokens_portales), flags=re.IGNORECASE, regex=True)
datf['portal'] = flag_portal.tolist()
datf = datf.loc[datf.portal == False]
# agrega algunos no captados por tokens
datf = datf.loc[~datf.user_screenname.isin([
                'C5N','elmundoes','telefe','politicomx','diarioeldeber','sumariumcom'
                ,'ElPortal24','venezuelaaldia','NTN24','NOTIFALCON','GobCDMX','UNAM_MX'
                ,'TUDNMEX','larepublica_pe','Latina_pe'
                ])]

# checks (segun cantidad followers y cantidad tuits)
# datf.groupby(['user_screenname','user_desc']).id.count().sort_values(ascending=False).head(100)
# most_followers = datf.drop_duplicates('user_id', keep="first", inplace=False).sort_values("user_followers", ascending=False)
# most_followers.loc[:,['user_screenname','user_desc']].head(100)

# sort by fecha creacion
datf = datf.sort_values(by='created')

#%% tagging (ProEvo - AntiEvo - untagged)
# definir tuits proevo-antievo-mixto segun HT (usa count en lugar de flag)
textos_temp = datf.texto.apply(clean_ht)
count_proevo = textos_temp.str.count('|'.join(h_proevo), flags=re.IGNORECASE)
count_antievo = textos_temp.str.count('|'.join(h_antievo), flags=re.IGNORECASE)
datf['proevo'] = count_proevo.tolist()
datf['antievo'] = count_antievo.tolist()
datf['mixto'] = (datf.proevo>0) & (datf.antievo>0)
# la mayor parte de los mixtos son AE que tienen "evonoestassolo"
    # --> al final sacamos evonoestassolo de la lista de hts!!!
# datf.loc[datf.mixto].texto.tolist()[:10]
# datf.loc[datf.mixto].texto.tolist()[-10:]

# FILTER: drop users que usan tag mixtos
users_mixtos = datf.loc[datf.mixto, 'user_id'].tolist()
datf = datf.loc[~datf.user_id.isin(users_mixtos)]
# reset index
datf = datf.reset_index(drop=True)

# tag finales:
    # se cuenta cantidad de AE-PE por usuario
    # para cada user: si mas del 90% de sus tuits tageados son AE (PE)
        # entonces se clasifican todos sus tuits como AE (PE)
    # todo el resto son tuits untagged
dat_temp = datf.groupby(['user_id']).agg({'proevo':'sum','antievo':'sum'}).reset_index()
dat_temp['proevo_perc'] = np.where((dat_temp.proevo>0) | (dat_temp.antievo>0)
                            , dat_temp.proevo/(dat_temp.proevo+dat_temp.antievo)*100, np.nan)
users_proevo = dat_temp.loc[dat_temp.proevo_perc > 90,'user_id'].tolist()
users_antievo = dat_temp.loc[dat_temp.proevo_perc < 10,'user_id'].tolist()
datf['proevo_final'] = datf.user_id.isin(users_proevo)
datf['antievo_final'] = datf.user_id.isin(users_antievo)
# clase
datf['clase'] = np.where(
    datf['proevo_final'] & ~datf['antievo_final'],
    'PE',
    np.where(
        datf['antievo_final'] & ~datf['proevo_final'],
        'AE','-'
    ))
# distribucion de la clase
# pd.crosstab(datf.proevo_final, datf.antievo_final)


# # EJEMPLOS:
# dat_temp.loc[(dat_temp.proevo_perc>90) & (dat_temp.proevo_perc<100)]
# datf.loc[datf.user_id=="109714931",['texto','proevo','antievo','clase']]
# dat_temp.loc[dat_temp.user_id=="109714931"]
# datf.loc[datf.user_id=="1151351098004922368",['texto','proevo','antievo','clase']]
# datf.loc[datf.user_id=="355736841",['texto','proevo','antievo','clase']]
# dat_temp.loc[dat_temp.user_id=="355736841"]
# datf.loc[datf.user_id=="2330908837",['texto','proevo','antievo','clase']]
# dat_temp.loc[dat_temp.user_id=="2330908837"]

# drop useless columns
datf = datf.drop(columns = ['lang_detected','portal','mixto','antievo'
                ,'proevo','antievo_final','proevo_final'])

#%% save as pickle
# tagged
tagged = datf.loc[datf.clase.isin(['PE','AE'])]
pickle.dump(tagged, open("data/working/tweets_tagged.p", "wb"))
# untagged
untagged = datf.loc[~datf.clase.isin(['PE','AE'])]
pickle.dump(untagged, open("data/working/tweets_untagged.p", "wb"))
