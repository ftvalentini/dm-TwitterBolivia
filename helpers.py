import re, string
from unidecode import unidecode
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# clean a hashtag
def clean_ht(ht):
    # saca #
    texto = re.sub(r'#', '', ht)
    # reemplza caracteres con acento por caracteres sin acento
    texto = unidecode(texto)
    # lowercase
    texto = texto.lower()
    return texto

# get clean hashtags usados para clasificar
def get_hts(ruta_ae="resources/hashtags_antievo.txt", ruta_pe="resources/hashtags_proevo.txt"):
    h_proevo = [clean_ht(line.rstrip("\n")) for line in open(ruta_ae)]
    h_antievo = [clean_ht(line.rstrip("\n")) for line in open(ruta_pe)]
    hts = h_proevo + h_antievo
    return hts

# clean_text para tokenizar y ajustar modelo
def clean_text(texto):
    # htags usados para clasificar
    hts = get_hts()
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

# stopwords con y sin acentos
    # import nltk
    # nltk.download("stopwords")
def get_stopwords():
    sw1 = stopwords.words('spanish')
    sw2 = [unidecode(w) for w in sw1]
    sw = list(set(sw1 + sw2))
    sw.extend(["x","xq","vs","ahi","hoy","q","u"])
    return sw

# selector de features en pipeline
class FeatSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        return df[self.variables]
    def get_feature_names(self):
        return self.variables
