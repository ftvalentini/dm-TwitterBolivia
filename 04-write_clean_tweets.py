# import json
import pickle
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


# read clean datos
tagged = pd.read_pickle('data/working/tweets_tagged.p')
untagged = pd.read_pickle('data/working/tweets_untagged.p')

all_tweets = tagged.append(untagged)

all_tweets.to_csv('data/working/all_tweets.csv')

# Tiempo desde la creación de la cuenta hasta el tweet en Dias.

#all_tweets['tiempo_user'] = all_tweets['created'] - all_tweets['user_created']
#all_tweets['tiempo_user'] = all_tweets['tiempo_user'].astype('timedelta64[D]')

# Lista con descriptivas
#user_creation_stats = []
#for clases in all_tweets.clase.unique():
#    print()
#    print(clases)
#    print(all_tweets[all_tweets.clase == clases][["tiempo_user"]].describe().reset_index())
#    user_creation_stats.append(all_tweets[all_tweets.clase == clases][["tiempo_user"]].describe().reset_index())
#
# user_creation_stats[0]

# Plot

#for clases in all_tweets.clase.unique():
#    print()
#    print(clases)
#    fig,axes = plt.subplots()
#    axes.hist(all_tweets[all_tweets.clase == clases]['tiempo_user'],bins=50)
#    plt.show()


