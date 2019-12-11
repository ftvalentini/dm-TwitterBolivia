#https://bhaskarvk.github.io/2015/01/how-to-use-twitters-search-rest-api-most-effectively./

# NOTA: este script se corrio 3 veces con 3 busquedas distintas (estan comentadas)

import tweepy as tw
import sys
import json
import os
import time

### PARAMETROS DE BUSQUEDA ###
searchQuery = '"evo morales" AND -filter:retweets'
# searchQuery = "bolivia AND -filter:retweets"  # this is what we're searching for
# searchQuery = "golpedeestadobolivia OR golpedeestadoenbolivia OR evopresidentelegitimo OR evonoestassolo OR evodictador OR bolivianohaygolpe  AND -filter:retweets"
### OUTPUT FILE ###
fName = 'data/raw/tweets_bolivia_evo.txt'
# fName = 'data/raw/tweets_bolivia_tokens.txt'
# fName = 'data/raw/tweets_bolivia.txt'

# credenciales FV (ver txt resources):
consumer_key = "XXX"
consumer_secret = "XXX"
access_token = "XXX"
access_token_secret = "XXX"

auth = tw.AppAuthHandler(consumer_key, consumer_secret) # usando esta autentificacion baja a mayor rate...
api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

maxTweets = 10000000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits

sinceId = False
max_id = False
# If results from a specific ID onwards are read, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
with open(fName, 'a', encoding='utf-16') as f:
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode = "extended")
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode = "extended",
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode = "extended",
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode = "extended",
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                if hasattr(tweet, "retweeted_status"):
                    is_retweet = 1
                    try:
                        texto = tweet.retweeted_status.full_text
                    except AttributeError:
                        texto = tweet.retweet_status.text
                else:
                    is_retweet = 0
                    try:
                        texto = tweet.full_text
                    except AttributeError:
                        texto = tweet.text
                dictio = {"id": tweet.id_str,
                          "texto": texto,
                          "created": tweet.created_at,
                          "user_id": tweet.user.id_str,
                          "user_screenname": tweet.user.screen_name,
                          "user_name": tweet.user.name,
                          "user_desc": tweet.user.description,
                          "user_followers": tweet.user.followers_count,
                          "user_friends": tweet.user.friends_count,
                          "user_listed": tweet.user.listed_count,
                          "user_created": tweet.user.created_at,
                          "user_verified": tweet.user.verified,
                          "user_statuses": tweet.user.statuses_count,
                          "user_lang": tweet.user.lang,
                          "is_retweet": is_retweet}
                json.dump(dictio, f, ensure_ascii=False, indent=4, default=str)
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tw.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            time.sleep(60*2)
            continue

print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
