from __future__ import absolute_import, print_function

# https://ljvmiranda921.github.io/notebook/2017/02/24/twitter-streaming-using-python/

# Import modules
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import sys
import json
# import csv
# import io

# Your credentials go here
consumer_key = "XcEdDI9jJnQP58eALgWjMeVr6"
consumer_secret = "lPamYjWmAijbIr2TOtg1lXuZFAH5oSPsurUNLiePLruti26DCa"
access_token = "2241662364-TfTTC57mrrA8yEDTQFDVaHacqNXGtVc3P92AH4b"
access_token_secret = "8kuVOhsWS6Ush1HXjmNHr6HSFjMxThphefbi3iBEzmbM0"

# fstatus = open('prueba_status.json', 'a')
# fdata = open('prueba_data.json', 'a')

output = open('stream_output.txt', 'a')
track_list = ['argentina','eleccionesargentina','elpaiselige','argentinadecide']


class Listener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    """

    def __init__(self, output_file=sys.stdout):
        super(Listener,self).__init__()
        self.output_file = output_file

    def on_status(self, status):
        json_status = json.dumps(status._json)
        print(str(json_status).encode('utf-8'), file=self.output_file)
        print(status.text.encode('utf-8'))

        # # Check if Retweet
        # if hasattr(status, "retweeted_status"):
        #     is_retweet = 1
        #     try:
        #         text = status.retweeted_status.extended_tweet["full_text"]
        #     except AttributeError:
        #         text = status.retweeted_status.text
        # else:
        #     is_retweet = 0
        #     try:
        #         text = status.extended_tweet["full_text"]
        #     except AttributeError:
        #         text = status.text

    # def on_data(self, data):
    #     json_data = json.loads(data)
    #     fdata.write(str(json_data))

    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return True # Don't kill the stream

    def on_timeout(self):
        print('Timeout...')
        return True # Don't kill the stream


if __name__ == '__main__':
    l = Listener(output_file=output)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l, tweet_mode = "extended")
    stream.filter(track=track_list, is_async=True)

# locations=[-55.251018,73.466332,-22.206848,52.688991]
