# Import the necessary packages:
import os
import pandas as p
import tweepy as tw
from datetime import datetime
from datetime import timedelta
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import re
import sys
import autocorrect
from autocorrect import Speller
from collections import Counter
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
consumer_key= os.environ.get("T_API_SENT")
consumer_secret= os.environ.get("T_CONSUMER_SECRET_SENT")
access_token= os.environ.get("T_ACCESS_TOKEN_SENT")
access_token_secret= os.environ.get("T_ACCESS_TOKEN_SECRET_SENT")
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth)

#Define a natural text cleaning function:
import warnings
warnings.filterwarnings("ignore")


lemmatizer = WordNetLemmatizer()
my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'

def remove_links(tweet):
    tweet = re.sub(r'http\S+', '', tweet) 
    tweet = re.sub(r'bit.ly/\S+', '', tweet) 
    tweet = tweet.strip('[link]') 
    return tweet
def remove_users(tweet):
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) 
    return tweet
def reduce_lengthening(word):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", word)
def clean_tweets (tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) 
    tweet = re.sub('\s+', ' ', tweet) 
    tweet = re.sub('([0-9]+)', '', tweet) 
    return tweet

#Collect the basic information related to candidates:
account_list = ["realDonaldTrump", "joebiden"]
## Registered Names:
name = [(auth_api.get_user(user_name_)).name for user_name_ in account_list] 
## User Names:
screen_name = [(auth_api.get_user(user_name_)).screen_name for user_name_ in account_list]
## User Bio:
description = [(auth_api.get_user(user_name_)).description for user_name_ in account_list]
## Total tweets by now:
statuses_count = [(auth_api.get_user(user_name_)).statuses_count for user_name_ in account_list]
## Number of Friends:
friends_count = [(auth_api.get_user(user_name_)).friends_count for user_name_ in account_list]
## Number of Followers:
followers_count = [(auth_api.get_user(user_name_)).followers_count for user_name_ in account_list]
info_user = p.DataFrame(list(zip(account_list,name,screen_name,statuses_count,friends_count,followers_count,description)),
                        columns=['label','Official Name','User Name', 'Tweets By Now', 'Friends', 'Follower', 'Status'])

#Set the Start Date & End Date Counter:
end_date = str((datetime.now()).date())
start_date = str((datetime.now()).date()-timedelta(1))

#Collect all the new tweets available for the candidate before the end_date:
data_temp=[]
for _names in account_list:
      alltweets = []
      new_tweets = auth_api.user_timeline(screen_name = _names,count=200,since_id = since_Id,
                                    include_rts = False,tweet_mode = 'extended')
      alltweets.extend(new_tweets)
      if len(alltweets)>0:
          newest = alltweets[0].id + 1
          while len(new_tweets) > 0:
              new_tweets = auth_api.user_timeline(screen_name = "realDonaldTrump",count=200,since_id=newest,
                                             include_rts = False,tweet_mode = 'extended')
              alltweets.extend(new_tweets)
              newest = alltweets[0].id + 1
          Id=[("id_"+str(t.id)) for t in alltweets]
          tmp_date=[t.created_at for t in alltweets]
          Created_at=[t.replace(tzinfo=None) for t in tmp_date]
          Full_text=[t.full_text for t in alltweets]
          Fav=[t.favorite_count  for t in alltweets]
          Re_tweet=[t.retweet_count  for t in alltweets]
          col_names=['str_id','created_at','full_text','favorite_count','retweet_count']
          tmp_df1_got3= p.DataFrame(list(zip(Id,Created_at,Full_text,Fav,Re_tweet)), columns=col_names)
          tmp_df1_got3['clean_text']=tmp_df1_got3.full_text.apply(clean_tweets)
      data_temp.append(tmp_df1_got3)

# 'info_user' will consist of basic info related to both the candidates
# 'data_temp' will consist of two dataframes having day level (that particular day's) tweets of both the candidates

# Import the necessary packages:
import GetOldTweets3 as got

#Use GOT to get historical tweets:
labels= ['donald trump','joe biden']
data_temp=[]
for label in labels:
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(label).setSince(start_date).setUntil(end_date).setMaxTweets(10000).setLang("en")
    All_tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    Id=[("id_"+str(t.id)) for t in All_tweets]
    tmp_date=[t.date for t in All_tweets]
    Created_at=[t.replace(tzinfo=None) for t in tmp_date]
    Full_text=[t.text for t in All_tweets]
    Fav=[t.favorites  for t in All_tweets]
    Re_tweet=[t.retweets  for t in All_tweets]
    col_names=['str_id','created_at','full_text','favorite_count','retweet_count']
    tmp_df1_got3= p.DataFrame(list(zip(Id,Created_at,Full_text,Fav,Re_tweet)), columns=col_names)
    tmp_df1_got3['clean_text']=tmp_df1_got3.full_text.apply(clean_tweets)
    time.sleep(15*60) 
    data_temp.append(tmp_df1_got3)
    
# 'data_temp' will consist of two dataframes having more than 20k tweets in which candidates were referred


# Import the necessary packages:
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

#Set the cursor element:
t_host = "host"
t_port = "xxxx" # default port for postgres server
t_dbname = "tweeter_election2020"
t_name_user = "karansinh_raj"
t_password = os.environ.get("POSTGRESQL_DB")
db_conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_name_user, password=t_password)
db_cursor = db_conn.cursor()

#Append the data into already existing PostgreSQL tables: (conceptualization of code - the input will vary as per the feeded object)
tables=["historical_doland","historical_joebiden","usersentiment_doland","usersentiment_joebiden"] #all the 4 datasets built in previous step
for table in tables:
  for row in table.iterrows():
    db_cursor.execute("INSERT INTO historical_doland VALUES (%s, %s, %s, %s,%s, %s)", row)
  db_conn.commit()