#!/usr/bin/env python
# coding: utf-8

# In[103]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import stem
stemmer = stem.PorterStemmer()
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
import string
punct = list(string.punctuation)
from collections import Counter
import requests
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().system('pip install PRAW')
import numpy as np
import praw
import datetime
import nltk
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install textblob')



# In[ ]:





# In[104]:


reddit = praw.Reddit(user_agent='VAD',
                     client_id='hy1MkGf12dvyggYsCk26EQ', client_secret="KBQpRTL5FZ_Wo_6XUu0Q_OrCDsuh6g",
                     username='No_Technician_9043 ', password='nlpsummative1')

subreddit = reddit.subreddit('')
hot_posts = subreddit.hot(limit=500)

data = []
for i, post in enumerate(hot_posts):
    title = post.title
    score = post.score
    comments = []
    for comment in post.comments:
        comments.append(comment.body)
    data.append([title, score, comments])
    print(f"Processed post {i+1}/500")


# In[150]:


import pandas as pd
import praw
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

reddit = praw.Reddit(
    user_agent='VAD',
    client_id='hy1MkGf12dvyggYsCk26EQ',
    client_secret="KBQpRTL5FZ_Wo_6XUu0Q_OrCDsuh6g",
    username='No_Technician_9043',
    password='nlpsummative1')

def get_post_data(post):
    title = post.title
    score = post.score
    comments = []
    post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        if isinstance(comment, praw.models.MoreComments):
            continue
        comments.append(comment.body)
    return [title, score, comments]

subreddit = reddit.subreddit('all')
posts = subreddit.search('I OR me', limit=10000)

data = []
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(get_post_data, post) for post in posts]
    for i, future in enumerate(tqdm(futures, total=len(futures))):
        post_data = future.result()
        data.append(post_data)
        print(f"Processed post {i+1}/{len(futures)}")

df = pd.DataFrame(data, columns=['title', 'score', 'comments'])
df.to_csv('reddit_I_or_me.csv', index=False)


# In[147]:


df


# In[105]:


df_fashion.to_csv('fashion_data.csv', index=False)


# In[106]:


df_fashion


# In[127]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('wordnet')

df_fashion = pd.DataFrame(data, columns=['title', 'score', 'comments'])

df_fashion['title_tokenized'] = df_fashion['title'].apply(lambda x: word_tokenize(x))

df_fashion['title_lemmatized'] = df_fashion['title_tokenized'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x]))

df_fashion['comments_tokenized'] = df_fashion['comments'].apply(lambda x: [word_tokenize(comment) for comment in x])

df_fashion['comments_lemmatized'] = df_fashion['comments_tokenized'].apply(lambda x: [' '.join([lemmatizer.lemmatize(word) for word in comment]) for comment in x])

print(df_fashion.head())



# In[128]:


print(df_fashion['title_lemmatized'].head())


# In[122]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud


all_titles = " ".join(df_fashion['title_tokenized'].sum())
wordcloud = WordCloud(width = 800, height = 800, background_color ='pink', min_font_size = 10).generate(all_titles)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


all_lemmatized_titles = " ".join(df_fashion['title_lemmatized'])
wordcloud = WordCloud(width = 800, height = 800, background_color ='grey', min_font_size = 10).generate(all_lemmatized_titles)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[130]:


df_fashion['title_lemmatized']


# In[131]:


from collections import Counter


all_words = [word for title in df_fashion['title_lemmatized'] for word in title.split()]


top_words = Counter(all_words).most_common(10)

print(top_words)


# In[136]:


import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_punct_and_stopwords(text):
    no_punct = text.translate(str.maketrans('', '', string.punctuation))
    no_stopwords = ' '.join(word for word in no_punct.split() if word.lower() not in stop_words)
    return no_stopwords

df_fashion['title_lemmatized_no_punct_and_stopwords'] = df_fashion['title_lemmatized'].apply(remove_punct_and_stopwords)


# In[137]:


from collections import Counter

# Create a list of all the words in the titles
all_words = [word for title in df_fashion['title_lemmatized_no_punct_and_stopwords'] for word in title.split()]

# Count the frequency of each word and get the top 10
top_words = Counter(all_words).most_common(10)

print(top_words)


# In[138]:


from textblob import TextBlob


def get_sentiment_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


df_fashion['comment_sentiment'] = df_fashion['comments'].apply(lambda x: [get_sentiment_polarity(comment) for comment in x])


df_fashion['post_sentiment'] = df_fashion['comment_sentiment'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
import matplotlib.pyplot as plt


plt.hist(df_fashion['post_sentiment'], bins=20)
plt.xlabel('Sentiment Polarity')
plt.ylabel('Number of Posts')
plt.title('Distribution of Post Sentiment Polarity in r/fashion')
plt.show()


# In[139]:


print(df_fashion.columns)


# In[11]:


def display_top_titles(df, title_col, sentiment_col):
    df_sorted = df.sort_values(sentiment_col, ascending=False)
    print("Top 10 most positive titles:")
    for title in df_sorted.head(10)[title_col]:
        print(title)
    print("\nTop 10 most negative titles:")
    for title in df_sorted.tail(10)[title_col]:
        print(title)

display_top_titles(df_fashion, 'title', 'post_sentiment')


# In[140]:


df_fashion.to_csv('df_fashion_post_coding.csv', index=False)

