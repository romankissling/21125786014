#!/usr/bin/env python
# coding: utf-8

# In[147]:


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





# In[84]:


df_fashion = pd.read_csv('df_fashion_post_coding.csv')


# In[85]:


df_fashion


# In[86]:


print(df_fashion['title_lemmatized'].head())


# In[87]:


from collections import Counter


all_words = [word for title in df_fashion['title_lemmatized_no_punct_and_stopwords'] for word in title.split()]


top_words = Counter(all_words).most_common(10)

print(top_words)


# In[88]:


#SENTIMENT ANALYSES 


# In[89]:


print(df_fashion.columns)


# In[90]:


from textblob import TextBlob

def get_sentiment_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df_fashion['comment_sentiment'] = df_fashion['comments'].apply(lambda x: [get_sentiment_polarity(comment) for comment in x])
df_fashion['title_sentiment'] = df_fashion['title'].apply(lambda x: get_sentiment_polarity(x))


df_fashion['combined_sentiment'] = df_fashion.apply(lambda x: x['title_sentiment'] + sum(x['comment_sentiment']), axis=1)


df_fashion['post_sentiment'] = df_fashion['combined_sentiment'] / (len(df_fashion['title']) + sum(df_fashion['comments'].apply(lambda x: len(x))))



# In[ ]:





# In[91]:


def display_top_titles(df, title_col, sentiment_col):
    df_sorted = df.sort_values(sentiment_col, ascending=False)
    print("Top 10 most positive titles:")
    for title in df_sorted.head(10)[title_col]:
        print(title)
    print("\nTop 10 most negative titles:")
    for title in df_sorted.tail(10)[title_col]:
        print(title)

display_top_titles(df_fashion, 'title', 'post_sentiment')



# In[92]:


from wordcloud import WordCloud


top_titles = df_fashion.sort_values('title_sentiment', ascending=False)['title'].head(100)


title_string = ' '.join(top_titles)


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(title_string)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# In[93]:





# In[94]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')



# In[95]:


import spacy
import nltk
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
from nltk.corpus import stopwords




stop_words = set(stopwords.words('english'))




def extract_entities(text):
    doc = nlp(text)
    cities = [entity.text for entity in doc.ents if entity.label_ == 'GPE']
    dates = [entity.text for entity in doc.ents if entity.label_ == 'DATE']
    persons = [entity.text for entity in doc.ents if entity.label_ == 'PERSON']
    nouns = [entity.text for entity in doc.ents if entity.label_ == 'NORP']
    facilities = [entity.text for entity in doc.ents if entity.label_ == 'FAC']
    organizations = [entity.text for entity in doc.ents if entity.label_ == 'ORG']
    locations = [entity.text for entity in doc.ents if entity.label_ == 'LOC']
    products = [entity.text for entity in doc.ents if entity.label_ == 'PRODUCT']
    events = [entity.text for entity in doc.ents if entity.label_ == 'EVENT']
    works_of_art = [entity.text for entity in doc.ents if entity.label_ == 'WORK_OF_ART']
    languages = [entity.text for entity in doc.ents if entity.label_ == 'LANGUAGE']
    
    return list(set(cities)), list(set(dates)), list(set(persons)), list(set(nouns)), list(set(facilities)), list(set(organizations)), list(set(locations)), list(set(products)), list(set(events)), list(set(works_of_art)), list(set(languages))


df_fashion[['cities', 'dates', 'persons', 'nouns', 'facilities', 'organizations', 'locations', 'products', 'events', 'works_of_art', 'languages']] = df_fashion.apply(
    lambda row: pd.Series(extract_entities(row['title'] + ' ' + row['comments'])), axis=1)


print(df_fashion.head())


# In[98]:


from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter


def filter_entities(entities):
    filtered = [entity for entity in entities if entity not in STOP_WORDS and len(entity) > 2 and not entity.isdigit()]
    return filtered


all_entities = df_fashion['cities'].tolist() + df_fashion['dates'].tolist()+ df_fashion['persons'].tolist()+ df_fashion['nouns'].tolist()+ df_fashion['facilities'].tolist()+ df_fashion['organizations'].tolist()+ df_fashion['locations'].tolist()+ df_fashion['products'].tolist()+ df_fashion['events'].tolist()+ df_fashion['works_of_art'].tolist()+ df_fashion['languages'].tolist()

all_entities = [item for sublist in all_entities for item in sublist]

all_entities = filter_entities(all_entities)

entity_counts = Counter(all_entities)

top_entities = entity_counts.most_common(20)
x, y = zip(*top_entities)
plt.bar(x, y)
plt.xticks(rotation=90)
plt.show()


# In[99]:


cities_counter = Counter([item for sublist in df_fashion['cities'].tolist() for item in sublist])
top_cities = cities_counter.most_common(5)
print("Top 5 cities:")
for city, count in top_cities:
    print(city, count)

dates_counter = Counter([item for sublist in df_fashion['dates'].tolist() for item in sublist])
top_dates = dates_counter.most_common(5)
print("\nTop 5 dates:")
for date, count in top_dates:
    print(date, count)

persons_counter = Counter([item for sublist in df_fashion['persons'].tolist() for item in sublist])
top_persons = persons_counter.most_common(5)
print("\nTop 5 persons:")
for person, count in top_persons:
    print(person, count)

organizations_counter = Counter([item for sublist in df_fashion['organizations'].tolist() for item in sublist])
top_organizations = organizations_counter.most_common(5)
print("\nTop 5 organizations:")
for org, count in top_organizations:
    print(org, count)

languages_counter = Counter([item for sublist in df_fashion['languages'].tolist() for item in sublist])
top_languages = languages_counter.most_common(5)
print("\nTop 5 languages:")
for lang, count in top_languages:
    print(lang, count)


# In[100]:


get_ipython().system('pip install gensim')
import pandas as pd
import gensim
from gensim import corpora




docs = df_fashion['title_lemmatized_no_punct_and_stopwords'].tolist()


tokenized_docs = [doc.split() for doc in docs]


word_freq = corpora.Dictionary(tokenized_docs)


corpus = [word_freq.doc2bow(doc) for doc in tokenized_docs]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=word_freq,
                                            num_topics=10,
                                            random_state=42,
                                            passes=10)


topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)


# In[101]:


import pandas as pd
from collections import Counter

topic_counts = Counter(topic for document_topics in lda_model[corpus] for topic, _ in document_topics)
top_topics = topic_counts.most_common(5)

data = []
for topic_id, count in top_topics:
    topic_words = [word.split("*")[1].strip().replace('"', '') for word in lda_model.print_topic(topic_id).split(" + ")]
    data.append({"Topic": f"Topic #{topic_id}", "Words": topic_words, "Num Docs": count})

df = pd.DataFrame(data)
df.set_index("Topic", inplace=True)
print(df)



# In[102]:


coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=word_freq, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()


print('\nCoherence Score: ', coherence_lda)


# In[126]:


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


title_tfidf = tfidf.fit_transform(df_fashion['title'])


feature_names = list(tfidf.vocabulary_.keys())


for i in range(len(df_fashion)):
    print("Title:", df_fashion['title'][i])
    for j in range(len(feature_names)):
        print(feature_names[j], title_tfidf[i,j])
        




# In[127]:


top_n = 10


for i in range(len(df_fashion)):
   
    scores = title_tfidf[i,:].toarray()[0].tolist()

    feature_scores = [(feature_names[j], scores[j]) for j in range(len(feature_names))]
    feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    
    print("Title:", df_fashion['title'][i])
    for j in range(top_n):
        print(feature_scores[j][0], feature_scores[j][1])
    print('\n')

    


# In[130]:


#I didnt like how it turned out the first time so I decided to try another library, NLTK also didnt work for me but gensim seems to do the job for this
from gensim import corpora, models


texts = [[word for word in title.lower().split() if word not in stop_words]
         for title in df_fashion['title']]


dictionary = corpora.Dictionary(texts)


corpus = [dictionary.doc2bow(text) for text in texts]


tfidf = models.TfidfModel(corpus)


corpus_tfidf = tfidf[corpus]


for i in range(len(df_fashion)):
    print("Title:", df_fashion['title'][i])
    for j in range(len(corpus_tfidf[i])):
        print(dictionary[corpus_tfidf[i][j][0]], corpus_tfidf[i][j][1])



# In[146]:


avg_scores = []
for i in range(len(df_fashion)):
    title_tfidf_scores = [score for word_id, score in corpus_tfidf[i]]
    avg_score = sum(title_tfidf_scores) / len(title_tfidf_scores)
    avg_scores.append(avg_score)


for i in range(len(df_fashion)):
    print("Title:", df_fashion['title'][i])
    print("Average TF-IDF score:", avg_scores[i])
    print()


# In[ ]:




