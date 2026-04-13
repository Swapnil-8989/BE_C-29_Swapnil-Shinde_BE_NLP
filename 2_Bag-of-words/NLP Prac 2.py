#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


# In[2]:


df = pd.read_csv('car data.csv')
print(df.head())


# In[3]:


df['text'] = df['Make'] + " " + df['Engine Fuel Type'] + " " + df['Driven_Wheels'] + " " 
df['text'] = df['text'].str.lower()
print(df['text'].head())


# In[4]:


df['text'] = df['text'].fillna('')


# In[5]:


df['text'] = df['text'].astype(str)


# In[6]:


df['text'] = df['text'].str.lower()


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(df['text'])


# In[8]:


count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(df['text'])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
print("Bag of Words (Count):")
print(bow_df.head())


# In[ ]:





# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

documents = ["I love AI", "AI is amazing"]

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(documents)

bow_df = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names_out())


# In[11]:


normalized_bow = bow_df.div(bow_df.sum(axis=1), axis=0)
print("Normalized Bow:")
print(normalized_bow.head())


# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


# In[2]:


df = pd.read_csv('car data.csv')
print(df.head())


# In[ ]:




