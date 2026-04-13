#!/usr/bin/env python
# coding: utf-8

# In[3]:


text = "Rohit Sharma is greatest captain of all format of india."
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, TweetTokenizer
from nltk.tokenize import MWETokenizer

print("Original Text:\n", text)

print("\nWhitespace Tokenization:")
print(text.split())

print("\nPunctuation-based Tokenization:")
print(word_tokenize(text))

print("\nTreebank Tokenization:")
treebank = TreebankWordTokenizer()
print(treebank.tokenize(text))

print("\nTweet Tokenization:")
tweet = TweetTokenizer()
print(tweet.tokenize(text))

mwe = MWETokenizer([('NLP', 'practical')])
print("\nMWE Tokenization:")
print(mwe.tokenize(text.split()))




from nltk.stem import PorterStemmer, SnowballStemmer
words = word_tokenize(text)
porter = PorterStemmer()
porter_stems = [porter.stem(word) for word in words]
print("\nPorter Stemming:")
print(porter_stems)
snowball = SnowballStemmer("english")
snowball_stems = [snowball.stem(word) for word in words]
print("\nSnowball Stemming:")
print(snowball_stems)



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]
print("\nLemmatization:")
print(lemmatized_words)


# In[ ]:




