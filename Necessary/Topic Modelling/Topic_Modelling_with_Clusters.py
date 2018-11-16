
from gensim import corpora, models, similarities
from itertools import chain

import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]


# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]


# In[9]:


# Create Dictionary.
id2word = corpora.Dictionary(texts)


# In[10]:


# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]


# In[11]:


num_topics = 4


# In[12]:


# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=num_topics,                                update_every=1, chunksize=10000, passes=10)


# In[21]:


print(lda)


# In[13]:


# Prints the topics.
for top in lda.print_topics():
  print(top)


# In[14]:


# We will iterate over the number of topics, get the top words in each cluster and add them to a dataframe.
def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 10);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);


# In[15]:


# Getting the LDA Topics
get_lda_topics(lda, num_topics)


# In[16]:


# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]


# In[17]:


#i=0
#extracted_sentences = []
#while i <=number_of_sentences_in_summary-1:
#    extracted_sentences.append(ranked[i][1])
#    i = i + 1
#extracted_summary = ''.join(extracted_sentences)
#print(extracted_summary)


# In[18]:


scores = list(chain(*[[score for topic_id,score in topic]                       for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print (threshold)

cluster1 = [j for i,j in zip(lda_corpus,documents) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(lda_corpus,documents) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(lda_corpus,documents) if i[2][1] > threshold]
cluster4 = [j for i,j in zip(lda_corpus,documents) if i[3][1] > threshold]


# In[19]:


print (cluster1)
print (cluster2)
print (cluster3)
print (cluster4)


# In[20]:




