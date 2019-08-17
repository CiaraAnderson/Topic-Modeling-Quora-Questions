#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


npr = pd.read_csv('quora_questions.csv')


# In[3]:


npr.head()


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[8]:


dtm = tfidf.fit_transform(npr['Question'])


# In[9]:


dtm


# In[14]:


from sklearn.decomposition import NMF


# In[15]:


nmf_model = NMF(n_components=20,random_state=42)


# In[16]:


nmf_model.fit(dtm)


# In[17]:


for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[18]:


npr.head()


# In[23]:


topic_results = nmf_model.transform(dtm)
npr['Topic'] = topic_results.argmax(axis=1)


# In[24]:


npr.head(20)


# In[ ]:




