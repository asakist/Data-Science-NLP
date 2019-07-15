#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Spacy for lemmatization

import spacy
import nltk
import re
import pandas as pd
import numpy as np
from pprint import pprint
# Gensim
import gensim
from gensim import corpora
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models import LdaModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk # nltk.download('stopwords') #run once
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[2]:


df = pd.read_json('en--2018-01-01---2018-12-31-preprocessed.jsonl', lines=True)
df.head()


# In[3]:


#exploration of the data
#check for NaN or zero values this is the same for pandas dataframes.

print(df.isnull())


# In[4]:


df1 = df['tokens']


# In[5]:


type(df1)


# In[6]:


pretokens = list(df1)


# In[7]:


#this is the number of documents
len(pretokens)


# In[8]:


pretokens[1]


# In[13]:


#prepare the stop_words from NLTK
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','subject', 're', 'edu', 'use']) 


# In[14]:


#define the functions for stop words
#it is important the function to return a list of list.Otherwise we losing information
def remove_stopwords(texts):
    return [[word for word in doc if word not in stop_words and len(word) > 3] for doc in texts]
    
# Remove Stop Words
data_words_nostops = remove_stopwords(pretokens)



# In[18]:


#save as a list directly
import pickle
with open('data_words_nostops2','wb') as f:
    pickle.dump(data_words_nostops1,f)


# In[9]:


#load a list
import pickle
with open('data_words_nostops2','rb') as f:
    data_words_nostops=pickle.load(f)


# In[15]:


# cleaning of the token words 

# matches all hyphens (minus) characters
pattern = '-' 
# empty string
replace = ' '
data_words_nostops = [[re.sub(pattern, replace, word) for word in doc] for doc in data_words_nostops] 

# clean the white spaces
pattern = '\s+' 
# empty string
replace = ''
data_words_nostops = [[re.sub(pattern, replace, word) for word in doc] for doc in data_words_nostops] 

# get rid of empty words
pattern = '' 
data_words_nostops = [[word for word in doc if word != pattern] for doc in data_words_nostops] 


# In[16]:


#save as a list directly
#import pickle
#with open('data_words_nostops1','wb') as f:
#    pickle.dump(data_words_nostops,f)


# In[ ]:


#load a list
import pickle
with open('data_words_nostops1','rb') as f:
    data_words_nostops=pickle.load(f)


# In[17]:


data_words_nostops[0]


# In[14]:


# build bigram & trigram models
#threshold : the higher the fewer phrases.
#min_count :ignore phrases with occurence less than the defined number 
bigram_phrases = Phrases(data_words_nostops, min_count=5, threshold = 100)
bigram_model = Phraser(bigram_phrases)
trigram_phrases = Phrases(bigram_phrases[data_words_nostops],threshold = 100)
trigram_model = Phraser(trigram_phrases)


# In[15]:


#save bigram_model
bigram_model.save('bigram_model')


# In[28]:


#load bigram_model
bigram_model=Phraser.load('bigram_model')


# In[ ]:


#for faster implementation, but even then we have to train first the model to get the bigram_phrases??right??
#bigram_model=gensim.models.phrases.Phraser 


# In[ ]:


print(trigram_model[bigram_model[data_words_nostops[8]]])


# In[31]:


#get the bigram data in a list of list
bigram_data = []
for doc in data_words_nostops:
    bigram_data.append(bigram_model[doc])
    


# In[32]:


len(bigram_data)


# In[33]:


#first lemmatization with spacy library
def lemmatization(bigram_data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    words_out = []
    for sublist in bigram_data:
        doc = nlp(" ".join(sublist)) 
        words_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return words_out


# In[34]:


import spacy
#disable parser,ner for computation time
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])


# In[36]:


data_lemmatized = lemmatization(bigram_data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[37]:


#save list data_lemmatized
#import pickle
#with open('data_lemmatized1','wb') as f:
#    pickle.dump(data_lemmatized,f)


# In[7]:


#load data_lemmatized
import pickle
with open('data_lemmatized1','rb') as f:
    data_lemmatized=pickle.load(f)


# In[27]:


#In our case is not useful to apply stemming,since will reduce the readability of the output. 
#But in case we have data that is needed
# Secondly, stemming with nltk library 
#stemmer = SnowballStemmer("english")
#stem_data = [[stemmer.stem(word) for word in doc]for doc in data_lemmatized]
#or
#for doc in data_lemmatized:
#    for word in doc:
#        stem_data.append(stemmer.stem(word))


# In[35]:


#stem_data[8]


# In[8]:


stem_data=data_lemmatized


# In[63]:


#Create a dictionary for our data(dictionary over all the words of our data)
dictionary = gensim.corpora.Dictionary(stem_data)


# In[64]:


#save dictionary to disk
#dictionary.save('dictionary_all_data')


# In[4]:


# Load dictionary back
dictionary = corpora.Dictionary.load('dictionary_all_data')


# In[65]:


#corpus=Document term frequency(the dictionary splited over each document of the data)
corpus = [dictionary.doc2bow(doc) for doc in stem_data]


# In[66]:


#save corpus to disk
#corpora.MmCorpus.serialize('corpus.mm',corpus)


# In[5]:


#load back corpus
corpus = corpora.MmCorpus('corpus.mm')


# In[29]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=7, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=1,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


#lda_model.save('20 topics-alldata')


# In[33]:


lda_model=LdaModel.load('20 topics-alldata')


# In[28]:


#top 10 of most important key words for each model-10 topics
pprint(lda_model.print_topics())


# In[7]:


#top 10 of most important key words for each model-20 topics
pprint(lda_model.print_topics())


# In[10]:


#distribution of each document over the topics
doc_lda = lda_model[corpus] 


# In[8]:


corpus[0] # sub-dictionary of document[0] 


# In[11]:


doc_lda[0] #topic 5 is the dominant topic here


# In[12]:


#doc_lda[0] is the distribution of doc 0 over the topics
#checking if the dominant topic(5) is related with the content of document[0].
#topic 5 is about people,pron,work,time.. so there is relation.Our model performed good here.
print(df['content'][0])


# In[19]:


# Compute Coherence Score#all data/ 10 topics
coherence_model_lda = CoherenceModel(model=lda_model, texts=stem_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[10]:


# Compute Coherence Score#all data/ 20 topics
coherence_model_lda = CoherenceModel(model=lda_model, texts=stem_data, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[20]:


# Visualize the 10-topics
pyLDAvis.enable_notebook()
plot = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(plot)


# In[21]:


#save the interactive plot
pyLDAvis.save_html(plot, 'lda.html_10_topics')


# In[ ]:


#load the pyLDAvis
pyLDAvis.enable_notebook(local=False)


# In[ ]:


#lda_model[coprus] needs 4 indexes to be expressed in case the lda_model.per_word_topics argument is True.
#checking if this parameter is True
#lda_model.per_word_topics


# In[106]:


def format_topics_sentences(ldamodel=lda_model, corpus=corpus,texts=df['content']):
    # Init output
    doc_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                doc_topics_df = doc_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    doc_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    doc_topics_df = pd.concat([doc_topics_df, contents], axis=1)
    return(doc_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df['content'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)


# In[109]:


df_dominant_topic.head()


# In[ ]:


#save
#df_dominant_topic_all.to_csv('df_dominant_topic_all')


# In[ ]:


#load the df_dominant_topic_all dataframe (20 topics)
df_dominant_topic_all= pd.read_csv('df_dominant_topic_all')


# In[41]:


#keep only the date from the datetime variable 'date'
from datetime import date
df_dominant_topic_all['just_date'] = df['date'].dt.date


# In[42]:


df_dominant_topic_all.head()


# In[28]:


#find the count of the words for each doc 
doc_len=[]
for d in range(len(df['tokens'])):
    a=len(df['tokens'][d])
    doc_len.append(a)


# In[29]:


# Plot histogram
plt.figure(figsize=(16,7))
plt.hist(doc_len,bins=500, color='navy')
plt.text(750, 10000, "Mean   : " + str(round(np.mean(doc_len))))
plt.text(750,  9000, "Median : " + str(round(np.median(doc_len))))
plt.text(750,  8000, "Stdev  : " + str(round(np.std(doc_len))))
plt.text(750,  7000, "10%ile  : " + str(round(np.quantile(doc_len, q=0.10))))
plt.text(750,  6000, "95%ile : " + str(round(np.quantile(doc_len, q=0.95))))

plt.gca().set(xlim=(0, 1000), ylabel='No_of_Document', xlabel='Document_word_count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,1000,10))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()


# In[43]:


#contribution of each topic over the body of documents
doc_per_topic = df_dominant_topic_all['Dominant_Topic'].value_counts().reset_index()


# In[72]:


#rename the columns
doc_per_topic.columns=['dominant_topic','num_doc']


# In[73]:


topic_contribution_perc=round(doc_per_topic['num_doc']/doc_per_topic['num_doc'].sum(),3)


# In[74]:


topic_doc_distribution=pd.concat([doc_per_topic,topic_contribution_perc],axis=1)


# In[75]:


topic_doc_distribution.columns=['dominant_topic','num_doc','perc_doc']


# In[69]:


topic_doc_distribution.head(10)


# In[51]:


plt.figure(figsize=(26,10)) 
topic_doc_distribution.plot.bar(x='dominant_topic',y='num_doc')
plt.show()


# In[71]:


#per day,per topic, amount of documents 
#setting as_index=false the group_by attributes are not used as indexes
per_day_results=df_dominant_topic_all.groupby([ 'Dominant_Topic', 'just_date'],as_index=False)['doc_num'].count()


# In[76]:


per_day_results['sum_of_doc']=per_day_results['doc_num']


# In[77]:


per_day_results=per_day_results.drop('doc_num',axis=1)


# In[78]:


per_day_results.head(10)


# In[79]:


from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

fig, axes = plt.subplots(10, 2, figsize=(16,30), sharey=True, dpi=160)

for i, ax in enumerate(axes.flat):
    ax.bar(x='just_date', height="sum_of_doc", data=per_day_results.loc[per_day_results.Dominant_Topic==i, :])
    ax.set_ylabel('Doc_Count')
    ax.set_title('Topic: ' + str(i), fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(per_day_results['just_date'], rotation=60, horizontalalignment= 'right')
    fig.tight_layout(w_pad=2)
    #fig.suptitle('Doc Count per Topic/per day', fontsize=22, y=1.00)    
plt.show()


# In[55]:


#replace nan with zero values
df_dominant_topic_all['Topic_Perc_Contrib']=df_dominant_topic_all['Topic_Perc_Contrib'].replace('nan',np.nan).fillna(0)


# In[56]:


#find the most representative docs for each dominant topic
most_repre_doc=[]
for i in df_dominant_topic_all['Dominant_Topic']:
    repre_doc=df_dominant_topic_all[df_dominant_topic_all.Dominant_Topic==i]
    if len(repre_doc['Topic_Perc_Contrib'])!=0:
        repre_doc1=repre_doc.loc[repre_doc['Topic_Perc_Contrib'].idxmax(),:]
        most_repre_doc.append(repre_doc1)


# In[57]:


#save 
#import pickle
#with open('most_repre_doc','wb') as f:
#    pickle.dump(most_repre_doc,f)


# In[ ]:


#load 
import pickle
with open('most_repre_doc','rb') as f:
    most_repre_doc=pickle.load(f)


# In[59]:


#try for the range of number that dominant topics are.20.
#here we are getting the 20 most representative doc for each topic
most_repre_doc20=[]
for i in range(20):
    repre_doc=df_dominant_topic_all[df_dominant_topic_all.Dominant_Topic==i]
    if len(repre_doc['Topic_Perc_Contrib'])!=0:
        repre_doc1=repre_doc.loc[repre_doc['Topic_Perc_Contrib'].idxmax(),:] 
        most_repre_doc20.append(repre_doc1)


# In[60]:


most_repre_doc20


# In[66]:


#placing manually labels to our Topics
#topic 0 --> label:Technology
df['content'][209239]


# In[62]:


#topic 2 --> label:Investments
df['content'][325465]


# In[61]:


#topic 4 --> label:federal-case-issue
df['content'][72650]


# In[53]:


#topic 8 --> label:U.S. military forces
df['content'][252828]


# In[54]:


#topic 10--> label: stock market
df['content'][418]


# In[55]:


#topic 11--> label: shares,trade,private company
df['content'][33809]


# In[56]:


#topic 17--> label: administration,company
df['content'][404]


# In[58]:


#topic 19--> label: elections
df['content'][6279]


# In[3]:


get_ipython().system('jupyter nbconvert --to html nlp-git-topic.ipynb')


# In[ ]:




