#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import glob
import nltk

import itertools
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import chart_studio.plotly as py
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE
import plotly.express as px

from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from fastcluster import linkage
from matplotlib.colors import rgb2hex, colorConverter


# In[3]:


all_files=glob.glob("/Users/tanyachauhan/Documents/Diksha_All_Csv/"+"/*.csv")
dflist = []


# In[4]:


for filename in all_files:
    # Dataframe of one file
    df_sm = pd.read_csv(filename, index_col=None, header=0)
    dflist.append(df_sm)
    
df = pd.concat(dflist, axis=0, ignore_index=True)


# In[5]:


df.dropna(subset=["Review Text"],inplace=True)


# In[6]:


df


# In[7]:


eng_data = df.loc[df['Reviewer Language']=='en']
eng_df = pd.DataFrame(eng_data)
# eng_df


# In[8]:


eng_df.reset_index(inplace = True) 


# In[11]:


eng_df 


# In[213]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[214]:


grammar =r"""
  NP: {<DT|JJ|NN.*>+}          
  PP: {<IN><NP>}            
  VP: {<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE: {<NP><VP>}          
  """
      
    


# In[215]:


def phrase_extraction(text, grammar):
    text = text.lower()
    sentence_re = r'''(?x)          
      (?:[A-Z]\.)+        
    | \w+(?:-\w+)*        
    | \$?\d+(?:\.\d+)?%?  
    | \.\.\.              
    | [][.,;"'?():_`-]    
    '''
    
    ls = [] 
    word_token_ls = text.split(" ")

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    
    chunker = nltk.RegexpParser(grammar)
    
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    for term in terms:
        ls.append(" ".join(term)) 
    return list(set(ls))


# In[17]:


ls= list(eng_df["Review Text"])



# In[18]:


out = map(lambda x:x.lower(), ls)  
review_text_lower = list(out)
review_text_lower


# In[19]:


# Numbers removing

import re
review_text_lower_wdoutno = list(map(lambda x: re.sub(r'\d+', '', x), review_text_lower)) 
review_text_lower_wdoutno


# In[20]:


# Remove punctuation
import string
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

review_text_wdout_punct = []
for i in review_text_lower_wdoutno:
    x = remove_punctuation(i)
    review_text_wdout_punct.append(x)
review_text_wdout_punct   


# In[21]:


# remove whitespace from text 
def remove_whitespace(text): 
    return " ".join(text.split()) 

review_text_wdout_whitespace = []
for i in review_text_wdout_punct:
    x = remove_whitespace(i)
    review_text_wdout_whitespace.append(x)
review_text_wdout_whitespace  


# In[22]:


# convert a list to string    
def listToString(s):  
    str1 = ""   
    for ele in s:  
        str1 += ele   
        str1 += ' '
    return str1  
        


# In[23]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
# remove stopwords function 
def remove_stopwords(text): 
    stop_words = list(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text 
  
review_text_wdout_stopwords = []
for i in review_text_wdout_whitespace:
    x = remove_stopwords(i)
    y = listToString(x)
    review_text_wdout_stopwords.append(y)
review_text_wdout_stopwords  


# In[24]:


# remove emoji
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

review_text_wdout_emojis = []
for i in review_text_wdout_stopwords:
    x = deEmojify(i)
    review_text_wdout_emojis.append(x)
review_text_wdout_emojis  


# In[25]:


# lemmatization
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer() 

# lemmatize string 
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas 
  
review_text_lemmas = []
for i in review_text_wdout_emojis:
    x = lemmatize_word(i)
    y = listToString(x)
    review_text_lemmas.append(y)
review_text_lemmas  


# In[26]:


# remove review text containing less than 3 words using regex (findall()) 

import re 

processed_review_text = []
for i in review_text_lemmas:
    res = len(re.findall(r'\w+', i)) 
    if(res>=3):
        processed_review_text.append(i)
        
processed_review_text    


# In[27]:


# Vectorization

def vectorization_of_list(input_list):
    #word embedding(vectorization)
    embed = hub.Module("/Users/tanyachauhan/Downloads/3")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(input_list))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[28]:


vectorized_review_frame = vectorization_of_list(processed_review_text)
vectorized_review_frame


# In[105]:


prob=['poor video quality','download problem','content not available','scanning problem','video availability problem']
for i in range(len(prob)):
    prob[i]=prob[i].lower()
print(prob)


# In[30]:


def vectorization_of_list(input_list):
    #word embedding(vectorization)
    embed = hub.Module("/Users/tanyachauhan/Downloads/3")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(input_list))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[168]:


# vectorized_review_frame.loc[0]


# In[99]:


vectorized_review_frame.index=processed_review_text    


# In[100]:


vectorized_review_frame


# In[163]:


# review_reset_df.index(inplace = True) 
def TSNE_3D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensinality
    X_embedded = TSNE(n_components=2).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter(vec_df,x="ft1",y="ft2")
    fig.show()


# In[164]:


TSNE_3D(vectorized_review_frame)


# In[102]:


review_reset_df


# In[35]:


final_review_frame=review_reset_df.drop(['level_0', 'index'], axis=1)
final_review_frame


# In[106]:


vectorized_a_frame.index=prob


# In[107]:


vectorized_a_frame


# In[115]:


list(list(vectorized_a_frame.iloc[0]))


# In[118]:


c=cosine_similarity([list(vectorized_review_frame.iloc[0])],[list(vectorized_a_frame.iloc[0])])
c


# In[111]:


list(vectorized_review_frame.iloc[0])


# In[133]:


from sklearn.metrics.pairwise import cosine_similarity
l1=[]
for i in range(len(vectorized_review_frame)):
#     l2=[]
    for j in range(len(vectorized_a_frame)):
            
        c=cosine_similarity([list(vectorized_review_frame.iloc[i])],[list(vectorized_a_frame.iloc[j])])
        l1.append([ vectorized_review_frame.index[i],vectorized_a_frame.index[j],c[0][0]])


# In[134]:


print(l1)


# In[274]:


review_prob_score_df=pd.DataFrame(l1)


# In[293]:


review_prob_score_df.rename(columns={0:'Review Text',1: 'Problem',2:'cosine_similarity'},inplace=True)
review_prob_score_df


# In[314]:


cosine_above_threshold=review_prob_score_df[review_prob_score_df['cosine_similarity']>=0.15]
cosine_above_threshold


# In[315]:


cosine_above_threshold.sort_values(by ='Problem' )


# In[319]:


content_problem=review_prob_score_df[review_prob_score_df['Problem']=="content not available"]
content_problem


# In[320]:


content_problem[content_problem['cosine_similarity']>=0.15]


# In[304]:


video_availability_problem=review_prob_score_df[review_prob_score_df['Problem']=="video availability problem"]
video_availability_problem


# In[307]:


video_availability_problem[video_availability_problem['cosine_similarity']>=0.15]


# In[308]:


download_problem=review_prob_score_df[review_prob_score_df['Problem']=="download problem"]
download_problem


# In[309]:


download_problem[download_problem['cosine_similarity']>=0.15]


# In[321]:


poor_video_quality_problem=review_prob_score_df[review_prob_score_df['Problem']=="poor video quality"]
poor_video_quality_problem


# In[322]:


poor_video_quality_problem[poor_video_quality_problem['cosine_similarity']>=0.15]


# In[323]:


scanning_problem=review_prob_score_df[review_prob_score_df['Problem']=="scanning problem"]
scanning_problem


# In[324]:


scanning_problem[scanning_problem['cosine_similarity']>=0.15]


# In[ ]:





# In[ ]:





# In[297]:


df_percent = pd.crosstab(review_prob_score_df.cosine_similarity,review_prob_score_df.Problem,
                         normalize = 'index').rename_axis(None)

# Multiply all percentages by 100 for graphing. 
df_percent *= 100


# In[298]:


df_percent[0:50]


# In[209]:


review_prob_score_df["Review Text"].value_counts()


# In[ ]:





# In[216]:


parsingkey4 = pd.DataFrame(columns=['Review Text', 'Keywords'])

for i in processed_review_text:
    if(i==''):
        parsingkey4 = parsingkey4.append({'Review Text': i, 'Keywords': '[]'}, ignore_index=True)
    else:
        x=phrase_extraction(i, grammar)
        parsingkey4 = parsingkey4.append({'Review Text': i, 'Keywords': x}, ignore_index=True)

parsingkey4
    
    


# In[ ]:





# In[217]:


# Sentiment Analysis
from textblob import TextBlob

# List of sentiments of each review text
sentiments = []

for i in processed_review_text:
    blob = TextBlob(i)
    s = blob.sentiment
    sentiments.append(s)
    
print(sentiments)


# In[218]:


sentiment = pd.DataFrame(sentiments)
sentiment


# In[221]:


parsingkey4.insert(2,"Polarity",sentiment["polarity"])


# In[222]:


parsingkey4


# In[224]:


parsingkey4.insert(3,"Subjectivity",sentiment["subjectivity"])


# In[225]:


parsingkey4


# In[226]:


def sentiment_score_count(review_df_ls):
    sentiment_class = []
    pos_score = []
    neu_score = []
    neg_score = []
    for i in processed_review_text:
        sid_obj = SentimentIntensityAnalyzer()  
        sentiment_dict = sid_obj.polarity_scores(i)  
        #print("Overall sentiment dictionary is : ", sentiment_dict) 
        neg_score.append(sentiment_dict['pos']) 
        neu_score.append(sentiment_dict['neg']) 
        pos_score.append(sentiment_dict['neu'])
        if sentiment_dict['compound'] >= 0.15 : 
            sentiment_class.append("Positive") 

        elif sentiment_dict['compound'] <= -0.15 : 
            sentiment_class.append("Negative") 

        else : 
            sentiment_class.append("Neutral")
            
    sentiment_df = pd.DataFrame(list(zip(neg_score, neu_score, pos_score, sentiment_class )), 
               columns =['pos_score','neg_score','neu_score','sentiment_class']) 
    return sentiment_df 


# In[227]:


sentiment_df = sentiment_score_count(processed_review_text)
sentiment_df


# In[273]:


joined_df = parsingkey4.join(sentiment_df) 
c


# In[272]:


final_df2 = joined_df.join(sentiment) 
final_df2


# In[270]:


neg = final_df2.loc[final_df2['sentiment_class']=='Negative']
neg_df = pd.DataFrame(neg)
neg_df


# In[231]:


list_neg_review= list(neg_df["Review Text"])
print(list_neg_review)


# In[268]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

import matplotlib.pyplot as plt
plt.style.use("ggplot")


# In[ ]:





# In[166]:


cosine_list = review_prob_score_df['cosine_similarity'].tolist()
cosine_list


# In[183]:


cosine_df=pd.DataFrame(cosine_list)
cosine_df.rename(columns={0:'cosine_similarity'},inplace=True)
cosine_df


# In[198]:


def TSNE_2D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensionality
    X_embedded = TSNE(n_components=2).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter(vec_df,x="ft1",y="ft2")
    fig.show()


# In[199]:


TSNE_2D(cosine_df)


# In[187]:


def dendrogram_genetator(df):
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    


# In[188]:


cosine_df


# In[189]:


get_ipython().run_line_magic('pylab', 'inline')
#Reduce Dimensinality
X_embedded = TSNE(n_components=2).fit_transform(cosine_df)
vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2"])
vec_df


# In[190]:


dendrogram_genetator(vec_df)


# In[191]:


def dendrogram_genetator_with_thresold(df,thresold):
    plt.figure(figsize=(10, 7))
#     y=800
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.axhline(thresold, color='r', linestyle='--')
    


# In[192]:


dendrogram_genetator_with_thresold(vec_df,40000)


# In[193]:


def hierarchial_clustering(df):
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(df)
    
    plt.figure(figsize=(10, 7))  
    plt.scatter(df['ft1'], df['ft2'], c=cluster.labels_) 
    


# In[194]:


hierarchial_clustering(vec_df)


# In[195]:


vec_df.set_index(cosine_df["cosine_similarity"], inplace = True) 
vec_df


# In[196]:


def cluster_element_extraction(vec_df):
    sns.set_palette('Set1', 10, 0.65)
    palette = (sns.color_palette())
    #set_link_color_palette(map(rgb2hex, palette))
    sns.set_style('white')
    
    np.random.seed(25)
    
    link = linkage(vec_df, metric='correlation', method='ward')

    figsize(8, 3)
    den = dendrogram(link, labels=vec_df.index)
    plt.xticks(rotation=90)
    no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
    sns.despine(**no_spine);

    plt.tight_layout()
    plt.savefig('feb2.png');
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    class Clusters(dict):
        def _repr_html_(self):
            html = '<table style="border: 0;">'
            for c in self:
                hx = rgb2hex(colorConverter.to_rgb(c))
                html += '<tr style="border: 0;">'                 '<td style="background-color: {0}; '                            'border: 0;">'                 '<code style="background-color: {0};">'.format(hx)
                html += c + '</code></td>'
                html += '<td style="border: 0"><code>' 
                html += repr(self[c]) + '</code>'
                html += '</td></tr>'

            html += '</table>'

            return html
    
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den['ivl'][i] for i in l]
        cluster_classes[c] = i_l
        
    return cluster_classes
    


# In[197]:


cluster_element_extraction(vec_df)


# In[ ]:





# In[ ]:





# In[160]:


import matplotlib.pyplot as plt
import seaborn as sns
fig_dims = (12,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x = 'Problem', y = 'cosine_similarity', ax=ax, data=review_prob_score_df)


# In[153]:


review_prob_score_df.to_csv ('/Users/tanyachauhan/Documents/review_problem_score.csv', index = False, header=True)


# In[180]:


return frame


# In[ ]:





# In[ ]:


vectorized_cosine_similarity_frame.index=cosine_list 


# In[38]:


a_reset_df


# In[1]:


# final_a_frame=a_reset_df.drop(['level_0', 'index'], axis=1)
# final_a_frame


# In[2]:


# listOfADFRows = final_a_frame.to_numpy().tolist()
# listOfADFRows


# In[3]:


# listOfRDFRows= final_review_frame.to_numpy().tolist()
# listOfRDFRows


# In[57]:


# cosine_similarity(review_reset_df[0],a_reset_df[0])


# In[ ]:





# In[98]:


print(l1)


# In[91]:


listOfADFRows[0]


# In[94]:


k=cosine_similarity([listOfADFRows[0]],[listOfRDFRows[0]])
k


# In[96]:


k[0][0]


# In[ ]:





# In[623]:


a=['poor video quality','download problem','content not available','scanning problem','video availability problem']
for i in range(len(a)):
    a[i]=a[i].lower()
print(a)
type(a)


# In[619]:


list_of_list=[processed_review_text,a]
list_of_list


# In[17]:


# new_df.insert(2,'Star Rating',eng_df['Star Rating'])


# In[19]:


# def unlisting_list_of_list(list_of_list):
#     merged = list(itertools.chain(*list_of_list))
#     kywrd_ls= list(set(merged))
#     return(kywrd_ls)

def unlisting_list_of_list(list_of_list):
    merged = list(itertools.chain(*list_of_list))
    review_ls= list(set(merged))
    return(review_ls)


# In[20]:


# kywrd_df_ls = list(new_df["Keywords"])
# kywrd_df_ls


# In[21]:


# merged = list(itertools.chain(*kywrd_df_ls))
# kywrd_ls= list(set(merged))
# kywrd_ls
# kywrd_ls=unlisting_list_of_list(kywrd_df_ls)
# print(kywrd_ls)


# In[22]:


review_df_ls = list(new_df["Review Text"])
review_df_ls


# In[135]:


def sentiment_score_count(review_df_ls):
    sentiment_class = []
    pos_score = []
    neu_score = []
    neg_score = []
    for i in review_df_ls:
        sid_obj = SentimentIntensityAnalyzer()  
        sentiment_dict = sid_obj.polarity_scores(i)  
        #print("Overall sentiment dictionary is : ", sentiment_dict) 
        neg_score.append(sentiment_dict['pos']) 
        neu_score.append(sentiment_dict['neg']) 
        pos_score.append(sentiment_dict['neu'])
        if sentiment_dict['compound'] >= 0.05 : 
            sentiment_class.append("Positive") 

        elif sentiment_dict['compound'] <= -0.05 : 
            sentiment_class.append("Negative") 

        else : 
            sentiment_class.append("Neutral")
            
    sentiment_df = pd.DataFrame(list(zip(neg_score, neu_score, pos_score, sentiment_class )), 
               columns =['pos_score','neg_score','neu_score','sentiment_class']) 
    return sentiment_df 


# In[136]:


sentiment_df = sentiment_score_count(review_df_ls)
sentiment_df


# In[137]:


# joining two dataframes
joined_df = new_df.join(sentiment_df) 
joined_df


# In[138]:


eng_df.sort_values("Review Submit Date and Time", axis = 0, ascending = True, 
                 inplace = True, na_position ='first') 
  
eng_df
list_of_date = eng_df['Review Submit Date and Time'].to_list()
# print(list_of_date)
list_year_month=[]
for i in list_of_date:
    list_year_month.append(i[:7])
print(list_year_month)
    


# In[139]:


# selected_col = eng_df[['Review Submit Date and Time']].copy()
# selected_col
selected_col=pd.DataFrame(list_year_month,columns = ["Review Submit Date and Time"]) 
selected_col


# In[140]:


final_df = joined_df.join(selected_col) 
final_df


# In[141]:


def polarity_and_subjectivity(review_df_ls):
    # List of sentiments of each review text
    sentiments = []

    for i in review_df_ls:
        blob = TextBlob(i)
        s = blob.sentiment
        sentiments.append(s)
    
    # Creating a dataframe of sentiments
    sentiment = pd.DataFrame(sentiments)
    return sentiment


# In[142]:


sentiment = polarity_and_subjectivity(review_df_ls)
sentiment


# In[143]:


final_df2 = final_df.join(sentiment) 
final_df2[51:100]


# In[144]:


final_df2.drop(["Keywords"],axis = 1,inplace = True) 
  
# display 
final_df2 


# In[346]:


pos = final_df2.loc[final_df2['sentiment_class']=='Positive']
pos_df = pd.DataFrame(pos)
pos_df[0:50]


# In[351]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[352]:


grammar =r"""
  NP: {<DT|JJ|NN.*>+}          
  PP: {<IN><NP>}            
  VP: {<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE: {<NP><VP>}          
  """
      
    


# In[353]:


def phrase_extraction(text, grammar):
    text = text.lower()
    sentence_re = r'''(?x)          
      (?:[A-Z]\.)+        
    | \w+(?:-\w+)*        
    | \$?\d+(?:\.\d+)?%?  
    | \.\.\.              
    | [][.,;"'?():_`-]    
    '''
    
    ls = [] 
    word_token_ls = text.split(" ")

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    
    chunker = nltk.RegexpParser(grammar)
    
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    for term in terms:
        ls.append(" ".join(term)) 
    return list(set(ls))


# In[354]:


list_pos_review= list(pos_df["Review Text"])
print(list_pos_review)


# In[504]:


def vectorization_of_list(positive_review):
    #word embedding(vectorization)
    embed = hub.Module("/Users/tanyachauhan/Downloads/3")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(positive_review))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[505]:


frame = vectorization_of_list(list_pos_review)
frame


# In[356]:


new_pos_df = pd.DataFrame(columns=['Review Text', 'Keywords'])

for i in list_pos_review:
    if(i == ''):
        new_pos_df = new_pos_df.append({'Review Text': i, 'Keywords': '[]'}, ignore_index=True)
    else:
        x = phrase_extraction(i, grammar)
        new_pos_df = new_pos_df.append({'Review Text': i, 'Keywords': x}, ignore_index=True)

new_pos_df[0:50]
    


# In[415]:


pos_l=list(new_pos_df["Keywords"])
pos_l


# In[417]:


list_pos_df=pd.DataFrame(pos_l)
pos_count=list_pos_df[0].value_counts()
pos_count[0:50]


# In[459]:


import re
my_pos_list=list_pos_df[0].tolist()
my_pos_list
new_pos_list = [y.lower() for y in list_pos_review if re.search('exlent', y)]
new_pos_list


# In[374]:


neg = final_df2.loc[final_df2['sentiment_class']=='Negative']
neg_df = pd.DataFrame(neg)
neg_df[0:50]


# In[375]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[376]:


grammar =r"""
  NP: {<DT|JJ|NN.*>+}          
  PP: {<IN><NP>}            
  VP: {<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE: {<NP><VP>}          
  """
      
    


# In[377]:


def phrase_extraction(text, grammar):
    text = text.lower()
    sentence_re = r'''(?x)          
      (?:[A-Z]\.)+        
    | \w+(?:-\w+)*        
    | \$?\d+(?:\.\d+)?%?  
    | \.\.\.              
    | [][.,;"'?():_`-]    
    '''
    
    ls = [] 
    word_token_ls = text.split(" ")

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    
    chunker = nltk.RegexpParser(grammar)
    
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    for term in terms:
        ls.append(" ".join(term)) 
    return list(set(ls))


# In[378]:


list_neg_review= list(neg_df["Review Text"])
print(list_neg_review)


# In[491]:


def vectorization_of_list(negative_review):
    #word embedding(vectorization)
    embed = hub.Module("/Users/tanyachauhan/Downloads/3")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(negative_review))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[492]:


frame = vectorization_of_list(list_neg_review)
frame


# In[493]:


negative_review_text_df = pd.DataFrame(list_neg_review,columns=['Review Text'])
negative_review_text_df


# In[495]:


frame.set_index(negative_review_text_df["Review Text"], inplace = True) 


# In[496]:


frame


# In[497]:


def TSNE_3D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensinality
    X_embedded = TSNE(n_components=3).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter_3d(vec_df,x="ft1",y="ft2",z="ft3")
    fig.show()


# In[498]:


TSNE_3D(frame)


# In[499]:


def dendrogram_genetator(df):
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    


# In[500]:


frame


# In[501]:


get_ipython().run_line_magic('pylab', 'inline')
#Reduce Dimensinality
X_embedded = TSNE(n_components=3).fit_transform(frame)
vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
vec_df


# In[509]:


dendrogram_genetator(vec_df)


# In[510]:


def dendrogram_genetator_with_thresold(df,thresold):
    plt.figure(figsize=(10, 7))
#     y=800
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.axhline(thresold, color='r', linestyle='--')
    


# In[511]:


dendrogram_genetator_with_thresold(vec_df,850)


# In[ ]:





# In[502]:


def hierarchial_clustering(df):
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(df)
    
    plt.figure(figsize=(10, 7))  
    plt.scatter(df['ft1'], df['ft2'], c=cluster.labels_) 
    


# In[503]:


hierarchial_clustering(vec_df)


# In[514]:


vec_df.set_index(negative_review_text_df["Review Text"], inplace = True) 
vec_df


# In[515]:


def cluster_element_extraction(vec_df):
    sns.set_palette('Set1', 10, 0.65)
    palette = (sns.color_palette())
    #set_link_color_palette(map(rgb2hex, palette))
    sns.set_style('white')
    
    np.random.seed(25)
    
    link = linkage(vec_df, metric='correlation', method='ward')

    figsize(8, 3)
    den = dendrogram(link, labels=vec_df.index)
    plt.xticks(rotation=90)
    no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
    sns.despine(**no_spine);

    plt.tight_layout()
    plt.savefig('feb2.png');
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    class Clusters(dict):
        def _repr_html_(self):
            html = '<table style="border: 0;">'
            for c in self:
                hx = rgb2hex(colorConverter.to_rgb(c))
                html += '<tr style="border: 0;">'                 '<td style="background-color: {0}; '                            'border: 0;">'                 '<code style="background-color: {0};">'.format(hx)
                html += c + '</code></td>'
                html += '<td style="border: 0"><code>' 
                html += repr(self[c]) + '</code>'
                html += '</td></tr>'

            html += '</table>'

            return html
    
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den['ivl'][i] for i in l]
        cluster_classes[c] = i_l
        
    return cluster_classes
    


# In[517]:


cluster_element_extraction(vec_df.head(3213))


# In[379]:


new_neg_df = pd.DataFrame(columns=['Review Text', 'Keywords'])

for i in list_neg_review:
    if(i == ''):
        new_neg_df = new_neg_df.append({'Review Text': i, 'Keywords': '[]'}, ignore_index=True)
    else:
        x = phrase_extraction(i, grammar)
        new_neg_df = new_neg_df.append({'Review Text': i, 'Keywords': x}, ignore_index=True)

new_neg_df[0:50]
    


# In[382]:


neg_l=list(new_neg_df["Keywords"])
neg_l


# In[383]:


list_neg_df=pd.DataFrame(neg_l)
neg_count=list_neg_df[0].value_counts()
neg_count[0:50]


# In[486]:


import re
my_neg_list=list_neg_df[0].tolist()
my_neg_list
new_neg_list = [y.lower() for y in list_neg_review if re.search('material', y)]
new_neg_list


# In[384]:


neu = final_df2.loc[final_df2['sentiment_class']=='Neutral']
neu_df = pd.DataFrame(neu)
neu_df[0:50]


# In[385]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[386]:


grammar =r"""
  NP: {<DT|JJ|NN.*>+}          
  PP: {<IN><NP>}            
  VP: {<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE: {<NP><VP>}          
  """
      
    


# In[387]:


def phrase_extraction(text, grammar):
    text = text.lower()
    sentence_re = r'''(?x)          
      (?:[A-Z]\.)+        
    | \w+(?:-\w+)*        
    | \$?\d+(?:\.\d+)?%?  
    | \.\.\.              
    | [][.,;"'?():_`-]    
    '''
    
    ls = [] 
    word_token_ls = text.split(" ")

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    
    chunker = nltk.RegexpParser(grammar)
    
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    for term in terms:
        ls.append(" ".join(term)) 
    return list(set(ls))


# In[388]:


list_neu_review= list(neu_df["Review Text"])
print(list_neu_review)


# In[389]:


new_neu_df = pd.DataFrame(columns=['Review Text', 'Keywords'])

for i in list_neu_review:
    if(i == ''):
        new_neu_df = new_neu_df.append({'Review Text': i, 'Keywords': '[]'}, ignore_index=True)
    else:
        x = phrase_extraction(i, grammar)
        new_neu_df = new_neu_df.append({'Review Text': i, 'Keywords': x}, ignore_index=True)

new_neu_df[0:50]
    


# In[391]:


neu_l=list(new_neu_df["Keywords"])
neu_l


# In[392]:


list_neu_df=pd.DataFrame(neu_l)
neu_count=list_neu_df[0].value_counts()
neu_count[0:50]


# In[393]:


# event_dictionary ={'Positive' : >0.5, 'Neutral' : 0.5, 'Negative' : <0.5} 
  
# # Add a new column named 'Price' 
# final_df2['polarity'] = final_df2['sentiment_class'].map(event_dictionary) 
  
# # Print the DataFrame 
# print(final_df2)

final_df2["sentiment_class"].value_counts()


# In[148]:


# final_df2["polarity"].value_counts()
# final_df2['polarity'].value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)

final_df2["polarity"].describe()


# In[316]:


final_df2["sentiment_class"].describe()


# In[317]:


# final_df2["Review Text"]=["Review Text"].str.lower()

# final_df2


# In[318]:


# a=final_df2["Review Text"].value_counts()
# a[0:50]


# In[319]:


final_df2["Review Text"].describe()


# In[320]:


mylist=final_df2["Review Text"].tolist()


# In[321]:


out = map(lambda x:x.lower(), mylist)
output = list(out)
print(output)
 


# In[322]:


# a=final_df2["Review Text"].value_counts()
# a[0:50]

lowerdata=pd.DataFrame(output)
lowerdata[0].value_counts()


# In[404]:


import re
my_list=lowerdata[0].tolist()
new_list = [x.lower() for x in mylist if re.search("problems", x)]
new_list
# for item in new_list:
#     print(item)


# In[338]:


def CountFrequency(new_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in new_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
  
    for key, value in freq.items(): 
        print(key,":", value) 
CountFrequency(new_list) 


# In[325]:


final_df2.to_csv('/Users/tanyachauhan/Documents/Diksha_Sentiments_Analysis.csv', index = False, header=True)
final_df2


# In[ ]:





# In[36]:


# reviewdf = final_df["Keywords"]
# reviewdf


# In[83]:


def wordcloud_generator(reviewdf):
    comment_words = ' '
    stopwords = set(STOPWORDS) 
    for val in reviewdf: 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width = 800, height = 800, 
    background_color = 'white', 
    stopwords = stopwords, 
    min_font_size = 10).generate(comment_words) 

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()
    


# In[84]:


wordcloud_generator(reviewdf)


# In[ ]:





# In[85]:


def sentiment_graph_for_each_rating(df):
    g = sns.FacetGrid(df, col = "sentiment_class")
    g.map(plt.hist, "Star Rating")
    plt.show()


# In[86]:


sentiment_graph_for_each_rating(final_df2)


# In[87]:


def vectorization_of_list(keyword_list):
    #word embedding(vectorization)
    embed = hub.Module("/Users/rishabhsrivastava/Downloads/vectorization_trained_dataset/")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(keyword_list))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[88]:


frame = vectorization_of_list(kywrd_ls)
frame


# In[127]:


review_frame = vectorization_of_list(review_df_ls)
review_frame


# In[128]:


review_text_df = pd.DataFrame(review_df_ls,columns=['Review Text'])
review_text_df


# In[129]:


review_frame.set_index(review_text_df["Review Text"], inplace = True) 


# In[131]:


review_frame


# In[130]:


review_frame.to_csv ('/Users/rishabhsrivastava/Downloads/CSVS/Vectorized_review_text.csv', index = False, header=True)


# In[89]:


keywordsdf = pd.DataFrame(kywrd_ls,columns=['keywords'])
keywordsdf


# In[90]:


frame.set_index(keywordsdf["keywords"], inplace = True) 
frame


# In[91]:


def TSNE_3D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensinality
    X_embedded = TSNE(n_components=3).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter_3d(vec_df,x="ft1",y="ft2",z="ft3")
    fig.show()


# In[92]:


TSNE_3D(frame)


# In[132]:


TSNE_3D(review_frame)


# In[93]:


# frame.insert(512,"keywords",frame.index)
# frame.reset_index(drop=True, inplace=True)


# In[94]:


# frame["keywords"]


# In[97]:


def dendrogram_genetator(df):
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    


# In[98]:


frame


# In[99]:


get_ipython().run_line_magic('pylab', 'inline')
#Reduce Dimensinality
X_embedded = TSNE(n_components=3).fit_transform(frame)
vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
vec_df


# In[133]:


get_ipython().run_line_magic('pylab', 'inline')
#Reduce Dimensinality
X_embedded = TSNE(n_components=3).fit_transform(review_frame)
review_vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
review_vec_df


# In[103]:


dendrogram_genetator(vec_df)


# In[134]:


dendrogram_genetator(review_vec_df)


# In[106]:


def dendrogram_genetator_with_thresold(df,thresold):
    plt.figure(figsize=(10, 7))
#     y=800
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.axhline(thresold, color='r', linestyle='--')
    


# In[137]:


dendrogram_genetator_with_thresold(vec_df,800)


# In[136]:


dendrogram_genetator_with_thresold(review_vec_df,1100)


# In[108]:


def hierarchial_clustering(df):
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(df)
    
    plt.figure(figsize=(10, 7))  
    plt.scatter(df['ft1'], df['ft2'], c=cluster.labels_) 
    


# In[109]:


hierarchial_clustering(vec_df)


# In[138]:


hierarchial_clustering(review_vec_df)


# In[110]:


#result.reset_index(drop=True, inplace=True)
vec_df.set_index(keywordsdf["keywords"], inplace = True) 
vec_df


# In[139]:


#result.reset_index(drop=True, inplace=True)
review_vec_df.set_index(review_text_df["Review Text"], inplace = True) 
review_vec_df


# In[111]:


def cluster_element_extraction(vec_df):
    sns.set_palette('Set1', 10, 0.65)
    palette = (sns.color_palette())
    #set_link_color_palette(map(rgb2hex, palette))
    sns.set_style('white')
    
    np.random.seed(25)
    
    link = linkage(vec_df, metric='correlation', method='ward')

    figsize(8, 3)
    den = dendrogram(link, labels=vec_df.index)
    plt.xticks(rotation=90)
    no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
    sns.despine(**no_spine);

    plt.tight_layout()
    plt.savefig('feb2.png');
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    class Clusters(dict):
        def _repr_html_(self):
            html = '<table style="border: 0;">'
            for c in self:
                hx = rgb2hex(colorConverter.to_rgb(c))
                html += '<tr style="border: 0;">'                 '<td style="background-color: {0}; '                            'border: 0;">'                 '<code style="background-color: {0};">'.format(hx)
                html += c + '</code></td>'
                html += '<td style="border: 0"><code>' 
                html += repr(self[c]) + '</code>'
                html += '</td></tr>'

            html += '</table>'

            return html
    
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den['ivl'][i] for i in l]
        cluster_classes[c] = i_l
        
    return cluster_classes
    


# In[112]:


cluster_element_extraction(vec_df)


# In[165]:


cluster_element_extraction(review_vec_df.head(3213))


# In[167]:


cluster_element_extraction(review_vec_df[3215:])


# In[166]:


review_vec_df.head(3214)


# In[113]:


vec_df


# In[114]:


new_df


# In[115]:



listoflist = new_df['Keywords'].tolist()


# In[116]:


def list_of_lists_to_list(list_of_list):
    keyword_list = []
    kywrd_freq_list = []
    for i in list_of_list:
        kywrd_freq_list.append(len(i))
        for j in i:
            keyword_list.append(j)

    return keyword_list, kywrd_freq_list


# In[117]:


keyword_list, kywrd_freq_list = list_of_lists_to_list(listoflist)
print(keyword_list, kywrd_freq_list)


# In[118]:


list_df = pd.DataFrame(keyword_list,columns =['Keywords']) 
list_df


# In[119]:


rating_list = new_df['Star Rating'].tolist()
print(len(rating_list))


# In[120]:


kywrd_rating_list =[]
for i in range(len(rating_list)):
    for j in range(len(kywrd_freq_list)):
        if(i==j):
            kywrd_rating_list.extend(repeat(rating_list[i],kywrd_freq_list[j]))
#print(kywrd_rating_list)
        


# In[121]:


rate_df = pd.DataFrame(kywrd_rating_list,columns =['Star Rating']) 
rate_df


# In[122]:


key_rate_df = list_df.join(rate_df)
key_rate_df


# In[123]:


def clustering_rating_wise(key_rate_df, rating):
    key_data = key_rate_df.loc[key_rate_df['Star Rating'] == rating]
    key_rate_df1 = pd.DataFrame(key_data)
#     return key_rate_df1
    ky_ls =key_rate_df1['Keywords'].tolist()
    
    frame = vectorization_of_list(ky_ls)
    frame.set_index(key_rate_df1["Keywords"], inplace = True) 
    return cluster_element_extraction(frame)


# In[124]:


clustering_rating_wise(key_rate_df, 5)


# In[125]:


key_rate_df["Keywords"].value_counts()[:50]


# In[142]:


eng_df


# In[145]:


required_cols = eng_df[['Review Text','Star Rating']]
required_cols


# In[147]:


def clustering_rating_wise(key_rate_df, rating):
    key_data = key_rate_df.loc[key_rate_df['Star Rating'] == rating]
    key_rate_df1 = pd.DataFrame(key_data)
#     return key_rate_df1
    ky_ls =key_rate_df1['Review Text'].tolist()
    
    frame = vectorization_of_list(ky_ls)
    frame.set_index(key_rate_df1["Review Text"], inplace = True) 
    return cluster_element_extraction(frame)


# In[148]:


clustering_rating_wise(required_cols, 5)


# In[ ]:




