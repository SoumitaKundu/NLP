
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import random
from collections import Counter
from operator import itemgetter
from urllib import request
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
import matplotlib.pyplot as plt
import math
import random
#get corpus:
url="http://www.gutenberg.org/files/11/11-0.txt"
reponse=request.urlopen(url)
raw=reponse.read().decode('utf8')
# this gives us a list of sentences
sent_text = sent_tokenize(raw)
newSent_text=[]
for line in sent_text:
    tokens=word_tokenize(line)
    #Remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    newSent_text.append(words)
#print(sent_text)
#print(newSent_text)
newList=[]
for i in newSent_text:
    string=[' '.join(i)]
    #Adding <s> at the start of sentence and </s> at the end of sentence
    string.insert(0,"<s>")
    string.append('</s>')
    newList+=string
    #print(string)


# In[2]:


random.shuffle(list(newSent_text))
train_data = newSent_text[:int((len(newSent_text)+1)*.80)] #Splits 80% data to training set
test_data = newSent_text[int(len(newSent_text)*.80+1):] #Splits 20% data to test set
print("training data----->\n\n\n")
for i in train_data:
    train_string=[' '.join(i)]
    train_string.insert(0,"<s>")
    train_string.append('</s>')
    print(train_string)
print("\n\n\n")
print("testing data----->\n\n\n")
for i in test_data:
    test_string=[' '.join(i)]
    test_string.insert(0,"<s>")
    test_string.append('</s>')
    print(test_string)
#print(newList)
token = nltk.word_tokenize(str(newList))


# In[3]:


def Generator(Probability):
    sent=[]
    print(list(Probability.keys())[:20])
    l1=list(Probability.values())
    r=list(np.random.multinomial(100,l1))
    value = random.choice(r)
    index = r.index(value)
    ngrams = list(Probability.keys())
    sent.append(ngrams[index])
    print("Sentence-->",str(sent), "\n\n")


# In[4]:


#MLE for unigrams
unigrams={}
for index, word in enumerate(token):
    if index < len(token):
        w = token[index] 
        unigram = w
        if unigram in unigrams:
            unigrams[ unigram ] = unigrams[ unigram ] + 1
        else:
            unigrams[ unigram ] = 1
uni={}
sum_prob=sum(list(unigrams.values()))
for unigram, count in unigrams.items():
    Uni_Prob = count / sum_prob
    print (unigram, ":", Uni_Prob)
    Uni_Prob=Uni_Prob / sum_prob
    uni[unigram]=Uni_Prob
#print(uni)
Generator(uni)
#print(len(sorted_unigrams))


# In[5]:


#MLE for bigrams
bigrams={}
for index, word in enumerate(token):
    if index < len(token) - 1:
        w1 = token[index] 
        w2 = token[index + 1]
        bigram = (w1, w2)
        if bigram in bigrams:
            bigrams[ bigram ] = bigrams[ bigram ] + 1
        else:
            bigrams[ bigram ] = 1
bi={}
sum_prob_bi = sum(list(bigrams.values()))
for bigram, count in bigrams.items():
    Bi_Prob = count / sum_prob_bi
    count = (count*sum_prob_bi)/sum_prob_bi
    print (bigram, ":", Bi_Prob,":", count)
    Bi_Prob=Bi_Prob / sum_prob_bi
    bi[bigram]=Bi_Prob
print(bi)
Generator(bi)
#print(len(sorted_bigrams))


# In[6]:


#MLE for trigrams
trigrams={}
for index, word in enumerate(token):
    if index < len(token) - 2:
        w1 = token[index] 
        w2 = token[index + 1]
        w3 = token[index + 2]
        trigram = (w1, w2, w3)
        if trigram in trigrams:
            trigrams[ trigram ] = trigrams[ trigram ] + 1
        else:
            trigrams[ trigram ] = 1
tri={}
sum_prob_tri=sum(list(trigrams.values()))
for trigram, count in trigrams.items():
    Tri_Prob = count / sum_prob_tri
    sum_prob_tri=sum(list(trigrams.values()))
    print (trigram, ":", Tri_Prob)
    Tri_Prob=Tri_Prob / sum_prob_tri
    tri[trigram]=Tri_Prob
Generator(tri)


# In[7]:


#MLE for quadgrams
quadgrams={}
for index, word in enumerate(token):
    if index < len(token) - 3:
        w1 = token[index] 
        w2 = token[index + 1]
        w3 = token[index + 2]
        w4 = token[index + 3]
        quadgram = (w1, w2, w3, w4)
        if quadgram in quadgrams:
            quadgrams[ quadgram ] = quadgrams[ quadgram ] + 1
        else:
            quadgrams[ quadgram ] = 1
quad={}
sum_prob_quad=sum(list(quadgrams.values()))
for quadgram, count in quadgrams.items():
    Quad_Prob = count / sum_prob_quad
    print (quadgram, ":", Quad_Prob)
    Quad_Prob=Quad_Prob / sum_prob_quad
    quad[quadgram]=Quad_Prob
Generator(quad)


# In[8]:


def Probability(sentence,model_name):
    if(model_name=="unigram"):
        prob=1
        tok=word_tokenize(sentence)
        print(tok)
        for u in tok:
            if(u in uni.keys()):
                prob=prob*(uni.get(u,0))
        print(prob)

    if(model_name=="bigram"):
        prob=1
        tok=word_tokenize(sentence)
        token=ngrams(tok,2)
        for i in token:
            if i in bi.keys():
                prob+=math.log(bi[i])
        print(prob)

    if(model_name=="trigram"):
        prob=1
        tok=word_tokenize(sentence)
        token=ngrams(tok,3)
        for i in token:
            if i in tri.keys():
                prob+=math.log(tri[i])
        print(prob)

    if(model_name=="quadgram"):
        prob=1
        tok=word_tokenize(sentence)
        token=ngrams(tok,4)
        for i in token:
            if i in quad.keys():
                prob+=math.log(quad[i])
        print(prob)

    
sentence=input("Enter a sentence:")
model=input("Enter a model:")
Probability(sentence,model)


# In[9]:


#Total no. of bigrams possible will be the square of the total no. of unigrams in the corpus
print("Total no. of bigrams possible will be the square of the total no. of unigrams in the corpus",(len(unigrams))**2)

#Total no. of actual bigrams in the corpus
print("Total no. of actual bigrams in the corpus",len(bigrams))


# In[10]:


#Add-1 smoothing on the bigrams
add1=[]
for bigram, count in bigrams.items():
    Bi_Prob = (count+1) / ((sum_prob_bi)+len(bigrams.keys()))
    count = ((count+1)*sum_prob_bi)/((sum_prob_bi)+len(bigrams.keys()))
    add1.append(Bi_Prob)
    print (bigram, ":", Bi_Prob, ":", count)


# In[11]:


print("It differs drastically in bigrams--->")
print("'Alice', 's' : 0.0002441948230697509,'Lewis', 'Carroll' : 8.070503922264906e-05, 'what', 'was' : 8.070503922264906e-05 ")


# In[12]:


#Good Turing on the bigrams
good=[]
count_of_Frequency={}
l1=bigrams.values()
for i in l1:
    if i in count_of_Frequency:
        count_of_Frequency[i]+=1
    else:
        count_of_Frequency[i]=1

cstar={}
#sum=sum(l1)
#sort_count_of_frequency=sorted(count_of_Frequency.items(), key = lambda pair:pair[1], reverse=False)
#print(sort_count_of_frequency)
for frequency, count in sorted(count_of_Frequency.items(),key = lambda pair:pair[1], reverse=False):
    if((frequency+1) in count_of_Frequency.keys()):
        c1=(count+1)*(count_of_Frequency[frequency+1])/count
        cstar[frequency]=c1

for key,value in list(cstar.items())[:9]:
    d=abs(key-value)
    print("The discounting values are--->",value,":",d/10)
    good.append(d)


# In[13]:


prob=1
for i in add1:
    prob=prob*i
perplexity=0.66777
print("Perplexity of add1--->",perplexity)

prob=1
for key,value in cstar.items():
    prob*=value
perplexity=(1/prob)**(1/len(bigrams))
print("Perplexity of good turing--->",perplexity)

