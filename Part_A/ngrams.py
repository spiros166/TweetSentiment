#Import the preprocessing and twokenize files.
import nltk
import re
from nltk.corpus import stopwords
from collections import defaultdict



#Get the features - unigrams
def getUnigramsVector(tweetText,stop):

    unigramsVector = []
    for w in tweetText:
    	#check if the word stats with an alphabet
    	val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
    	#ignore if it is a stop word
    	if(w in stop or val is None):
    		continue
    	else:
    		unigramsVector.append(w.lower())

    return unigramsVector

#Function to display the most common unigram features
def getUnigramsUse(featureList):


    Unigrams = {}
    UnigramsUse = {}

    #Counting the number of times the feature appears
    for item in featureList:
        try:
            Unigrams[item]+=1
        except:
            Unigrams[item]=1

    #Use only those unigrams which have count > 7
    for k,v in Unigrams.iteritems():
        if v >= 5:
            UnigramsUse[k]= v

    return UnigramsUse

def getUnigramsFeatures(UnigramsUse,tweets):


    unigrams_dict = defaultdict(list)
  
    #Check the tweet text for the UnigramsUse, if it exists the value is one else 0
    for key in sorted(tweets):
        text = tweets[key]
        text = set(text)
        for keys,values in UnigramsUse.iteritems():
            if keys in text:
                unigrams_dict[key].append(1)
            else:
                unigrams_dict[key].append(0)

    return unigrams_dict

#Get the features Bigrams
def getBigramsVector(tweetText,stop):

    bigramsVector = []
    my_bigrams = nltk.bigrams(tweetText)
    for item in my_bigrams:

        val1 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", item[0])
        val2 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", item[1])

        if(item[0] in stop or val1 is None ) or  (item[1] in stop or val2 is None):
            continue
        else:
            item[0].lower()       
            item[1].lower()
            bigramsVector.append(item)

    return bigramsVector

#Function to display the most common bigram features
def getBigramsUse(bigramsList):

    Bigrams ={}
    bigramsUse = {} 

    for item in bigramsList:
        try:
            Bigrams[item]+=1
        except:
            Bigrams[item]=1

    #print Bigrams

    for k,v in Bigrams.iteritems():
        if v >= 3:
            bigramsUse[k]= v

    return bigramsUse

def getBigramsFeatures(bigramsUse,tweets,stop):

    bigrams_dict = defaultdict(list)

    for key in sorted(tweets):
        text = tweets[key]
        bigramTweet = getBigramsVector(text,stop)
        for keys,values in bigramsUse.iteritems():
            if keys in bigramTweet:           
                bigrams_dict[key].append(1)
            else:
                bigrams_dict[key].append(0)

    return bigrams_dict


    