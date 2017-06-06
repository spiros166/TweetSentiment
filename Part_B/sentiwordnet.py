import nltk
import re
from nltk.corpus import sentiwordnet as swn


def sentiwordnet_lex(tweets):
    pos_tweets = {}
    for id in tweets:
        pos_tweets[id]=nltk.pos_tag(tweets[id])

    sentiwordnet={}
    for tweet_id in pos_tweets:
        pos_score = 0
        neg_score = 0
        for word in pos_tweets[tweet_id]:
            flag = False
            pos=''
            if re.match("^VB",word[1]):
                pos = 'v'
                flag=True
            elif re.match("^NN|^PR",word[1]):
                pos = 'n'
                flag = True
            elif re.match("^RB",word[1]):
                pos = 'r'
                flag = True
            elif re.match("^JJ",word[1]):
                pos = 'a'
                flag = True
            list_synsets=list(swn.senti_synsets(word[0],pos))
            if flag and len(list_synsets) != 0:
                syn=word[0]+'.'+pos+'.01'
                #print(syn)
                #print(swn.senti_synset(syn))
                synset=list_synsets[0]
                pos_score+= synset.pos_score()
                neg_score+= synset.neg_score()
        sentiwordnet[tweet_id]=[pos_score,neg_score]
    return sentiwordnet

#TESTING
#print(list(swn.senti_synsets('starting','v')))
#print(list(swn.senti_synsets('start','v')))
#print(list(swn.senti_synsets('love','v')))
#list1=list(swn.senti_synsets('loving','v'))
#print(list1[1])

#text = ['I','do','not','love','pizza']
#text2 = ['starting','hate','pizza']
#tweets = {'1':text,'2':text2}
#print(sentiwordnet_lex(tweets))
