import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import preprocessing_B

def subjectivity_lex(tweets,store_file):  
    #open subjectivity lexicon and insert it in a list of dicts, where each dict is a word
    source = 'subjectivityLexicon/subjclueslen1-HLTEMNLP05.tff'
    with open(source, 'r') as fo:
        sub_clues = [c.strip().replace('type=', 'rel=').split()
                     for c in fo]
    sub_clues = [dict([d.split('=') for d in sc if len(d.split('=')) == 2])
                 for sc in sub_clues]

    #get the file with all the negators
    with open('NegatorsLexicon/Negators.txt', 'r') as f:
        negators = f.readlines()
    #remove whitespace characters like `\n` at the end of each line
    negators = [x.strip() for x in negators]

    stopset = set(stopwords.words('english'))#set of stopwords
    pos_tweets={} #dict with tweets with POS tags
    #create new dict tweet_id as keys and values tuples (word,POS tag)
    for id in tweets:
        pos_tweets[id]=nltk.pos_tag(tweets[id])

    st = LancasterStemmer()#stemmer

    subjectivity={}#dict with the sentiment of each tweed, key=tweet_id and value=[positive_score,negative_score]

    #open file to write results
    with open(store_file, 'w') as out:
        #iterate through pos_tweets
        for tweet_id in pos_tweets:
            pos_score = 0 #positive score starts from 0
            neg_score = 0 #negative score starts from 0
            negation=False #if the tweet has a negator, this turns True
            
            #iterate through every word of each tweet
            for word in pos_tweets[tweet_id]:
                #if the word is in stopset continue to next word
                if word[0] not in stopset:
                    #change POS to correct input for lexicon
                    if re.match("^VB",word[1]):
                        pos = 'verb'
                        flag=True
                    elif re.match("^NN|^PR",word[1]):
                        pos = 'noun'
                    elif re.match("^RB",word[1]):
                        pos = 'adv'
                    elif re.match("^JJ",word[1]):
                        pos = 'adj'
                    else:
                        pos = 'anypos'
                    #iterate through clues of the lexicon
                    for sc in sub_clues:
                        #if the clue is not a stem, and is the correct POS and word=clue
                        if sc['stemmed1'] == 'n' and pos == sc['pos1'] and word[0]==sc['word1']:
                            #add 1 to positive for strong polarity
                            #add 0.5 to positive for weak polarity
                            # the same for negative
                            if sc['priorpolarity']== 'positive' and sc['rel']=='strongsubj':
                                pos_score+=1
                            elif sc['priorpolarity']== 'positive' and sc['rel']=='weaksubj':
                                pos_score+=0.5
                            elif sc['priorpolarity']== 'negative' and sc['rel']=='strongsubj':
                                neg_score+=1
                            elif sc['priorpolarity']== 'negative' and sc['rel']=='weaksubj':
                                neg_score+=0.5
                            break
                        #else if it is a stem and is the same POS match the stem with the word
                        elif sc['stemmed1'] == 'y' and pos == sc['pos1'] and re.match(st.stem(sc['word1']),word[0]):
                            if sc['priorpolarity'] == 'positive' and sc['rel'] == 'strongsubj':
                                pos_score += 1
                            elif sc['priorpolarity'] == 'positive' and sc['rel'] == 'weaksubj':
                                pos_score += 0.5
                            elif sc['priorpolarity'] == 'negative' and sc['rel'] == 'strongsubj':
                                neg_score += 1
                            elif sc['priorpolarity'] == 'negative' and sc['rel'] == 'weaksubj':
                                neg_score += 0.5
                            break
                #change negation to True if there is a negator word in the tweet
                if word[0] in negators:
                    negation=True
            #reverse the polarity if negation is true
            if negation:
                subjectivity[tweet_id] = [neg_score,pos_score]
            else:
                subjectivity[tweet_id] = [pos_score,neg_score]
            #print(subjectivity[tweet_id])
            #writing in a file    
            line=str(tweet_id)+','+str(subjectivity[tweet_id][0])+','+str(subjectivity[tweet_id][1])+'\n'
            out.write(line)
        
    return subjectivity

#
#MAIN
#

print("##############################################################################")
print("# Running the subjectivity lexicon for every dataset.\n# The outpout is 3 txt files which will be stored in the Data/subjectivity folder.\n# This may take a while....\n#")
tweets_trainB, emoticons, hashtags, capitals, longs, sentiment=preprocessing_B.crawler("Data/twitter-train-cleansed-B.tsv")
tweets_devB, emoticons, hashtags, capitals, longs, sentiment=preprocessing_B.crawler("Data/twitter-dev-gold-B.tsv")
tweets_testB, emoticons, hashtags, capitals, longs, sentiment=preprocessing_B.crawler("Data/twitter-test-B.tsv")

subjectivity_lex(tweets_trainB,'Data/subjectivity/subjectivity_trainB.txt')
subjectivity_lex(tweets_devB,'Data/subjectivity/subjectivity_devB.txt')
subjectivity_lex(tweets_testB, 'Data/subjectivity/subjectivity_testB.txt')

print("# Success!")
print("##############################################################################")


