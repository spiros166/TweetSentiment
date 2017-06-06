from nltk.corpus import opinion_lexicon
import csv
from nltk.corpus import stopwords
import preprocessing_A

def opinion_lex(tweets, store_file):

    #get the file with all the negators
    with open('NegatorsLexicon/Negators.txt', 'r') as f:
        negators = f.readlines()
    #remove whitespace characters like `\n` at the end of each line
    negators = [x.strip() for x in negators]

    #import negative words and positive words from opinion lexicon
    negative_words=opinion_lexicon.negative()
    positive_words=opinion_lexicon.positive()

    #dictionary that will store tweets_id as keys and polarity score as value
    opinion = {}

    #stopset is a list of stopwords of english language
    stopset = set(stopwords.words('english'))

    #open a file where opinion dict will be stored to be used in the classifiers
    with open(store_file, 'w') as out:
        #iterate through tweets
        for tweet_id in tweets:
            score=0 #polarity score of tweet
            negation=1 
            #iterate through words in tweet
            for word in tweets[tweet_id]:
                #if the word is in the stopset we dont surch its sentiment
                if word not in stopset:
                    if word in negative_words:
                        score -= 1 #reduce score by one if it is a negative word
                    elif word in positive_words:
                        score += 1 #increase score by one if it is a positive word
                if word in negators: 
                    #if there was a negator (ex. not, nothing, aren't etc), we change negation to -1
                    negation=-1
            #the sentiment score is the multiplication of negation with score
            opinion[tweet_id]=negation*score
            
            #writing in a file    
            line=str(tweet_id)+','+str(opinion[tweet_id])+'\n'
            out.write(line)

    return opinion

print("##############################################################################")
print("# Running the opinion lexicon for every dataset.\n# The outpout is 3 txt files which will be stored in the Data/opinion folder.\n# This may take a while...\n#")
tweets_trainA, emoticons, hashtags, capitals, longs, sentiment=preprocessing_A.crawler("Data/twitter-train-cleansed-A.tsv")
tweets_devA, emoticons, hashtags, capitals, longs, sentiment=preprocessing_A.crawler("Data/twitter-dev-gold-A.tsv")
tweets_testA, emoticons, hashtags, capitals, longs, sentiment=preprocessing_A.crawler("Data/twitter-test-A.tsv")

opinion_lex(tweets_trainA,'Data/opinion/opinion_trainA.txt')
opinion_lex(tweets_devA,'Data/opinion/opinion_devA.txt')
opinion_lex(tweets_testA, 'Data/opinion/opinion_testA.txt')

print("# Success!")
print("##############################################################################")





