#Import csv library to read the .tsv file

import twokenize, csv, re, json

#Uncomment if you want to use these libraries to check english words
#from nltk.corpus import words
#import enchant

#replace URLs with "URLLINK"
def replaceURLs(tweet):
    for i, token in enumerate(tweet):
        tweet[i] = re.sub(r"http\S+|www\S+", "$URLLINK$", token)
    return tweet

#replace user mentions with "USERMENTION"
def replaceUserMentions(tweet):
    for i, token in enumerate(tweet):
        tweet[i] = re.sub("(@[A-Za-z0-9_]+)", "$USERMENTION$", token)
    return tweet

#replace hashtags with "HASHTAGS" 
def replaceHashtags(tweet):
    for i, token in enumerate(tweet):
        tweet[i] = re.sub("(#[A-Za-z0-9_]+)", "$HASHTAG$", token)
    return tweet

#replace emoticons with "EMOTICONS" 
def replaceEmoticons(tweet):
    emoticon_string=r"[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?"
    for i, token in enumerate(tweet):
        if re.match(emoticon_string, token):
            tweet[i]="$EMOTICON$"
    return tweet
    
def replaceCapitals(tweet):
    for i, token in enumerate(tweet):
        tweet[i] = str.lower(token)
    return tweet

def replaceUnicode(tweet):
    for i, token in enumerate(tweet):
        tweet[i] = token.encode('utf-8')
    return tweet

#replace all non-alphanumeric
def replaceRest(tweet):
    result = re.sub("[^a-zA-Z0-9]", " ", tweet)
    return re.sub(' +',' ', result)




#store emoticons in a list
def storeEmoticons(tweet):
    emoticon_string=r"[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?"
    emoticons=[]
    for i, token in enumerate(tweet):
        if re.match(emoticon_string, token):
            emoticons.append(token)
    return emoticons

#store hashtags in a list
def storeHashtags(tweet):
    hashtag_string=r"#[A-Za-z0-9_]+"
    hashtags=[]
    for i, token in enumerate(tweet):
        if re.match(hashtag_string, token):
            hashtags.append(token)
    return hashtags

def storeCapitals(tweet):
    capital_string=r"^[A-Z][A-Z][A-Z]+$"
    exclude_string=r"^[AEIOU][AEIOU][AEIOU]|^[BCDFGHJKLMNPQRSTVWXZ][BCDFGHJKLMNPQRSTVWXZ][BCDFGHJKLMNPQRSTVWXZ]"
    capitals=[]
    for i, token in enumerate(tweet):
        if re.match(capital_string, token):
            if not re.match(exclude_string,token):
                
                #if diction.check(token):
                #Uncomment if you want to use english dictionaries to test the words... too slow
                #if token in words.words():

                capitals.append(token)
                    
    return capitals

def storeLongWords(tweet):
    long_string=r".*([a-z])\1\1.*"
    longs=[]
    for i, token in enumerate(tweet):
        if re.match(long_string, token):
            if token!='iii':
                #print(token)
                longs.append(token)
    return longs


def isolatePhrase(tweet,start,end):

    new_text=[]
    if start>=2: 
        if end<len(tweet)-2:
            new_text=tweet[(start-2):(end+3)]
        elif end<len(tweet)-1:
            new_text=tweet[(start-2):(end+2)]
        else:
            new_text=tweet[(start-2):end+1]
    elif start==1: 
        if end<len(tweet)-2:
            new_text=tweet[(start-1):end+3] 
        elif end<len(tweet)-1:
            
            new_text=tweet[(start-1):end+2]
            
        else:
            new_text=tweet[(start-1):end+1]
    elif start==0:
        if end<len(tweet)-2:
            new_text=tweet[start:end+3]
        elif end<len(tweet)-1:
            new_text=tweet[start:end+2]
        else:
            new_text=tweet[start:end+1]

    return " ".join(new_text)


#Crawler will read the tsv file
#Outputs: 
#           1) a dictionary with key="tweet_id" and value="tokenized processed tweet_text"
#           2) a dictionary with key="tweet_id" and value="emoticons"
#           3) a dictionary with key="tweet_id" and value="hashtags"
#           4) a dictionary with key="tweet_id" and value="uppercase_words"

def crawler(file):

    #Dictionary initilization.
    tweets = {}
    emoticons = {}
    hashtags = {}
    capitals = {}
    longs = {}
    sentiment = {}

    #Read the file and store it in tweets dictionary
    with open(file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        #diction = enchant.request_dict("en_US")
        for line in tsvreader:

            tweet_id = str(line[1])+'$'+str(line[2])+'$'+str(line[3])
            splitted_text=str(line[5]).split(" ")

            start = int(line[2])
            end= int(line[3])
            isolated_phrase= isolatePhrase(splitted_text,start,end)

            tokenized_text = twokenize.tokenizeRawTweetText(isolated_phrase)
            tokenized_text = replaceUnicode(tokenized_text)

            capitals[tweet_id] = storeCapitals(tokenized_text)
            emoticons[tweet_id] = storeEmoticons(tokenized_text)
            hashtags[tweet_id] = storeHashtags(tokenized_text)

            new_text =replaceCapitals(tokenized_text)
            new_text = replaceURLs(tokenized_text)
            new_text = replaceUserMentions(new_text)
            new_text = replaceEmoticons(new_text)           
            new_text = replaceHashtags(new_text)

            longs[tweet_id] = storeLongWords(new_text)
            sentiment[tweet_id] = str(line[4])
            #print(sentiment)
            tweets[tweet_id] = new_text
    tsvfile.close()


    return (tweets, emoticons, hashtags, capitals,longs, sentiment)


#Call crawler and store Outputs
#Tweets will be used for lexicons and ngramms
#tweets, emoticons, hashtags, capitals, longs, sentiment=crawler("Data/twitter-train-cleansed-A.tsv")

#import subjectivity
#subjectivity.subjectivity_lex(tweets)

#TESTING
#for key in sorted(hashtags)[50:100]:
    #print(key,tweets[key],hashtags[key],emoticons[key],capitals[key])
#for key in sorted(emoticons)[50:1000]:
    #print(key,emoticons[key])
#for key in sorted(tweets):
    #print(key, tweets[key])
#for key in sorted(capitals)[50:100]:
    #print(capitals[key])
#for key in sorted(longs):
    #print(key,longs[key])

#for key in sorted(sentiment):
    #print(key, sentiment[key])


#print(tweets.get('812957996'))
#print(tweets.get('33442620'))
#print(tweets.get('369152026'))



