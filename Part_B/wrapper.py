import sys, random
import preprocessing_B, ngrams, emoticonlexicon, sentiwordnet
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict


def wrapper(input_file, opinion_file, subjectivity_file, unigramsUse, bigramsUse, store_file):
    reload(sys)  
    sys.setdefaultencoding('utf8')

    tweets={}
    emoticons={}
    hashtags={}
    capitals={}
    longs={}
    sentiment = {}

    
    unigrams_dict = defaultdict(list)
    bigrams_dict = defaultdict(list)

    opinion_l ={}
    emoticon_l ={}
    subj_l = {}
    sentiword_l = {}

    tweets, emoticons, hashtags, capitals, longs, sentiment = preprocessing_B.crawler(input_file)

    unigramsList =[]
    bigramsList = []
    stop = set(stopwords.words('english'))
    for key in sorted(tweets):
        tweetText = tweets[key]
        unigramsVector = ngrams.getUnigramsVector(tweetText,stop)
        unigramsList.extend(unigramsVector)
        bigramsVector = ngrams.getBigramsVector(tweetText,stop)
        bigramsList.extend(bigramsVector)


    if not unigramsUse:
        unigramsUse = ngrams.getUnigramsUse(unigramsList)
        bigramsUse =ngrams.getBigramsUse(bigramsList)

    unigrams_dict = ngrams.getUnigramsFeatures(unigramsUse,tweets)
    bigrams_dict = ngrams.getBigramsFeatures(bigramsUse,tweets,stop)


    emoticon_l=emoticonlexicon.emoticon_lex(emoticons)

    sentiword_l=sentiwordnet.sentiwordnet_lex(tweets)
    #subj_l=subjectivity.subjectivity_lex(tweets)
    #opinion_l = opinion.opinion_lex(tweets)

    for i in sorted(longs):
            if len(longs[i])>0:
                longs[i]=1
            else:
                longs[i]=0


    for i in sorted(capitals):
            if len(capitals[i])>1:
                capitals[i]=1
            else:
                capitals[i]=0

    for i in sorted(emoticon_l):
        if emoticon_l[i]==-1:
            emoticon_l[i]=0
        elif emoticon_l[i]==1 or emoticon_l[i]==2:
            emoticon_l[i]=1
        else:
            emoticon_l[i]=0.5

    for i in sorted(sentiword_l):
        sentiword_list=sentiword_l[i]
        for j in range(0,2):
            if sentiword_list[j]>2:
                sentiword_list[j]=2
            sentiword_list[j]=sentiword_list[j]/2
        sentiword_l[i]=sentiword_list


    for i in sentiment:
        if sentiment[i]=='positive':
            sentiment[i]=1
        elif sentiment[i]=='neutral':
            sentiment[i]=0
        elif sentiment[i]=='negative':
            sentiment[i]=-1
        else:
            sentiment[i]=3


    opinions={}
    with open(opinion_file) as opinion:
        for line in opinion:
            opinion_line=line.split(',')
            opinion_line[1]=float(opinion_line[1].strip())
            if opinion_line[1]>5:
                opinion_line[1]=5
            elif opinion_line[1]<-5:
                opinion_line[1]=-5
            opinion_line[1]=(opinion_line[1]/5)+0.5


            opinions[opinion_line[0]]=opinion_line[1]
    

    subjectivities={}
    with open(subjectivity_file) as subjectivity:
        for line in subjectivity:
            subj_list=[]
            subjectivity_line=line.split(',')
            subj_list.append(float(subjectivity_line[1].strip()))
            if subj_list[0]>3:
                subj_list[0]=3
            subj_list[0]=subj_list[0]/3

            subj_list.append(float(subjectivity_line[2].strip()))
            if subj_list[1]>3:
                subj_list[1]=3
            subj_list[1]=subj_list[1]/3

            subjectivities[subjectivity_line[0]]=subj_list



    unigram_value=random.choice(unigrams_dict.values())
    unigram_csv=''
    bigram_value=random.choice(bigrams_dict.values())
    bigram_csv=''


    for j in range(0,len(unigram_value)):
        unigram_csv+=',uni'+str(j)
    for j in range(0,len(bigram_value)):
        bigram_csv+=',bi'+str(j)
    train_data_contents = "id, sentiment, sentiwordnet_positive, sentiwordnet_negative,subjectivity_positive, subjectivity_negative, opinions, emoticons, capitals, longs "+unigram_csv+bigram_csv
    for key in sorted(sentiment):
        unigram_csv=''
        bigram_csv=''
        for j in range(0,len(unigrams_dict[key])):
            unigram_csv+=','+str(unigrams_dict[key][j])
        for j in range(0,len(bigrams_dict[key])):
            bigram_csv+=','+str(bigrams_dict[key][j])

        train_data_contents += '\n'+str(key)+','+str(sentiment[key])+','+str(sentiword_l[key][0])+','+str(sentiword_l[key][1])+','+str(subjectivities[key][0])+','+str(subjectivities[key][1])+','+str(opinions[key])+','+str(emoticon_l[key])+','+str(capitals[key])+','+str(longs[key])+unigram_csv+bigram_csv
    with open(store_file, 'w') as output:
            output.write(train_data_contents)

    return unigramsUse, bigramsUse
#MAIN
#store the unigrams and bigrams which where extracted on the training set, so we can use the same on the test set.
unigramsUsed={}
bigramsUsed={}

#Run
print("##################################################################################################")
print("# Wrapping features for train dataset...")
unigrams, bigrams=wrapper("Data/twitter-train-cleansed-B.tsv", "Data/opinion/opinion_trainB.txt","Data/subjectivity/subjectivity_trainB.txt",unigramsUsed, bigramsUsed, "Data/train_data_B.csv")
print("# Done!\n#")

print("# Wrapping features for development dataset...")
wrapper("Data/twitter-dev-gold-B.tsv","Data/opinion/opinion_devB.txt","Data/subjectivity/subjectivity_devB.txt", unigrams, bigrams,"Data/development_data_B.csv")
print("# Done!\n#")

print("# Wrapping features for development dataset...")
wrapper("Data/twitter-test-B.tsv","Data/opinion/opinion_testB.txt","Data/subjectivity/subjectivity_testB.txt", unigrams, bigrams,"Data/test_data_B.csv")
print("# Done!\n#")
print("# Wrapping just finished! Go to Data folder to find txt files with the extracted features!")
print("###################################################################################################")






