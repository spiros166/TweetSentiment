import csv
# Emoticon Sentiment Lexicon created by Hogenboom et. al.,
# #containing a list of  477 emoticons which are scored either 1 (positive), 0 (neutral) or -1 (negative).
def emoticon_lex(tweets):

    with open('EmoticonLexicon/EmoticonSentimentLexicon.txt', 'r') as f:
        lexicon = []
        for line in csv.reader(f, dialect="excel-tab"):
            lexicon.append(line)
    emoticon_score = {}
    for tweet_id in tweets:
        score = 0
        for word in tweets[tweet_id]:
            for i in lexicon:
                if i[0] == word:
                    score += float(i[1])
        emoticon_score[tweet_id] = score
    return emoticon_score

#Test
#text = [':D',':-(',':)']
#text2 = [':-P']
#tweets = {'1':text,'2':text2}
#print(emoticon_lex(tweets))