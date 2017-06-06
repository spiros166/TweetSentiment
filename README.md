# TweetSentiment
A text classifier to tackle sentiment classification in Twitter for task 10 of Semeval 2015 using Python and the Sklearn library. 
http://alt.qcri.org/semeval2015/task10/

It perfroms sentiment analysis for two different tasks:
  
  Task A: Contextual Polarity Disambiguation: Given a message containing a marked instance of a word or phrase, determine whether that instance is positive, negative or neutral in that context.
  
  Task B: Message Polarity Classification: Given a message, classify whether the message is of positive, negative, or neutral sentiment. For messages conveying both a positive and negative sentiment, whichever is the stronger sentiment should be chosen.
  
  
  *** Run classifiers for your own data simply by changing the data folder content.
  
  
  
Guidelines to run the code

a. Run the opinion.py and subjectivity.py which will create the lexicons. Since, it is time consuming we run it separately and store the results and you can access this data in the Data folder.

b. Run the wrapper.py file, which imports preprocessing, twokenize, ngrams, SentiWordNet and emoticonlexicon. This also takes the output of the files run in the 1st step. The output is a csv file which has all the tweets ID’s, sentiment and features.

c. Run the classifier.py file for SVM and classifier_NB_RF.py for Naïve Bayes and Random forest. The output of this is a text file with the tweet ID’s and predicted sentiment of the test data. You can access these files from the Data folder.

d. Run the Svm_crossValidation.py to run the SVM classifier with 5 fold cross validation

