import pandas as pd
import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

# NaiveBayes and RandomForest CLASSIFIER FUNCTION
# This method trains the NaiveBayes and RandomForest classifiers. 
# It takes as arguments the filename of the file with the training features(train_data)
#, the filename of the file with the test features(test_data)
def NB_RF_classifier(train_data,test_data):
	#-------------------------------
	#------ READ TRAIN DATA --------
	#-------------------------------
	
	#Read data file with tweets, their sentiment and their features for train_set
	train_dataframe = pd.read_csv(train_data)

	#Create np array with train labels (positive, neutral , negative) size= 1 x number of tweets
	train_labels = train_dataframe.iloc[:,1]
	train_labels = np.array(train_labels)
	
	#Create np array with normalized train features (lexicons, ngrams, bigrams etc) 
	#size= number of tweets x number of features
	train_features = train_dataframe.iloc[:,2:]
	train_features = np.array(train_features)

	#print labels and features from training set
	print "train labels: "
	print train_labels
	print 
	print "train features:"
	print train_features

	

	#-------------------------------
	#--- TRAINING CLASSIFIERS ------
	#-------------------------------
	#1. Naive Bayes
	print('NB is training now....')
	#TRAIN CLASSIFIER
	cl_NB = GaussianNB().fit(train_features, train_labels)
	print(cl_NB)
	
	#2. Random Forrest
	print('RandomForest is training now....')
	#TRAIN CLASSIFIER
	cl_RF = RandomForestClassifier(n_estimators=100)
	cl_RF.fit(train_features, train_labels)
	print(cl_RF)
	

	#-------------------------------
	#------ TEST SET: FEATURES -----
	#-------------------------------
	#Read data file with tweets, their sentiment and their features for test set
	test_dataframe = pd.read_csv(test_data)

	tweet_id =test_dataframe.iloc[:,0]
	tweet_id = np.array(tweet_id)
	# array with test labels (positive, neutral , negative) size= 1 x number of tweets
	test_labels = test_dataframe.iloc[:,1]
	test_labels = np.array(test_labels)
	#np array with normalized test features (lexicons, ngrams, bigrams etc) size= number of tweets x number of features
	#test_features = test_dataframe.iloc[:,2:]
	test_features = test_dataframe.iloc[:,2:]
	test_features = np.array(test_features)


	#-------------------------------
	#-- PREDICTION OF TEST LABELS --
	#-------------------------------
	#1. NaiveBayes
	#print('Naive Bayes is predicting the test set now....')
	#Run NB classifier
	resultsNB = cl_NB.predict(test_features)
	
	#Print results of NB 
	print('------------------------')
	print('NB results:')
	print('------------------------')
	print('Predicted Classes:')
	print(resultsNB)

	#Here we print the accuracy and f1 score of the results
	#this is only going to give a score if the labels are not uknown
	print(precision_recall_fscore_support(test_labels,resultsNB,average="macro"))
	pos=f1_score(test_labels,resultsNB,average="macro",labels=[1])
	neg=f1_score(test_labels,resultsNB,average="macro",labels=[-1])
	num_correct = (resultsNB == test_labels).sum()
	recall = float(num_correct) / len(test_labels)
	print "model accuracy (%): ", recall * 100, "%"
	print('f1 average score',(neg+pos)/2.0) 
	print('------------------------')

	#2. RandomForest
	print('Random Forest is predicting the test set now....')
	#Run RF classifier
	resultsRF = cl_RF.predict(test_features)
	
	#Print results of RF
	print('------------------------')
	print('RF results:')
	print('------------------------')
	print('Predicted Classes:')
	print(resultsRF)

	#Here we print the accuracy and f1 score of the results
	#this is only going to give a score if the labels are not uknown
	print(precision_recall_fscore_support(test_labels,resultsRF,average="macro"))
	pos=f1_score(test_labels,resultsRF,average="macro",labels=[1])
	neg=f1_score(test_labels,resultsRF,average="macro",labels=[-1])
	num_correct = (resultsRF == test_labels).sum()
	recall = float(num_correct) / len(test_labels)
	print "model accuracy (%): ", recall * 100, "%"
	print('f1 average score',(neg+pos)/2.0) 
	print('------------------------')





#Calling NB and RF_classifier for the development sets
NB_RF_classifier('Data/train_data_B.csv','Data/development_data_B.csv')