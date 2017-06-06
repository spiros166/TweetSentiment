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

# SVM CLASSIFIER FUNCTION
# This method trains the SVM classifier with C=1000, gamma=0.1 and runs on the test set. 
# It takes as arguments the filename of the file with the training features(train_data)
#, the filename of the file with the test features(test_data), and the path of the 
# file that the results are going to be printed in (results_filename)
# In the results_filename it writes the tweet id and the predicted class
def svm_classifier(train_data,test_data,results_filename):
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
	print('SVM is training now....')
	#TRAIN CLASSIFIER
	#cl_SVM = svm.SVC(kernel="rbf",C=100,gamma=0.1,cache_size=4000)
	cl_SVM = svm.SVC(kernel="rbf",C=1000,gamma=0.1,cache_size=3000)


	#FIT CLASSIFIER
	cl_SVM.fit(train_features, train_labels)
	print('The parameters of the SVM CLASSIFIER:')
	print(cl_SVM)


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
	print('SVM is predicting the test set now....')
	#Run SVM classifier
	results = cl_SVM.predict(test_features)
	
	#Print results of SVM 
	print('------------------------')
	print('SVM results:')
	print('------------------------')
	print('Predicted Classes:')
	print(results)

	#Here we print the accuracy and f1 score of the results
	#this is only going to give a score if the labels are not uknown
	print(precision_recall_fscore_support(test_labels,results,average="macro"))
	pos=f1_score(test_labels,results,average="macro",labels=[1])
	neg=f1_score(test_labels,results,average="macro",labels=[-1])
	num_correct = (results == test_labels).sum()
	recall = float(num_correct) / len(test_labels)
	print "model accuracy (%): ", recall * 100, "%"
	print('f1 average score',(neg+pos)/2.0) 
	print('------------------------')


	#-------------------------------
	#-- WRITING RESULTS TO A FILE --
	#-------------------------------

	#Write results in the results_filename
	with open(results_filename,'w') as f:
	    for i in range(0,len(results)):
	    	sentiment=''
	    	if results[i]== 1:
	    		sentiment='positive'
	    	elif results[i]== -1:
	    		sentiment='negative'
	    	else:
	    		sentiment='neutral'

	    	line_out= str(tweet_id[i])+'\t'+sentiment+'\n'
	    	f.write(line_out)
	return



#MAIN

#Calling svm_classifier for the development sets
#svm_classifier('Data/train_data_B.csv','Data/development_data_B.csv',"Data/development_output_B.txt")
#svm_classifier('Data/train_data_A.csv','Data/development_data_A.csv',"Data/development_output_A.txt")

#Calling svm_classifier for the test sets
svm_classifier('Data/train_data_B.csv','Data/test_data_B.csv',"Data/Output/test_output_B.txt")
#svm_classifier('Data/train_data_A.csv','Data/development_data_A.csv',"Data/development_output_A.txt")