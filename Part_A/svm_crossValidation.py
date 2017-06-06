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
from sklearn.model_selection import KFold

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
	f1=[]
	accuracy=[]
	
	#Read data file with tweets, their sentiment and their features for train_set
	train_dataframe = pd.read_csv(train_data)

	#Create np array with train labels (positive, neutral , negative) size= 1 x number of tweets
	train_labels = train_dataframe.iloc[:,1]
	train_labels = np.array(train_labels)
	
	#Create np array with normalized train features (lexicons, ngrams, bigrams etc) 
	#size= number of tweets x number of features
	#train_features = train_dataframe.iloc[:,2:802]
	train_features = train_dataframe.iloc[:,2:]
	train_features = np.array(train_features)
	np.delete(train_features, [7,8,9], axis=1)
	print(train_features[:,8:12])

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
	np.delete(test_features, [7,8,9], axis=1)


	k=0
	kf = KFold(n_splits=10)
	print("----------------------------------------------------------------------------------------------------")
	print("SVM will run now with cross validation for number of folds: ",kf.get_n_splits(train_features))
	for train_index, test_index in kf.split(train_features):
		k+=1
		print('SVM is training now for k =',k)
		#TRAIN CLASSIFIER
		cl_SVM = svm.SVC(kernel="linear",C=0.5, gamma=0.05, cache_size=3000)

		#FIT CLASSIFIER
		cl_SVM.fit(train_features[train_index], train_labels[train_index])

		print('SVM is predicting the test set now for k =',k)
		#Run SVM classifier
		results = cl_SVM.predict(train_features[test_index])
		
		#Here we print the accuracy and f1 score of the results
		#this is only going to give a score if the labels are not uknown
		pos=f1_score(train_labels[test_index],results,average="macro",labels=[1])
		neg=f1_score(train_labels[test_index],results,average="macro",labels=[-1])
		num_correct = (results == train_labels[test_index]).sum()
		recall = float(num_correct) / len(train_labels[test_index])
		f1.append(((neg+pos)/2.0)*100)
		accuracy.append(recall * 100)

	#-------------------------------
	#-- PREDICTION OF TEST LABELS --
	#-------------------------------
	print('SVM is predicting the development set now....')
	#Run SVM classifier
	results = cl_SVM.predict(test_features)

	#Here we print the accuracy and f1 score of the results
	#this is only going to give a score if the labels are not uknown
	print(precision_recall_fscore_support(test_labels,results,average="macro"))
	pos=f1_score(test_labels,results,average="macro",labels=[1])
	neg=f1_score(test_labels,results,average="macro",labels=[-1])
	num_correct = (results == test_labels).sum()
	recall = float(num_correct) / len(test_labels)
	print "model accuracy (%): ", recall * 100, "%"
	print('f1 average score',((neg+pos)/2.0)*100) 
	print('------------------------')

	print('The parameters of the SVM CLASSIFIER:')
	print(cl_SVM)
	return f1,accuracy

f1_sc=[]
acc=[]
f1_sc, acc=svm_classifier('Data/train_data_A.csv','Data/development_data_A.csv',"Data/Output/development_output_A.txt")
print("\n------------------------------------------------")
print("model accuracy (%): ", f1_sc)
print("")
print('f1 average score', acc) 
print('--------------------------------------------------')

