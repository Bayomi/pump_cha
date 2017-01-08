import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import pre_processing as pp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools


"""
Process the data using the pre_processing.py file parameters.
@param {String} trainValuesFile - The training values file directory string.
@param {String} trainLabelsFile - The training labels file directory string.
@param {String} testValuesFile - The testing values file directory string.
@return {Array} the processed training and testing variables
"""
def preprocess(trainValuesFile, trainLabelsFile, testValuesFile):
	#Getting the pre processed data
	train_variables = pp.getProcessedData(trainValuesFile)
	test_variables = pp.getProcessedData(testValuesFile)

	train_labels = pd.read_csv(trainLabelsFile) 
	train_variables = pd.merge(train_variables,train_labels,on = 'id')

	listTrain = list(train_variables)
	listTest = list(test_variables)

	#get the same group of variables for both lists
	for el in listTrain:
		if not(el in listTest):
			if el != 'status_group':
				del train_variables[el]
	
	for el2 in listTest:
		if not(el2 in listTrain):
			del test_variables[el]

	return train_variables, test_variables

	
"""
Divide into predictors (values) and targets (dependent variable label).
@param {Array} train_variables - the reference array for the train values.
@return {Array} the division between predictors (values) and targets
"""
def getPredictorsAndTargets(train_variables):
	data_merged = train_variables

	#Define Labels
	data_merged.loc[data_merged['status_group'] == 'functional', 'dependentVariable'] = 1
	data_merged.loc[data_merged['status_group'] == 'non functional', 'dependentVariable'] = -1
	data_merged.loc[data_merged['status_group'] == 'functional needs repair', 'dependentVariable'] = 0    
	   
	nameOfTargetVariable = 'dependentVariable'

	myColumns = list(train_variables)
	myColumns.remove('id')
	myColumns.remove('status_group')
	myColumns.remove('dependentVariable')

	predictors = data_merged[myColumns]

	targets = data_merged.dependentVariable

	return predictors, targets


"""
Divide into predictors (values) ids.
@param {Array} test_variables - the reference array for the testing values.
@return {Array} the division between predictors (values) and ids
"""
def divideTestAndIds(test_variables):

	myColumns = list(test_variables)
	myColumns.remove('id')

	pred_test = test_variables[myColumns]
	ids = test_variables['id']

	return pred_test, ids


"""
Process the data using the pre_processing.py file parameters.
@param {Array} predictors - the reference array for the training values.
@param {Array} targets - the reference array for the training labels.
@param {Number} percentage - The percentage to split the data into test/train.
@return {Array} the arrays of training predictions, training tests, trainings targets and training tests
"""
def splitData(predictors, targets, percentage):
	#60% train, 40% test
	pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=percentage)

	pred_train.shape
	pred_test.shape
	tar_train.shape
	tar_test.shape

	return pred_train, pred_test, tar_train, tar_test


"""
Process the data using the pre_processing.py file parameters.
@param {Array} pred_train - the reference array for the training values.
@param {Array} tar_train - the reference array for the training labels.
@param {Array} pred_test - the reference array for the testing values.
@return {Array} the processed training and testing variables
"""
def getPredictions(pred_train, tar_train, pred_test):
	#Build model on training data
	classifier=RandomForestClassifier(n_estimators=800, max_features=0.45, min_samples_leaf=1)
	classifier=classifier.fit(pred_train,tar_train)

	predictions = classifier.predict(pred_test)

	return predictions

"""
Process the data using the pre_processing.py file parameters and get the score.
@param {Array} pred_train - the reference array for the training values.
@param {Array} tar_train - the reference array for the training labels.
@param {Array} pred_test - the reference array for the testing values.
@return {Array} the processed training and testing variables
"""
def getScore(pred_train, tar_train, pred_test, tar_test):
	#Build model on training data
	classifier=RandomForestClassifier(n_estimators=15)
	classifier=classifier.fit(pred_train,tar_train)

	score = classifier.score(pred_test, tar_test)

	return score


"""
Get all the permutations from array of arrays.
@param {Array of arrays} pred_train - array with all the arrays to permute.
@return {Array of arrays} all the permutations
"""
def getPermutations(gen): 
	permutes = []
	for array in itertools.product(*gen):
		permutes.append(list(array))

	return permutes

"""
Test parameters.
@param {Array} pred_train - the reference array for the training values.
@param {Array} tar_train - the reference array for the training labels.
@param {Array} pred_test - the reference array for the testing values.
@return {Array} the processed training and testing variables
"""
def testParameters(pred_train, tar_train, pred_test, tar_test):
	#Build model on training data

	n_estimators = [800]
	max_features = [0.5]
	min_samples_leaf = [1]

	gen = []
	gen.append(n_estimators)
	gen.append(max_features)
	gen.append(min_samples_leaf)

	permutes = getPermutations(gen)
	print permutes
	scores = []
	tries = []

	for permute in permutes:
		n_estimators_try = permute[0]
		max_features_try = permute[1]
		min_samples_leaf_try = permute[2]
		classifier=RandomForestClassifier(n_estimators=n_estimators_try, max_features=max_features_try, min_samples_leaf=min_samples_leaf_try, random_state=1, n_jobs = -1)
		classifier.fit(pred_train,tar_train)
		score = classifier.score(pred_test, tar_test)
		tries.append(permute)
		scores.append(score)

	max_value = max(scores)
	max_index = scores.index(max_value)
	max_try = tries[max_index]

	print max_value
	print max_try

	
	#classifier=RandomForestClassifier(n_estimators=15, max_features="auto", min_samples_leaf=50, random_state=1)
	#classifier=classifier.fit(pred_train,tar_train)

	#score = classifier.score(pred_test, tar_test)

	#print score



"""
Go to all the training steps and save the output into the result folder.
@param {String} trainining_file - The training values file directory string.
@param {String} training_labels- The training labels file directory string.
@param {String} test_file - The testing values file directory string.
@param {String} result_file - The output file for results.
"""
def trainAndSaveResult(trainining_file, training_labels, test_file, result_file):
	fileNameValues = trainining_file
	fileNameLabels = training_labels
	fileNameTest = test_file

	#process
	train_variables, test_variables = preprocess(fileNameValues, fileNameLabels, fileNameTest)
	predictors, targets = getPredictorsAndTargets(train_variables)
	pred_test, ids = divideTestAndIds(test_variables)

	#predict
	predictions = getPredictions(predictors, targets, pred_test)

	#save to a data frame
	predictions_dataFrame = pd.DataFrame({"id": ids})
	predictions_dataFrame['status'] = predictions

	#re-define Labels
	predictions_dataFrame.loc[predictions_dataFrame['status'] == 1, 'status_group'] = 'functional'
	predictions_dataFrame.loc[predictions_dataFrame['status'] == -1, 'status_group'] = 'non functional'
	predictions_dataFrame.loc[predictions_dataFrame['status'] == 0, 'status_group'] = 'functional needs repair'

	#send to csv file
	del predictions_dataFrame['status']
	predictions_dataFrame.to_csv(result_file, index=False, dtype=object)

"""
Test the efficiency by the division of the training sample.
@param {String} trainining_file - The training values file directory string.
@param {String} training_labels- The training labels file directory string.
@param {String} test_file - The testing values file directory string.
@param {String} result_file - The output file for results.
"""
def testEfficiency(trainining_file, training_labels, test_file, result_file):
	fileNameValues = trainining_file
	fileNameLabels = training_labels
	fileNameTest = test_file
	percentage = 0.6

	#process
	train_variables, test_variables = preprocess(fileNameValues, fileNameLabels, fileNameTest)
	predictors, targets = getPredictorsAndTargets(train_variables)
	pred_train, pred_test, tar_train, tar_test = splitData(predictors, targets, percentage)

	#predict
	testParameters(pred_train, tar_train, pred_test, tar_test)



#determine the files directory for input and output
fileNameValues = "data/trainV.csv"
fileNameLabels = "data/trainL.csv"
fileNameTest = "data/test.csv"
fileResult = 'result/data.csv'

#run the training method and save the results
#trainAndSaveResult(fileNameValues, fileNameLabels, fileNameTest, fileResult)

#test the efficiency
testEfficiency(fileNameValues, fileNameLabels, fileNameTest, fileResult)


