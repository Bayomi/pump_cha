import pandas as pd
import patsy as ps    
import statsmodels.api as sm
import numpy as np
from sklearn import linear_model, cross_validation, svm
import matplotlib.pyplot as plt
import pre_processing as pp


def getXY():

    #Reading the dependent variables
    fileName="data/trainingSetLabels.csv"
    train_labels = pd.read_csv(fileName)    

    #Getting the pre processed data
    train_variables = pp.getProcessedData("data/trainingSetValues.csv")
    
    data_merged = pd.merge(train_variables,train_labels,on = 'id')

    #Merging both dataframes

    #Define Labels
    data_merged.loc[data_merged['status_group'] == 'functional', 'dependentVariable'] = 1
    data_merged.loc[data_merged['status_group'] == 'non functional', 'dependentVariable'] = 0
    data_merged.loc[data_merged['status_group'] == 'functional needs repair', 'dependentVariable'] = 1    
   

    nameOfTargetVariable = 'dependentVariable'

    myColumns = list(train_variables)
    myColumns.remove('id')
    #print myColumns
    X = data_merged.as_matrix(columns = myColumns)

    y = np.ravel(data_merged.as_matrix(columns = [nameOfTargetVariable]))
    
    return X, y

def getScoreLogModel(X, y):

    # Linear Regression
    linModel = linear_model.LinearRegression() 
    linModel.fit(X,y) # This line is only necessary if you want to use the parameters
    #print linModel.coef_

    # Source of the implementation Idea: http://scikit-learn.org/stable/modules/cross_validation.html
    X_train, X_test, y_train, y_test = getTrainAndTest(X, y)
    
    # Logistic Regression    
    logModel = linear_model.LogisticRegression().fit(X_train,y_train)
    scoreLogModel=logModel.score(X_test, y_test)
    return scoreLogModel

def getTrainAndTest(X, y):
	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
	return X_train, X_test, y_train, y_test 


X, y = getXY()
print getScoreLogModel(X, y) * 100

    