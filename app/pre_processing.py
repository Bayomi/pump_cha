import pandas as pd 
import numpy as np


def getProcessedData(file_name):
    
    data = pd.read_csv(file_name)
    missing_percentage = list()

    #choose the relevant variables
    #data = data_all[['amount_tsh', 'population', 'funder', 'basin', 'region', 'scheme_management', 'permit','construction_year','extraction_type','management','payment','quality_group','source_type','waterpoint_type']]
    
    """
    Working only with the Independent variables first
    """

    ###construction_year
    #construction year -> age
    data['construction_year'] = 2016 - data['construction_year'] 
    data['construction_year'] = data['construction_year'].replace([2016], [0])
    #missing data -> mean of the values
    medianConstructionFrame = data['construction_year'].replace(0, np.NaN)
    data['construction_year'] = data['construction_year'].replace(0, medianConstructionFrame.median())

    #normalize
    data['construction_year'] = data['construction_year']/data['construction_year'].max()

    ###amount_tsh
    medianConstructionFrame = data['amount_tsh'].replace(0, np.NaN)
    data['amount_tsh'] = data['amount_tsh'].replace(0, medianConstructionFrame.median())

    #normalize
    data['amount_tsh'] = data['amount_tsh']/data['amount_tsh'].max()

    ###population
    medianConstructionFrame = data['population'].replace(0, np.NaN)
    data['population'] = data['population'].replace(0, medianConstructionFrame.median())

    #normalize
    data['population'] = data['population']/data['population'].max()

    ###funder
    ###categorical => Government (1); Other(0)
    data.loc[data['funder'] != 'Government Of Tanzania', 'funder'] = 0
    data.loc[data['funder'] == 'Government Of Tanzania', 'funder'] = 1

     ###permit
    ###categorical => True (1); False(0) ; Missing()
    data.loc[data['permit'] == True, 'permit'] = 1
    data.loc[data['permit'] == False, 'permit'] = -1
    data['permit'] = data['permit'].replace(np.NaN, 0)

    #normalize
    data['latitude'] = data['latitude']/data['latitude'].min()
    
    #normalize
    data['longitude'] = data['longitude']/data['longitude'].max()

    #choose the relevant variables
    
    categoricalColumns=['public_meeting','quantity','basin','extraction_type','region','scheme_management','water_quality','management_group', 'payment','source_type','waterpoint_type']  
    
    dfCatVar = data[categoricalColumns]

    #get dummies for the choosen variables
    df_dummy = pd.concat([pd.get_dummies(dfCatVar[col]) for col in dfCatVar], axis=1, keys=dfCatVar.columns)
    nonCategoricalColumns = ['id','amount_tsh','population','construction_year','latitude','longitude']
    
    #joining together the variables
    df_final = pd.concat([df_dummy,data[nonCategoricalColumns]], axis = 1)

    
    return df_final

def getData():
	fileName = "trainingSetValues.csv"
	return getProcessedData(fileName)

