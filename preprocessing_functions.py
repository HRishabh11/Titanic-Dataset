#A study of classification technique in the Titani Dataset

#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#This function checks for missing values
#If more than 25% of onservations are missing, it drops the feature
#If less than 25% of observations are missing, it imputes them using the mean
def missing_data(data):
    col_names = data.columns
    total = data.isnull().sum()
    percent = round((data.isnull().sum()/data.isnull().count()),4)*100
    table = pd.concat([total, percent],axis =1, keys = ["Total"," percent"])
    print("-------------------------------------------------")
    print("The missing data summary is:") 
    print(table)
    print("-------------------------------------------------")
    rm_col = []
    impute_col = []
    for i in range(len(total)):
        if percent[i] > 25:
            rm_col.append(i)
        elif percent[i] != 0 and percent[i] <= 25:
            impute_col.append(i)  
    #Removing the missing data col
    data = data.drop(col_names[np.array(rm_col)],axis = 1)
    impute_col = np.array(impute_col)
    #Imputing the missing datacol
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan,strategy = "mean")
    data.iloc[:,impute_col] = imputer.fit_transform(data.iloc[:,impute_col])
    return data

#data is the training data
#col is the column indexes to encode. It should be a list of indices
def encoder(data,col):
    for i in col:
        dummy = pd.get_dummies(data.iloc[:,i])
        dummy = dummy.drop(dummy.columns[[1]],axis =1)
        data = pd.concat([data,dummy],axis = 1)
    data = data.drop(data.columns[col],axis = 1)
    return data

#Comparing the distributions for age between survivors and victims
#the input variable should be a string
def dist_compare(data,var):
   plt.figure(figsize =[10,10])
   sns.distplot(data[data['Survived'] == 1][var],kde = False, color = 'b',label = "Survived")
   sns.distplot(data[data['Survived'] == 0][var],kde = False, color = 'r', label = "Dead")
   plt.legend()
   plt.title("Comparing distribution of {} for survivors and victims".format(var))
   plt.show()

