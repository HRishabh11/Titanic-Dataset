#Description of dataset feature
#survival:    Survival 
#PassengerId: Unique Id of a passenger. 
#pclass:    Ticket class     
#sex:    Sex     
#Age:    Age in years     
#sibsp:    # of siblings / spouses aboard the Titanic     
#parch:    # of parents / children aboard the Titanic     
#ticket:    Ticket number     
#fare:    Passenger fare     
#cabin:    Cabin number     
#embarked:    Port of Embarkation

#import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the modules
#from importlib import import_module
#import_module('preprocessing_functions')
#import_module('classifier_functions')
#import_module('accuracy_functions')

#Importing self created functions
import preprocessing_functions

#Loading the Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Data Visualization to understand the data
#Visulaizing the data based on Sex
plt.figure(figsize = [10,10])
sns.barplot(x = "Sex", y ="Survived", data =train)
plt.title("Barplot for Sex")
plt.show()
print("-------------------------------------------------")
print("From the Barplot for Sex, we can see:")
print("The above generated bar plot shows that more female survived than males.")
print("This may be because of the fact that females were evacuated first.")
print("-------------------------------------------------")

#Visualizing the data based on PClass
plt.figure(figsize = [10,10])
sns.barplot(x = "Pclass", y ="Survived", data = train)
plt.title("Barplot for Pclass")
plt.show()
print("-------------------------------------------------")
print("From the Barplot for Pclass, we can see:")
print("The above barplot shows that passengers in the 1st class survived more than the other two.")
print("The lowest survival was in 3rd class.")
print("This may be because the priority during evacuation was given to 1st class passengers.")
print("-------------------------------------------------")
#checking the distribution plots to see the distribution of passengers based on various passengers
plt.figure(figsize = (10,10))
sns.distplot(train["Age"],color = 'r')
plt.title("Distribution of the passengers of Titanic based on Age")
plt.show()

plt.figure(figsize = (10,10))
sns.distplot(train["Fare"],color = 'b')
plt.title("Distribution of the passengers of Titanic based on Fare")
plt.show()

#comparing the distribution of Survivors and Victims using distribution plots
#Using the function I created dist_compare

preprocessing_functions.dist_compare(train,'Age')
preprocessing_functions.dist_compare(train,"Fare")




#Now we will start our Analysis
#Dropping useless features
X = train.drop(['Survived','PassengerId',"Name","Ticket","Embarked"], axis = 1)
y = train['Survived']
X_test = test.drop(['PassengerId','Name','Ticket','Embarked'],axis = 1)

#splitting the training data into train and validaion set
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2,random_state = 0)

#Dealing with missing data and categorical variables
#I have created two functions missing_data and encoder
#missing_data removes features with more than 25% values missing and imputes the missing value for others
#encoder creates dummy variables
#encoder deals with dummy variable traps
from preprocessing_functions import missing_data, encoder
X_train = missing_data(X_train)
X_train = encoder(X_train,[0,1])
X_valid = missing_data(X_valid)
X_valid = encoder(X_valid,[0,1])
X_test = missing_data(X_test)
X_test = encoder(X_test,[0,1])


####################################

#Logistic Regression

####################################
from classifier_functions import logit_classifier, predict
from accuracy_functions import accuracy, roc, dim_plot
#Training the model

#Finding the logistic Regression classifier using logit_classifier function
l_classifier, l_score = logit_classifier(X_train,y_train)

#predicting the training set values
y_train_pred_scores,y_train_pred = predict(l_classifier,X_train.drop(['Parch','Fare'],axis = 1))

#Checking the accuracy of the model
accuracy(y_train_pred,y_train,"Train Set")    

#plotting the ROC curve
roc(y_train,y_train_pred_scores,"ROC of train set-Logistic regression")

#Validating the results
#Checking the validation set results
y_valid_pred_scores, y_valid_pred = predict(l_classifier,X_valid.drop(['Parch','Fare'],axis = 1))

#checking the accuracy of the validation set
accuracy(y_valid_pred,y_valid,"Validation Set")

#plotting the ROC curve for validation set
roc(y_valid,y_valid_pred_scores,"ROC of Validation set-Logistic Regression")

#Predicting the outcomes for test set
#For test set
#preparing the test set
X_test =X_test.drop(['Parch','Fare'],axis =1)

#predicting the outcome for test set
y_pred_scores, y_pred = predict(l_classifier, X_test)

#2 dimensional representation of our prediction
dim_plot(X_test,y_pred,"2 dim representation of our classification -Logistic")

#############################################

#Decision Trees

#############################################

from classifier_functions import decision_tree
dt_classifier, dt_score = decision_tree(X_train,y_train)

#predicting the results for train set
y_train_pred = dt_classifier.predict(X_train)

#Checking the accuracy
accuracy(y_train_pred, y_train,"Train Set")

#plotting roc
roc(y_train,y_train_pred,"ROC for Decision Tress- Train set")

#predcting the results for validation set
y_valid_pred = dt_classifier.predict(X_valid)

#checking the accuracy in the validation set
accuracy(y_valid_pred,y_valid,"Validation Set")

#roc for validation set
roc(y_valid,y_valid_pred,"ROC for Decision Tress- Validation set")

#previously for Logistic model we removed some features from our data
#Undoing this
X_test = test.drop(['PassengerId','Name','Ticket','Embarked'],axis = 1)
X_test = missing_data(X_test)
X_test = encoder(X_test,col=[0,1])
#Predicting output for test set
y_test_pred = dt_classifier.predict(X_test)


dim_plot(X_test,y_test_pred,"2 dim representation of our classification -Decision Tress")


########################################

#Random Forest

########################################

from classifier_functions import random_forest
rf_classifier, rf_score = random_forest(X_train,y_train)

#predicting the outcomes
y_train_pred = rf_classifier.predict(X_train)

#checking the accuracy again
accuracy(y_train_pred,y_train,"Train Set")

#roc curve for training data
roc(y_train,y_train_pred,"ROC for Random Forest-Train")

#Looking at validation set
y_valid_pred = rf_classifier.predict(X_valid)

#checking the accuracy
accuracy(y_valid_pred,y_valid,"Validation Set")

#roc curve for validation data
roc(y_valid,y_valid_pred,"ROC for Random Forest- Validation")

#############################################

#Naive Bayes Classifier

#############################################
from classifier_functions import naive_bayes
nb_classifier, nb_score = naive_bayes(X_train,y_train)

#predicting the outcomes
y_train_pred = nb_classifier.predict(X_train)

#checking the accuracy
accuracy(y_train_pred,y_train,"Train Set")

#plotting the roc
roc(y_train,y_train_pred,"ROC for Naive Bayes-Train")

#checking for validation set
y_valid_pred = nb_classifier.predict(X_valid)

#checking the accuracy 
accuracy(y_valid_pred, y_valid,"Validation Set")

#plotting the roc
roc(y_valid,y_valid_pred,"ROC for Naive Bayes-Validation")

##############################################

#Best Classifier

##############################################
#Finding the best model
from model_functions import best_classifier
best_classifier(X_train,y_train,X_valid,y_valid)

#It seems the Randon Forest model performs on this data.
#We have determined that the best model is Random Forest

##############################################

#Cross-Validation

#############################################
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
score = cross_val_score(rf,X_train,y_train,cv = 10, scoring = "accuracy")
print("Scores:",score)
print("Mean:",score.mean())
print("Variance:",score.var()) 

#Our average accuracy is 81% and variance of the accuracies is 0.07%.

##############################################

#HyperParameter TUning

##############################################

param_grid = {"criterion":["gini","entropy"],"min_samples_leaf":[1,5,10,25,50,70],
              "min_samples_split":[2,4,10,12,16,18,25,35],"n_estimators":[100,400,700,1000,1500]}
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators = 100, max_features = 'auto', oob_score = True, random_state = 1,n_jobs = -1)
clf = GridSearchCV(estimator = rf, param_grid = param_grid, n_jobs = -1)
clf.fit(X_train,y_train)
clf.best_params_

###############################################

#Final Model and Analysis

###############################################
#we have determined that the best classifier is the Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, criterion = 'gini',min_samples_leaf = 1, min_samples_split = 10)
rf.fit(X_train,y_train)
y_valid_pred = rf.predict(X_valid)
accuracy(y_valid_pred,y_valid,"Validation Set")
roc(y_valid,y_valid_pred,"ROC curve for Random Forest with best parameters - Validation Set")
#Now making the best prediction for our test set
best_pred = rf.predict(X_test)
best_pred
