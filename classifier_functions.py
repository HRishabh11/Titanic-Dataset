#Setting up classifiers
#This script contains functions to create classifiers

#importing packages
#A study of classification technique in the Titani Dataset

#Logistic Regression
def logit_classifier(train_features,train_labels):
    #logistic Regression
    #backward selection algorithm for feature selection using p-values
    col_names = train_features.columns
    score = 1  
    while score > 0:
        import statsmodels.api as sm
        regressor = sm.Logit(train_labels,train_features)
        result = regressor.fit()
        l = list(result.pvalues > 0.05)
        droplist = [i for i, x in enumerate(l) if x]
        train_features = train_features.drop(col_names[droplist],axis = 1)
        score = sum(l)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(train_labels,result.predict(train_features).astype(int))
    score = (cm[0,0]+cm[1,1])/cm.sum()
    return result, score
    #this function returns a classifier object

#Decision Trees
def decision_tree(train_features,train_labels):   
    from sklearn.tree import DecisionTreeClassifier
    result = DecisionTreeClassifier()
    result.fit(train_features,train_labels)
    score = round(result.score(train_features, train_labels)*100,2)
   # print("The accuracy on this training set is {}".format(score))
    return result, score

#Random Forest
def random_forest(train_features,train_labels):
    from sklearn.ensemble import RandomForestClassifier
    result = RandomForestClassifier()
    result.fit(train_features,train_labels)
    score = round(result.score(train_features,train_labels)*100,2)
   # print("The accuracy on this training set is {}".format(score))
    return result, score

#Gaussian Naive Bayes
def naive_bayes(train_features,train_labels):
    from sklearn.naive_bayes import GaussianNB
    result = GaussianNB()
    result.fit(train_features,train_labels)
    score = round(result.score(train_features,train_labels)*100,2)
   # print("The accuracy of this classifier is {}".format(score))
    return result, score
    
#function to predict the labels
#returns both prediction probabilities and outcomes
def predict(classifier,features):
    #predicting the reults
    pred_prob = classifier.predict(features)
    pred = (pred_prob > 0.5) 
    pred *= 1
    return pred_prob, pred