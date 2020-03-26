#This is a function to find the best Classifier.
#It takes in the classifiers from classifiers_functions.py 
#from classifier_functions import logit_classifier, decision_trees, random_forest, naive_bayes
def best_classifier(train_features,train_labels,valid_features,valid_labels):
    import pandas as pd
    from classifier_functions import logit_classifier, decision_tree, random_forest, naive_bayes
    l_classifier, l_score = logit_classifier(train_features,train_labels)
    dt_classifier, dt_score = decision_tree(train_features,train_labels)
    rf_classifier, rf_score = random_forest(train_features, train_labels)
    nb_classifier, nb_score = naive_bayes(train_features,train_labels)
    results = pd.DataFrame({'Model':['Logistic Regression','Decision trees',
                                     'Random Forest','Naive Bayes'],
                            'Accuracy':[l_score,dt_score,rf_score,nb_score]})
    results = results.sort_values(by = "Accuracy",ascending = False)
    results = results.reset_index(drop = True)
    print("---------------------------------")
    print("Results for Training Set:")
    print(results)
    print("---------------------------------")
    l_data = valid_features.drop(["Parch","Fare"],axis =1)
    from classifier_functions import predict
    l_pred = predict(l_classifier,l_data)[1]
    dt_pred = dt_classifier.predict(valid_features)
    rf_pred = rf_classifier.predict(valid_features)
    nb_pred = nb_classifier.predict(valid_features)
    pred_values = [l_pred, dt_pred, rf_pred, nb_pred]
    Accuracy = []
    from sklearn.metrics import confusion_matrix
    for i in pred_values:
        cm = confusion_matrix(valid_labels, i)
        score = (cm[0,0]+cm[1,1])/cm.sum()
        Accuracy.append(score*100)
    results_2 = pd.DataFrame({'Model':['Logistic Regression','Decision trees',
                                     'Random Forest','Naive Bayes'],
                               'Accuracy': Accuracy})
    print("---------------------------------")
    print("Results for Validation Set")
    print(results_2.sort_values(by = "Accuracy", ascending = False))
    print("---------------------------------")
    