#This is a function to find the best Classifier.
#It takes in the classifiers from classifiers_functions.py 
#from classifier_functions import logit_classifier, decision_trees, random_forest, naive_bayes
def best_classifier(train_features,train_labels):
    import pandas as pd
    from classifiers.py import logit_classifier, decision_tree, random_forest, naive_bayes
    l_score = logit_classifier(train_features,train_labels)[1]
    dt_score = decision_tree(train_features,train_labels)[1]
    rf_score = random_forest(train_features, train_labels)[1]
    nb_score = naive_bayes(train_features,train_labels)[1]
    results = pd.DataFrame({'Model':['Logistic Regression','Decision trees',
                                     'Random Forest','Naive Bayes'],
                            'Accuracy':[l_score,dt_score,rf_score,nb_score]})
    results = results.sort_values(by = "Accuracy",ascending = False)
    results = results.reset_index(drop = True)
    print("---------------------------------")
    print(results)
    print("---------------------------------")