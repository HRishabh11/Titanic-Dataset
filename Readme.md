# Classification in Titanic Dataset #
This is a simple project where I am trying to find the best classifier for the Titanic Dataset. This folder contains the following files:
1. preprocessing_functions.py
2. classiifer_functions.py
3. accuracy_functions.py
4. model_functions.py
5. titanic.py

## Description of each file ##
1. preprocessing_functions.py 
	- missing_data() function takes in the Titanic Dataset and checks for missing values. If 			any feature has more than 25% missing value, it removes them. If it has less than 			25% missing values it imputes them using the mean.
	- encoder() function encodes the categorical variables. You have to specify the columns 		with categorical variables as a list. 

2. classifier_functions.py
	- logit_regression() function does logistic regression and outputs the resulting 			classifier and the accuracy of the model on the training data. It also does 			variable selection based on the pvalues. 
	- decision_trees() function returns the decision trees classifier and the accuracy.
	- random_forest() function returns the random forest classifier and the accuracy.
	- naive_bayes() function returns the naive bayes classifier and the accuracy.
	**All the above classifier takes in the training data and the training label as its 		inputs**
	- predict() function takes in the logistic classifier(output from logit_regression()) and 			the prediction features, and returns the predicted values. We have to manually 			remove the variables of prediction data that were removed by logit_regression 			function while selecting features in training data.

3. accuracy_functions.py
	- accuracy() gives the accuracy of the model, the confusion matrix and the classification 			report.It takes in three inputs- actual outcomes, predicted outcomes and the data 			type(training or validation).
	- roc() plots the roc curve. It takes in three inputs- actual outcomes, predicted outcomes 			and the data type(training or validation).
	- dim_plot() uses PCA to reduce the dimension of the training data to 2 dimensions and 			plots the data and their classification. Useful for data visualization. It takes 			in three inputs- actual outcomes, predicted outcomes and the data type(training or 			validation).

4. model_functions.py
	- best_classifier() gives the best classifier. It takes in four inputs-training fetures, 			training labels, validation features and validation data and gives the accuracies 			of the four classifiers for both training set and validation set.

5. titanic.py script contains the analysis that I coducted. I used the previous defined functions 		and other functions to find the best classifier. I conducted cross validation and 		Hyperparameter tuning to get the best parameters.
