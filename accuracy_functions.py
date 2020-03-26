#This script contains functions used for checking the accuracy, model validation
#Also contains functions for ROC curve

#importing packages
import matplotlib.pyplot as plt
import seaborn as sns


#this function gives the accuracy of the classifier
#prints confusion matrix and classification report
#imputs = predicted outcomes and true outcomes
#data_type means either training set or validation set or test set
def accuracy(predicted_outcomes, actual_outcomes, data_type):
    #checking the accuracy
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    cm = confusion_matrix(actual_outcomes,predicted_outcomes)
    print('----------------------------------------------------')
    print("The confusion matrix for {} is given below:".format(data_type))
    print(cm)
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    print("The accuracy of Classifier for train set is {} %".format(round(accuracy*100,2)))
    report = classification_report(actual_outcomes, predicted_outcomes)
    print("----------------------------------------------------")
    print("The classification for {} report is given below: ".format(data_type))
    print(report)
    print('----------------------------------------------------')
    
#function to plot ROC
#plots the ROC curve and gives AUC value
#inputs true outcomes and predicted probabilities
def roc(actual_outcomes, predicted_prob,plot_title):
    #ROC curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(actual_outcomes,predicted_prob)
    fpr,tpr,thresholds = roc_curve(actual_outcomes,predicted_prob)
    
    #plotting the ROC
    plt.figure(figsize = [10,10])
    plt.plot(fpr,tpr,label = 'Score: {}'.format(round(logit_roc_auc,2)))
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_title)
    plt.legend(loc = 'lower right')
    plt.show()

#plotting a 2 dimensional representation of our classification
#I have used PCA to decompose the test data set into two principle component
def dim_plot(data,predicted,plot_title):
    #Visualising the test results in two dimensions
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components = 2)
    test_2dim = pca.fit_transform(data_scaled)
    
    #plotting the results
    plt.figure(figsize = [10,10])
    sns.scatterplot(x = test_2dim[:,0],y=test_2dim[:,1],hue = predicted,palette = 'pastel')
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.title(plot_title)
    plt.legend()
    plt.show()