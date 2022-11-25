"""
@Harsha Rauniyar 
HW5: Experiments. Implementing different models using scikit-learn for expirements
"""

#importing necessary libraries to build ML models and preprocessing
import sys
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#function make the csv file for the confusion matrix
def make_file(confusion_matrix_df,approach,dataset_name,random_seed):
    file_name="results-"+approach+"-"+dataset_name+"-"+random_seed+".csv"
    confusion_matrix_df.to_csv(file_name,sep='\t')

#taking in random seed for the program
random_seed=int(sys.argv[1])

#list of files we have to go over and build the models
file_names=["monks1.csv","hypothyroid.csv","mnist_1000.csv","votes.csv"]

#looping over all files
for file in file_names:
    
    filename=file
    #reading in file
    df=pd.read_csv(file)

    #getting unique labels from the label column
    labels=df["label"].unique()

    #pre-processing the dataset: converting the categorical attributes to one hot encodings
    column_names=list(df.columns.values)
    attributes=column_names[1:]
    ohe_attributes=[]
    check_row=df.iloc[0]
    for i in range(len(attributes)):
        if type(check_row[i+1])==str:
            ohe_attributes.append(attributes[i])


    
    dummies=pd.get_dummies(df,columns=ohe_attributes)
    merged_df=pd.concat([df,dummies],axis="columns")
    final_df=merged_df.drop(ohe_attributes, axis="columns")
    final_df = final_df.iloc[: , 1:]
    final_df=final_df.sample(random_state=random_seed, frac=1)

    #splitting into training, and test sets
    X=final_df.iloc[:, final_df.columns != 'label']
    y=final_df["label"]
    X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=random_seed)


    """
    Decision Tree creating model and making confusion matrix for each data-set
    """
    decisiontree_model=DecisionTreeClassifier(random_state=random_seed)
    decisiontree_model=decisiontree_model.fit(X_train, y_train)
    decision_tree_predictions=decisiontree_model.predict(X_test)
    
    
    confusion_matrixpass=confusion_matrix(y_test,decision_tree_predictions,labels=labels)
    confusion_matrix_df=pd.DataFrame(confusion_matrixpass, index=labels, columns=labels)
    print("Decision Tree accuracy for "+filename+" :"+str((accuracy_score(y_test,decision_tree_predictions)))) 
    make_file(confusion_matrix_df,"DecisionTrees",filename,str(random_seed))


    """
    Random Forest creating model and making confusion matrix for each data-set
    """

    random_forest_model= RandomForestClassifier(random_state=random_seed)
    random_forest_model=random_forest_model.fit(X_train,y_train)
    random_forest_predictions= random_forest_model.predict(X_test)
    
    confusion_matrixpass=confusion_matrix(y_test,random_forest_predictions,labels=labels)
    confusion_matrix_df=pd.DataFrame(confusion_matrixpass, index=labels, columns=labels)
    print("Random Forest accuracy for "+filename+" :"+str((accuracy_score(y_test,random_forest_predictions)))) 
    make_file(confusion_matrix_df,"Random Forests",filename,str(random_seed))


    """
    Neural Network creating model and making confusion matrix for each data-set
    """

    #decide hidden layer size for this
    #how is this working
    neural_network= MLPClassifier(random_state=random_seed, max_iter=500, hidden_layer_sizes=(len(attributes)),)
    neural_network=neural_network.fit(X_train,y_train)
    neural_network_predictions=neural_network.predict(X_test)

    confusion_matrixpass=confusion_matrix(y_test,neural_network_predictions,labels=labels)
    confusion_matrix_df=pd.DataFrame(confusion_matrixpass, index=labels, columns=labels)
    print("Neural Network accuracy for "+filename+" :"+str((accuracy_score(y_test,neural_network_predictions)))) 
    make_file(confusion_matrix_df,"Neural Networks",filename,str(random_seed))










