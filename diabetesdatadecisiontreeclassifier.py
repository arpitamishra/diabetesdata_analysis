#loading required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#reading data
def decisiontreeclassifier():
    print("this is Decision tree classifier:")
    data = pd.read_csv(".\Diabetesdata.csv")
    print(data.head())#displaying first five rows of the dataset.
    #extracting fetaures and labels
    features = data.drop('Outcome',axis=1)
    print(features)
    label = data['Outcome']
    print(label)
    #splitting dataset into training set and test set
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.5)
    # print(x_test,x_train,y_train,y_test)
    #defining the decision tree model.
    max_depth = int(input("enter the max_depth of tree"))
    min_samples_split = int(input("enter the min_samples_split of tree"))
    min_samples_leaf = int(input("enter the min_samples_leaf of tree"))
    random_state = int(input("enter the random_state of tree"))
    min_impurity_decrease = float(input("enter the min_impurity_decrease of tree"))
    dtree = DecisionTreeClassifier(max_depth=max_depth,criterion='gini',splitter='best',min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=8,presort=True,random_state=random_state,min_impurity_decrease=min_impurity_decrease)
    dtree.fit(x_train,y_train)#fitting the model
    #prediction
    prediction = dtree.predict(x_test)
    print(prediction)
    #applying the scoring metrics.
    print("This is the confusion matrix:",confusion_matrix(y_test,prediction))
    print("This is the classification report:",classification_report(y_test,prediction))
    print("This is the accuracy:",accuracy_score(y_test,prediction))

if __name__=="__main__":
    decisiontreeclassifier()
