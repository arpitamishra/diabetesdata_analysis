#loading required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#reading data
def randomforestclassifier():
    print("this is Random forest classifier:")
    data = pd.read_csv(".\Diabetesdata.csv")
    print(data.head())
    #extracting fetaures and labels
    features = data.drop('Outcome',axis=1)
    print(features)
    label = data['Outcome']
    print(label)
    #splitting dataset into training set and test set
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.5)
    # print(x_test,x_train,y_train,y_test)
    #fit the model
    n_estimators =int(input("enter the number of estimators"))
    #max_features=(input("enter the number of max_features"))
    min_samples_leaf = int(input("enter the number of min_samples_leaf"))
    random_state = int(input("enter the number of random_state"))
    min_samples_split = int(input("enter the number of min_samples_split"))
    model = RandomForestClassifier(n_estimators,criterion='gini', max_features='auto', min_samples_leaf=min_samples_leaf,random_state=random_state,min_samples_split=min_samples_split)
    model.fit(x_train,y_train)
    prediction = model.predict(x_test)
    print(prediction)
    #apply evaluation metrics
    print("This is the confusion matrix:",confusion_matrix(y_test,prediction))
    print("This is the classification report:",classification_report(y_test,prediction))
    print("This is the accuracy:",accuracy_score(y_test,prediction))
if __name__=="__main__":
    randomforestclassifier()
