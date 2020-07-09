# load required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#load required data
def naivebayesclassifier():
    print("this is naive bayes classifier:")
    data = pd.read_csv(".\Diabetesdata.csv")
    print(data.head())
    #define features and value to be predicted
    features = data.drop('Outcome',axis=1)
    print(features.head())
    label = data['Outcome']
    print(label.head())
    #split data into training and test data
    x_train,x_test,y_train,y_test = train_test_split(features,label,train_size=0.5)
    print(x_test,x_train,y_train,y_test)
    #fit the model
    model = GaussianNB()
    model.fit(x_train,y_train)
    predicted= model.predict(x_test) # 0:Overcast, 2:Mild
    #print("Predicted Value:", predicted)
    #apply evaluation metrics
    print("The accuracy is:",accuracy_score(y_test,predicted))

if __name__=="__main__":
    naivebayesclassifier()