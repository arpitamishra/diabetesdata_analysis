import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def knnclassifier():
    data = pd.read_csv("Diabetesdata.csv",usecols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
    print(data.head())
    Pregnancies=data['Pregnancies']
    Glucose=data['Glucose']
    BloodPressure=data['BloodPressure']
    SkinThickness=data['SkinThickness']
    Insulin=data['Insulin']
    BMI=data['BMI']
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    Age=data['Age']
    Outcome=data['Outcome']
    s = stats.normaltest(Outcome)
    print(s)
    features = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    print(features)
    x = data.drop("Outcome",axis=1)
    print(" x  is ",x)
    y = data.iloc[:,:-768]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50)
    # print(X_train,y_train,X_test,y_test)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,X_test)
    predicted = model.predict(X_test)
    print(predicted)
    accuracy = accuracy_score(y_test,predicted)
    print("accuracy score is",accuracy)
    classification_report = classification_report(predicted,y_test)
    print("classification report",classification_report)
if __name__=="__main__":
    knnclassifier()