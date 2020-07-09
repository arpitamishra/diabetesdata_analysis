 #loading required libraries
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
#reading data
def svmclassifier():
    print("this is SVM classifier:")
    data = pd.read_csv(".\Diabetesdata.csv",usecols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
    #print(data.head())
    Pregnancies=data['Pregnancies']
    # print(Pregnancies)
    Glucose=data['Glucose']
    # print(Glucose)
    BloodPressure=data['BloodPressure']
    # print(BloodPressure)
    SkinThickness=data['SkinThickness']
    # print(SkinThickness)
    Insulin=data['Insulin']
    # print(Insulin)
    BMI=data['BMI']
    # print(BMI)
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    # print(DiabetesPedigreeFunction)
    Age=data['Age']
    # print(Age)
    Outcome=data['Outcome']
    # print(Outcome)
    features = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    #print(features)
    #x = data.iloc[:,:-384]
    #extracting fetaures and labels
    X = data.drop("Outcome",axis=1)
    print(" x  is ",X)
    #y = data.iloc[:,:-768]
    y = data['Outcome']
    print("y is",y)
    #splitting dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # print(X_train,y_train,X_test,y_test)
    #fit the model
    classifier = svm.SVC(kernel='rbf')
    predict = classifier.fit(X,y)
    Y_pred = classifier.predict(X_test)
    print(Y_pred)
    #apply evaluation metrics
    print(confusion_matrix(y_test,Y_pred))
    print(classification_report(y_test,Y_pred))
    print("The accuracy is:",accuracy_score(y_test,Y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, Y_pred, pos_label=5)
    print(auc(fpr, tpr))

if __name__=="__main__":
    svmclassifier()