#load required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
#read required data
def logisticregressionclassifier():
    print("this is logistic regression classifier:")
    data = pd.read_csv(".\Diabetesdata.csv")
    print(data.head()) # display first 5 rows of data
    #define features and value to be predicted
    x = data.drop("Outcome",axis=1)
    print("x is",x)
    y = data['Outcome']
    print("y is ",y)
    #split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    # fit the model
    logit= LogisticRegression(random_state=0,solver='lbfgs',multi_class='ovr',max_iter=1000).fit(x,y)
    X_pred=(logit.predict(X_test))
    print("The predictions are",X_pred)
    #apply metrics for evaluation
    print(confusion_matrix(y_test,X_pred))
    print(classification_report(y_test,X_pred))
    print("The accuracy of the logistic regression  model is:",accuracy_score(y_test,X_pred))

if __name__=="__main__":
    logisticregressionclassifier()