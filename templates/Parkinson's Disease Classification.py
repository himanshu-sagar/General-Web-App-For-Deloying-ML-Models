import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs

from io import BytesIO
import base64

# Loading DataSet
def load_data(file="\pd_speech_features.csv"):
    file="static"+file
    data=pd.read_csv(file)
    print(data.head())
    # cleaning
    col_names = data.columns
    for c in col_names:
        data[c] = data[c].replace("?", np.NaN)

    data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return data


# Explore DataSet
def data_visualisation(data):
    img_0 = BytesIO()
    img_1 = BytesIO()
    sns.countplot(x="class",data=data)
    plt.savefig(img_0, format='png')
    plt.close()
    img_0.seek(0)
    plot_url_0 = base64.b64encode(img_0.getvalue()).decode('utf8')

    sns.catplot(x="class",y="numPulses",data=data)
    plt.savefig(img_1, format='png')
    plt.close()
    img_1.seek(0)
    plot_url_1 = base64.b64encode(img_1.getvalue()).decode('utf8')

    return plot_url_0,plot_url_1


def data_asArray(data):

    X = data.values[:, 0:754]
    Y = data.values[:,754]

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.25, random_state = 50)

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)

    return X_train, X_test, y_train, y_test,X,Y

def model(algorithm,X_train, X_test, y_train, y_test,epochs=100):
    global model
    if algorithm== "DecisionTreeClassifier":
        model = DecisionTreeClassifier(criterion = "gini", random_state = 50,max_depth=5, min_samples_leaf=5)
        model_details=model.fit(X_train,y_train)

        score=model.score(X_train,y_train)
        
        pred=model.predict(X_test)

        accuracy=accuracy_score(y_test,pred)*100

        return model_details,accuracy
    elif algorithm == "SVM":
        model=svm.SVC()
        model_details=model.fit(X_train,y_train)
        pred=model.predict(X_test)
        accuracy=accuracy_score(y_test,pred)*100
        return model_details,accuracy


def predict(test_row,X,Y):
    output=model.predict(X[test_row:test_row+1,:])
    return Y[test_row],output



def run(algorithm):
    global X,Y
    data=load_data()
    plot_url_0,plot_url_1=data_visualisation(data)
    X_train, X_test, y_train, y_test,X,Y=data_asArray(data)
    
    model_details,accuracy=model(algorithm,X_train, X_test, y_train, y_test,150)

    return model_details,accuracy, plot_url_0,plot_url_1




    




    