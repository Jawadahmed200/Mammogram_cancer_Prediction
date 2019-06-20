from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression



data = pd.read_csv('data1.csv')
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        print("YES")
        classifier = request.form.get('hidd')
        print(classifier)
        a_1=request.form.get('ex1')
        a_2=request.form.get('ex2')
        a_3=request.form.get('ex3')
        a_4=request.form.get('ex4')
        a_5=request.form.get('ex5')
        a_6=request.form.get('ex6')
        a_7=request.form.get('ex7')
        a_8=request.form.get('ex8')
        a_9=request.form.get('ex9')
        if classifier == '':
            forest_reg = RandomForestClassifier(random_state=42)
            forest_reg.fit(X_train, y_train)
            forest_reg_pred = forest_reg.predict(X_test)
            accuracy = accuracy_score(y_test, forest_reg_pred)
            pred = forest_reg.predict([[a_1, a_2, a_3, a_4,a_5,a_6,a_7,a_8,a_9]])
            if pred == [1]:
                # print('Random Classifier Accuracy is : ', ac)
                # print('Malignant Tumor')
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Random Forest Classifier', ac='Malignant Tumor', score=accuracy)
            else:
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Random Forest Classifier', ac='Benign Tumor',score=accuracy)
        elif classifier == 'KNN':
            neigh = KNeighborsClassifier(n_neighbors=21)
            neigh.fit(X_train, y_train)
            neigh_pred = neigh.predict(X_test)
            accuracy=accuracy_score(y_test, neigh_pred)
            pred = neigh.predict([[a_1, a_2, a_3, a_4,a_5,a_6,a_7,a_8,a_9]])
            if pred == [1]:
                # print('Random Classifier Accuracy is : ', ac)
                # print('Malignant Tumor')
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='KNN Classifier', ac='Malignant Tumor',score=accuracy)
            else:
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='KNN Classifier', ac='Benign Tumor', score=accuracy)

        elif classifier == 'SVM':
            svm_linear = SVC(kernel='linear')
            svm_linear.fit(X_train, y_train)
            svm_linear_pred = svm_linear.predict(X_test)
            accuracy = accuracy_score(y_test, svm_linear_pred)
            pred = svm_linear.predict([[a_1, a_2, a_3, a_4,a_5,a_6,a_7,a_8,a_9]])
            if pred == [1]:
                # print('Random Classifier Accuracy is : ', ac)
                # print('Malignant Tumor')
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='SVM Classifier', ac='Malignant Tumor',score=accuracy)
            else:
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='SVM Classifier', ac='Benign Tumor', score=accuracy)
        elif classifier == 'Decision Tree':
            dtc = tree.DecisionTreeClassifier()
            dtc = dtc.fit(X_train, y_train)
            dtc_pred = dtc.predict(X_test)
            accuracy=accuracy_score(y_test, dtc_pred)
            pred = dtc.predict([[a_1, a_2, a_3, a_4,a_5,a_6,a_7,a_8,a_9]])
            if pred == [1]:
                # print('Random Classifier Accuracy is : ', ac)
                # print('Malignant Tumor')
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Decision Tree Classifier', ac='Malignant Tumor',score=accuracy)
            else:
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Decision Tree Classifier', ac='Benign Tumor',score=accuracy)
        elif classifier == 'Naive Bayes':
            scaler = MinMaxScaler()
            X_minmax = scaler.fit_transform(X)
            mnb = MultinomialNB()
            X_train_MinMax, X_test_MinMax, y_train1, y_test1 = train_test_split(X_minmax, y, test_size=0.25,random_state=42)
            mnb.fit(X_train_MinMax, y_train1)
            mnb_pred = mnb.predict(X_test_MinMax)
            accuracy=accuracy_score(y_test1, mnb_pred)
            pred = mnb.predict([[a_1, a_2, a_3, a_4,a_5,a_6,a_7,a_8,a_9]])
            if pred == [1]:
                # print('Random Classifier Accuracy is : ', ac)
                # print('Malignant Tumor')
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Naive Bayes Classifier', ac='Malignant Tumor',score=accuracy)
            else:
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Naive Bayes Classifier', ac='Benign Tumor',score=accuracy)

        else:
            forest_reg = RandomForestClassifier(random_state=42)
            forest_reg.fit(X_train, y_train)
            forest_reg_pred = forest_reg.predict(X_test)
            accuracy = accuracy_score(y_test, forest_reg_pred)
            pred = forest_reg.predict([[a_1, a_2, a_3, a_4,a_5,a_6,a_7,a_8,a_9]])
            if pred == [1]:
                # print('Random Classifier Accuracy is : ', ac)
                # print('Malignant Tumor')
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Random Forest Classifier', ac='Malignant Tumor',score=accuracy)
            else:
                f = open("new data.txt", "a")
                f.write(a_1)
                f.write(',')
                f.write(a_2)
                f.write(',')
                f.write(a_3)
                f.write(',')
                f.write(a_4)
                f.write('\n')
                f.close()
                return render_template('index.html', msg='Random Forest Classifier', ac='Benign Tumor',score=accuracy)

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


app.run(debug=True)
