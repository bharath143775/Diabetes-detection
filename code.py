# importing the libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the dataset:
data = pd.read_csv("https://raw.githubusercontent.com/AP-Skill-Development-Corporation/Tirumala-ML/main/Day-5/diabetes.csv")

#read top 5 rows
data.head()

#info of dataframe
data.info()

#check for null values
data.isnull().sum()

#check statistics of data
data.describe()

# splitting of dataset:separating the input and output values.
Features_col =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[Features_col]
y = data["Outcome"]

# training and testing the model:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

# fit the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

#predict the values and check predicted values
y_predict = lr.predict(X_test)
y_predict

#  checking the accuracy of the model and print matrix
from sklearn import metrics
c_matrix = metrics.confusion_matrix(y_test,y_predict)
print("Confusion Matrix: ",c_matrix)

# check scores accuracy,precision,recall
print("Accuracy :",metrics.accuracy_score(y_test,y_predict))
print("precision:",metrics.precision_score(y_test,y_predict))
print("recall:",metrics.recall_score(y_test,y_predict))

# plot graph of regression
y_pre = lr.predict_proba(X_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test,y_pre)
a = metrics.roc_auc_score(y_test,y_pre)
plt.plot(fpr,tpr,label = "data,auc ="+str(a))
plt.legend(loc= "best")
plt.show()


# plot for confusion matrix
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(c_matrix),annot = True,cmap = "Purples",fmt = 'g')
ax.xaxis.set_label_position("top")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("confusion matrix")
plt.show()
