# Apply classification algorithms - logistic regression, Naive Bayes, SVM and KNN on following dataset
# and make a table in word document comparing the evaluation metrics.

#import file and convert data
import pandas as pd
import pandas_profiling as pp
import numpy as np
df = pd.read_csv("heart.csv")

#check data import
print(df.columns)

#get information on dataset
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.describe(include=['object']))

#run profile report to see what data needs cleaning
#profile = pp.ProfileReport(df)
#profile.to_file("heart_EdA.html")
# 5 categorical values need converting
#(Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
# resting BP and cholesterol need 0 values converting

#tidy data up
#calculate mean values for resting BP and cholesterol
mean_Cholesterol = df["Cholesterol"].mean()
mean_RestingBP = df["RestingBP"].mean()
print("\n\n--------------------------------------------------------------------------")
print("average Cholesterol ",mean_Cholesterol)
print("average RestingBP ",mean_RestingBP)
#replcae 0 values in resting BP and cholesterol to mean values
df["Cholesterol"] = df["Cholesterol"].replace({0:mean_Cholesterol})
df["RestingBP"] = df["RestingBP"].replace({0:mean_RestingBP})
print(df.info())
#convert objects to numerical values for data modeling
#start with sex - only 2 possible values, can be handled with data replacement
#assign M to 1 and F to 2
df["ExerciseAngina"] = df["ExerciseAngina"].replace({"Y":"1"})
df["ExerciseAngina"] = df["ExerciseAngina"].replace({"N":"0"})
df["ExerciseAngina"] = df["ExerciseAngina"].astype(int)
df["Sex"] = df["Sex"].replace({"M":"1"})
df["Sex"] = df["Sex"].replace({"F":"2"})
df["Sex"] = df["Sex"].astype(int)

#check whether replacement worked
print(df.head())

#convert categorical data into numerical by creating dummy variables
df = pd.get_dummies(df,columns=["ChestPainType","RestingECG","ST_Slope"])
print("\n\n--------------------------------------------------------------------------")
print("after data conversion")
print(df.info())

#setup datasets for testing
y = df["HeartDisease"]
X = df.drop("HeartDisease",axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.30,random_state=0)
#Support Vector Machine modelling
from sklearn.svm import SVC
model_SVC = SVC(kernel="linear")
model_SVC.fit(X_train,y_train)
y_pred_SVC = model_SVC.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cnfn_SVC = confusion_matrix(y_test,y_pred_SVC)
cr_SVC = classification_report(y_test,y_pred_SVC)
acc_SVC = accuracy_score(y_test, y_pred_SVC)

print("\n\n--------------------------------------------------------------------------")
print("Support Vector Machine algorithm (Linear)")
print(cnfn_SVC)
print(cr_SVC)
print("Accuracy of SVM is: ",acc_SVC)
print("\n\n--------------------------------------------------------------------------")

#Naive Bayes modelling
from sklearn.naive_bayes import GaussianNB
model_NB = GaussianNB()
model_NB.fit(X_train,y_train)
y_pred_NB = model_NB.predict(X_test)
cnfn_NB = confusion_matrix(y_test,y_pred_NB)
cr_NB = classification_report(y_test,y_pred_NB)
acc_NB = accuracy_score(y_test, y_pred_NB)

print("\n\n--------------------------------------------------------------------------")
print("Naive Bayes (Gaussian)")
print(cnfn_NB)
print(cr_NB)
print("Accuracy of Naive Bayes is: ",acc_NB)
print("\n\n--------------------------------------------------------------------------")

#K nearest neighbours modelling
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=10) #experiment with different k values - in this case dataset is clean so doesn't really change
model_knn.fit(X_train,y_train)
y_pred_knn = model_knn.predict(X_test)
cnfn_knn = confusion_matrix(y_test,y_pred_knn)
cr_knn = classification_report(y_test,y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\n\n--------------------------------------------------------------------------")
print("K nearest neighbours")
print(cnfn_knn)
print(cr_knn)
print("Accuracy of K nearest neighbours is: ",acc_knn)
print("\n\n--------------------------------------------------------------------------")

from sklearn.linear_model import LogisticRegression
model_logr = LogisticRegression()
model_logr.fit(X_train,y_train)
y_pred_logr = model_logr.predict(X_test)
cnfn_logr = confusion_matrix(y_test,y_pred_logr)
ac_logr = accuracy_score(y_test,y_pred_logr)
cr_logr = classification_report(y_test,y_pred_logr)

print("\n\n--------------------------------------------------------------------------")
print("Logistic Regression")
print(cnfn_logr)
print(cr_logr)
print("Accuracy of Logistic Regression is: ",ac_logr)
print("\n\n--------------------------------------------------------------------------")



from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
#confusion matrix and classification report won't work here because the results are float not integer
#use MAE and RMSE instead
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse = mean_squared_error(y_test, y_pred_lr)
import math
rmse = math.sqrt(mse) #use square root function from math
mae = mean_absolute_error((y_test), y_pred_lr)

print("\n\n--------------------------------------------------------------------------")
print("Linear Regression")
print("RMSE is: ",rmse)
print("MAE is: ",mae)
print("\n\n--------------------------------------------------------------------------")

from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier(n_estimators=25)
model_rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)
cnfn_rfc = confusion_matrix(y_test,y_pred_rfc)
cr_rfc = classification_report(y_test,y_pred_rfc)
acc_rfc = accuracy_score(y_test, y_pred_rfc)

print("\n\n--------------------------------------------------------------------------")
print("Random Forest Classifier")
print(cnfn_rfc)
print(cr_rfc)
print("Accuracy of Random Forest Classifier is: ",acc_rfc)
print("\n\n--------------------------------------------------------------------------")

from sklearn.tree import DecisionTreeClassifier
model_dtc = DecisionTreeClassifier(criterion="entropy")
model_dtc.fit(X_train,y_train)
y_pred_dtc = model_dtc.predict(X_test)
cnfn_dtc = confusion_matrix(y_test,y_pred_dtc)
ac_dtc = accuracy_score(y_test,y_pred_dtc)
cr_dtc = classification_report(y_test,y_pred_dtc)

print("\n\n--------------------------------------------------------------------------")
print("Decision Tree")
print(cnfn_dtc)
print(cr_dtc)
print("Accuracy of Decision Tree is: ",ac_dtc)
print("\n\n--------------------------------------------------------------------------")