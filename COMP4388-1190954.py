import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#READ DATAFRAME
df = pd.read_csv('Customer Churn.csv')


#SHOW DATAFRAME INFO AND display table
df.info()
display(df.head(3500))

## CHANGE BINARY CATEGORES INTO TWO COLUMNS EACH
complainsDF = pd.get_dummies(df.Complains, prefix='Complains')
planDF = pd.get_dummies(df.Plan, prefix='Plan')
statusDF = pd.get_dummies(df.Status, prefix='Status')
churnDF = pd.get_dummies(df.Churn, prefix='Churn')
dfScaledCats = pd.concat([df, complainsDF, planDF, statusDF, churnDF], axis='columns')
df_final = dfScaledCats.drop(columns=['Plan', 'Status', 'Churn', 'Complains'])
display(df_final.describe())

## PLOT CHURN DISTRIBUTION
sns.displot(df['Churn'])

## PLOT CHURN DISTRIBUTION BY AGE GROUP
sns.countplot(data=df, x="Age Group", hue="Churn").set(xlabel='Age Group', ylabel='Churn Count')

## PLOT CHURN DISTRIBUTION BY CHARGE AMOUNT
sns.countplot(data=df, x="Charge Amount", hue="Churn").set(xlabel='Charge Amount', ylabel='Churn Count')

## CHARGE AMOUNT DETAILS
sns.histplot(data=df, x="Charge Amount").set(xlabel='Charge Amount', ylabel='Customer Count')
charge=df[["Charge Amount"]]
(charge.describe())

## SHOW CORRELATION
dfScaledCats.corr()

## NORMALIZATION AND SPLITTING THE DATA
df_norm= MinMaxScaler().fit(df_final)
df_final_normalized = df_norm.transform(df_final)
df_final_normalized = pd.DataFrame(df_final_normalized, columns=df_final.columns)

display(df_final_normalized.describe())

x= df_final_normalized.drop(columns=['Customer Value'])
y= df_final_normalized[["Customer Value"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
#X_train.plot.box(figsize=(20,5), rot=90)

X_train.plot.box(figsize=(20,5), rot=90)
y_train.describe()

## LINEAR REGRESSION
regr = linear_model.LinearRegression()
LRM1 = regr.fit(X_train, y_train)
y_pred_LRM1 = regr.predict(X_test)
mse = []
mae = []
re = []
mse.append(mean_squared_error(y_test, y_pred_LRM1))
mae.append(mean_absolute_error(y_test, y_pred_LRM1))
re.append(r2_score(y_test, y_pred_LRM1))

LRM2 = regr.fit(X_train[['Subscription Length', 'Freq. of use', 'Status_active']], y_train)
y_pred_LRM2 = regr.predict(X_test[['Subscription Length', 'Freq. of use', 'Status_active']])
mse.append(mean_squared_error(y_test, y_pred_LRM2))
mae.append(mean_absolute_error(y_test, y_pred_LRM2))
re.append(r2_score(y_test, y_pred_LRM2))

LRM3 = regr.fit(X_train[['Freq. of SMS', 'Freq. of use', 'Seconds of Use']], y_train)
y_pred_LRM3 = regr.predict(X_test[['Freq. of SMS', 'Freq. of use', 'Seconds of Use']])
mse.append(mean_squared_error(y_test, y_pred_LRM3))
mae.append(mean_absolute_error(y_test, y_pred_LRM3))
re.append(r2_score(y_test, y_pred_LRM3))

df_svr = pd.DataFrame({'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'R^2 Score': re},
                      index=['LRM1', 'LRM2', 'LRM3'])
df_svr

## CLASSIFICATION KNN, NAIVE BAYES, LOGISTIC REGRESSION
df_norm= MinMaxScaler().fit(df_final)
df_final_normalized = df_norm.transform(df_final)
df_final_normalized = pd.DataFrame(df_final_normalized, columns=df_final.columns)
x= df_final_normalized.drop(columns=['Churn_yes','Churn_no'])
y= df_final_normalized[["Churn_yes"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
scores = []

### KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_kNN = roc_auc_score(y_test, y_pred, multi_class='ovr')
scores.append(accuracy_kNN)
confusion_mat = confusion_matrix(y_test, y_pred)
confMat = pd.DataFrame(confusion_mat, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
display(confMat)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve kNN')
plt.show()

### NAIVE BAYES
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_NB = roc_auc_score(y_test, y_pred, multi_class='ovr')
scores.append(accuracy_NB)
confusion_mat = confusion_matrix(y_test, y_pred)
confMat = pd.DataFrame(confusion_mat, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
display(confMat)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Naive Bayes')
plt.show()

### LOGISTIC REGRESSION
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_LR = roc_auc_score(y_test, y_pred, multi_class='ovr')
scores.append(accuracy_LR)
confusion_mat = confusion_matrix(y_test, y_pred)
confMat = pd.DataFrame(confusion_mat, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
display(confMat)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression')
plt.show()
df_acc = pd.DataFrame({'ROC_AUC_SCORE': scores}, index=['kNN', 'Naive Bayes', 'Logistic Regression'])
df_acc


