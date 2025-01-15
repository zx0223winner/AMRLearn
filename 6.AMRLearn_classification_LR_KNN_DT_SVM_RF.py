##*************************************************************************##
##          Step9. machine learning classification method                  ##          
##*************************************************************************##
# pip3 install -U scikit-learn scipy matplotlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import preprocessing

#while read line; do python3 9.AMR_Learn_classification_LR_KNN_DT_SVM_RF.py feature2target_processing.txt $line;done <list.txt
# list.txt refer to antibiotics name, which can do a batch job

import sys
if len(sys.argv)!=3: #if the input arguments not 4, showing the usage.
    print("Usage:python3 AMRLearn_classification_LR_KNN_DT_SVM_RF.py <feature2target_processing.txt> <Cefotaxime>\n; e.g., 'Spectinomycin','Cefotaxime','Ceftazidime'")
    sys.exit()

os.system('mkdir '+ sys.argv[2])

#writing out the log file
f = open(sys.argv[2]+'/'+sys.argv[2]+'.log', 'w')
sys.stdout = f

# loading data
#hsd_data = pd.read_csv(sys.argv[1],sep='\t').fillna(0) #empty lines, so should have fillna()


hsd_data = pd.read_csv(sys.argv[1],sep='\t').fillna(0) #empty lines, so should have fillna()


#print(hsd_data.head())

#hsd_data = hsd_data.set_index('locus_tag')
# hsd_data.reset_index()

# AUC values are interpreted as follows: 0.5-0.6 (failed), 0.6-0.7 (worthless), 0.7-0.8 (poor), 0.8-0.9 (good), > 0.9 (excellent).

#predictors = hsd_data.drop(['Spectinomycin','Lincomycin','Florfenicol','chloramphenicol','Doxycycline','Cefotaxime','Ceftazidime','Ciprofloxacin','Ofloxacin','Polymyxin_B','Erythromycin','Rifampin','Trimethoprim','amikacin','Tetracycline','locus_tag'], axis=1).values
#sys.argv[2] = the one you want to remove from next line.
#manual correction
#names = hsd_data.drop(['Spectinomycin','Lincomycin','Florfenicol','chloramphenicol','Doxycycline','Cefotaxime','Ceftazidime','Ciprofloxacin','Ofloxacin','Polymyxin_B','Erythromycin','Rifampin','Trimethoprim','amikacin','Tetracycline','locus_tag'], axis=1).columns



# loading data
#hsd_data = pd.read_csv(sys.argv[1],sep='\t')
#print(hsd_data.head())

# creating features and arget arrays
#hsd_data = hsd_data.set_index('index')
# print(hsd_data.head())
#X = hsd_data.drop('hsd', axis=1).values

# Annotate this line to switch the antibiotics types from R,S to R,I,S
hsd_data = hsd_data[~(hsd_data[sys.argv[2]]==0)]

X = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).values

y = hsd_data[sys.argv[2]].values
#y = np.argmax(y, axis=1)


print("Preprocessing: check the class imbrance of the target y, and then make stratified sampling for y")
print(hsd_data[sys.argv[2]].value_counts())
print("\n\n")
print("Preprocessing: check the variance of the features X, and then make scale for x (features)")
print(pd.DataFrame(X).var().head())
print("\n\n")


#fitting a regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state=42) 

#scale the data, preprocessing
# Note: only preprocess the X_train to be trained for the model, otherwise it is called data leakage
# using the fit_transform method, and preprocess the test features using the transform method. Using the transform method means that the test features won't be used to fit the model and avoids data leakage. 
scaler = preprocessing.StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train) # or X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
print("X_train variance")
print(pd.DataFrame(X_train).var().head())
print("X_test variance")
print(pd.DataFrame(X_test).var().head())
print("\n\n")
#X = scale(X)

print("Supervised classification models\n")
print("1.LogisticRegression model\n")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print("Training data:" + str(logreg.score(X_train,y_train)))
print("Testing data: "+ str(logreg.score(X_test,y_test))+"\n")
print("# Overfitting: the model/estimator/classifier for the training data is too complex, making the training data high accuracy, but low accuracy in test data."+'\n'+"# Underfitting: the training data is too simple, making the training lower accuracy, but high accuracy in test data.")
cv_results = cross_val_score(logreg, X, y, cv=3)
print("5-fold cross validation" + str(cv_results)+"\n")
# print(np.mean(cv_results))
# np.vectorize(cv_results)
print("\n\n\n"+"zxwinner_LogisticRegression model:  "+ str(np.mean(cv_results))+"\n\n\n")

print(classification_report(y_test, y_pred))

# https://www.kaggle.com/discussions/questions-and-answers/146653
#if Multiclass case:
y_pred_prob = logreg.predict_proba(X_test)
print("Multiclass case: 1.LogisticRegression model AUC: "+ str(roc_auc_score(y_test, y_pred, multi_class='ovr'))+"\n") ##y_pred_prob
print("balanced accuracy score: " + str(balanced_accuracy_score(y_test, y_pred))+"\n\n")



print("2.k-nearest neighbors model\n")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Training data:" + str(knn.score(X_train,y_train)))
print("Testing data: "+ str(knn.score(X_test,y_test)))
cv_results = cross_val_score(knn, X, y, cv=3)
print("5-fold cross validation" + str(cv_results)+"\n")

print("2.1 tuning the model to select best hyperparameters\n")
# ROC curve/ receiver operating characteristic curve
# Area under the ROC curve (AUC)
# larger area under the ROC curve = better model
param_grid = {'n_neighbors': np.arange(1, 50)}
knn_cv = GridSearchCV(knn, param_grid, cv=3)
knn_cv.fit(X_train, y_train)
print("best_params:" + str(knn_cv.best_params_))
print("best_score:"+ str(knn_cv.best_score_)+"\n")

print("2.2 rerun the model\n")
knn_new = KNeighborsClassifier(n_neighbors=5)
knn_new.fit(X_train,y_train)
y_pred = knn_new.predict(X_test)
y_pred_proba = knn_new.predict_proba(X_test)
print("Training data:" + str(knn_new.score(X_train,y_train)))
print("Testing data: "+ str(knn_new.score(X_test,y_test)))
cv_results = cross_val_score(knn_new, X, y, cv=3)
print("\n\n\n"+"zxwinner_k-nearest neighbors model:  "+ str(np.mean(cv_results))+"\n\n\n")
print("5-fold cross validation" + str(cv_results)+"\n")

print(classification_report(y_test, y_pred))

auc = round(metrics.roc_auc_score(y_test, y_pred,multi_class='ovr'), 4) #y_pred_proba
print("Multiclass case: 2.k-nearest neighbors model AUC: "+ str(auc)+"\n")
print("balanced accuracy score: " + str(balanced_accuracy_score(y_test, y_pred))+"\n\n")



print("3.DecisionTreeClassifier model\n")
# decision trees
clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
clf.fit(X_train, y_train)
# Predict for 1 observation
#clf.predict(X_test.iloc[0].values.reshape(1, -1)) 
# Predict for multiple observations
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
print("Training data:" + str(clf.score(X_train,y_train)))
print("Testing data: "+ str(clf.score(X_test,y_test)))
cv_results = cross_val_score(clf, X, y, cv=3)
print("5-fold cross validation" + str(cv_results)+"\n")


print("3.1 tuning the model to select best hyperparameters\n")
#decision tree, max_depth
# List of values to try for max_depth:
max_depth_range = list(range(1, 10))
# List to store the average RMSE for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth = depth,
    random_state = 0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    #accuracy.append(score)
    print("Max_depth:"+str(depth)+"     "+"Score:"+str(score)+"\n")

print("3.2 rerun the model\n")
# decision trees
clf_4 = DecisionTreeClassifier(max_depth = 4, random_state = 0)
clf_4.fit(X_train, y_train)
# Predict for 1 observation
#clf.predict(X_test.iloc[0].values.reshape(1, -1)) 
# Predict for multiple observations
y_pred = clf_4.predict(X_test)
print("Training data:" + str(clf_4.score(X_train,y_train)))
print("Testing data: "+ str(clf_4.score(X_test,y_test)))
cv_results = cross_val_score(clf_4, X, y, cv=3)

print("\n\n\n"+"zxwinner_DecisionTreeClassifier model:  "+ str(np.mean(cv_results))+"\n\n\n")
print("5-fold cross validation" + str(cv_results)+"\n")

print(classification_report(y_test, y_pred))

auc = round(metrics.roc_auc_score(y_test, y_pred,multi_class='ovr'), 4) # y_pred_proba
print("Multiclass case: 3.DecisionTreeClassifier model AUC: "+ str(auc)+"\n")

print("balanced accuracy score: " + str(balanced_accuracy_score(y_test, y_pred))+"\n\n") 

#y_test = np.argmax(y_test, axis=0)
#y_pred = np.argmax(y_pred, axis=0)
#auc = round(metrics.roc_auc_score(y_test, y_pred,multi_class='ovr'), 4)
#print("auc : "+str(auc))


print("4.SVM support vector machine\n")
svc = SVC(probability=True)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
y_pred_proba = svc.predict_proba(X_test)
print("Training data:" + str(svc.score(X_train,y_train)))
print("Testing data: "+ str(svc.score(X_test,y_test)))
cv_results = cross_val_score(svc, X, y, cv=3)

print("\n\n\n"+"zxwinner_SVM support vector machine:  "+ str(np.mean(cv_results))+"\n\n\n")
print("5-fold cross validation" + str(cv_results)+"\n")
#print(y_pred[:10])

print(classification_report(y_test, y_pred))
auc = round(metrics.roc_auc_score(y_test, y_pred, multi_class='ovr'), 4) # y_pred_proba
print("Multiclass case: 4.SVM support vector machine AUC: "+ str(auc)+"\n")

print("balanced accuracy score: " + str(balanced_accuracy_score(y_test, y_pred))+"\n\n")

# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# print("auc : "+str(auc))


print("5.RF Random forest\n")
clf = RandomForestClassifier(max_depth = 6, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
print("Training data:" + str(clf.score(X_train,y_train)))
print("Testing data: "+ str(clf.score(X_test,y_test)))
cv_results = cross_val_score(clf, X, y, cv=3)

print("\n\n\n"+"zxwinner_RF Random forest:  "+ str(np.mean(cv_results))+"\n\n\n")
print("5-fold cross validation" + str(cv_results)+"\n")

auc = round(metrics.roc_auc_score(y_test, y_pred,multi_class='ovr'), 4) # y_pred_proba
print("Multiclass case: 5.RF Random forest AUC: "+ str(auc)+"\n")

print("balanced accuracy score: " + str(balanced_accuracy_score(y_test, y_pred))+"\n\n")



print("3 performance evaluation\n")

# metrics from the confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# print("auc : "+str(auc))
# ROC curve/ receiver operating characteristic curve
# Area under the ROC curve (AUC)
# larger area under the ROC curve = better model


y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)  # y_pred_prob
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
# 
y_pred_prob = logreg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred))# y_pred_prob

print("4 multiple ROC performance evaluation\n")
#set up plotting area
#plt.figure(0).clf()



# fit logistic regression model and plot ROC curve
# reg = LogisticRegression()
# reg.fit(X_train, y_train)
# y_pred = reg.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))


#fit model and plot ROC curve
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="K-nearest neighbors, AUC="+str(auc))

#fit decision trees model and plot ROC curve
clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="decision trees, AUC="+str(auc))

#fit Random forest model and plot ROC curve
clf = RandomForestClassifier(max_depth = 6, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Random forest, AUC="+str(auc))


#fit support vector machine model and plot ROC curve
svc = SVC(probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="support vector machine, AUC="+str(auc))

#add legend
plt.legend()

plt.savefig(sys.argv[2]+".png")
f.close()
#plt.show()

# add legend
# plt.legend()

