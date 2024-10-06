##*************************************************************************##
##          Step4. linear regression models of the AMR data analysis       ##          
##*************************************************************************##


# pip3 install -U scikit-learn scipy matplotlib
#while read line; do python3 4.AMR_Learn.py feature2target.txt $line;done <list.txt

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn import metrics

import sys
if len(sys.argv)!=3: #if the input arguments not 4, showing the usage.
    print("Usage:python3 AMR_Learn_linear.py <feature2target.txt> <name of antibiotics>\n\n e.g.,python3 AMR_Learn_linear.py feature2target.txt Spectinomycin \n\n more antibiotics to try: 'Spectinomycin','Lincomycin','Florfenicol','chloramphenicol','Doxycycline','Tigecycline','Cefotaxime','Ceftazidime','Ciprofloxacin','Ofloxacin','Polymyxin_B','Erythromycin','Rifampin','Trimethoprim','amikacin','Tetracycline',")
    sys.exit()

os.system('mkdir '+ sys.argv[2])

#writing out the log file
f = open(sys.argv[2]+'/'+sys.argv[2]+'.log', 'w')
sys.stdout = f

# loading data
hsd_data = pd.read_csv(sys.argv[1],sep='\t').fillna(0) #empty lines, so should have fillna()

#print(hsd_data.head())

# creating features and arget arrays
#hsd_data = hsd_data.set_index('index')
# print(hsd_data.head())
X = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).values
#sys.argv[2] = the one you want to remove from next line.
#manual correction
names = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).columns
#X = X[:, :2]
y = hsd_data[sys.argv[2]].values

#scale the data, preprocessing
X = scale(X)
#print(X)

#fitting a regression model
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=42)

# print("Supervised classification models\n")
# print("1.LogisticRegression model\n")
# logreg = LogisticRegression()
# logreg.fit(X_train,y_train)
# y_pred = logreg.predict(X_test)
# print("Training data:" + str(logreg.score(X_train,y_train)))
# print("Testing data: "+ str(logreg.score(X_test,y_test))+"\n")
# print("# Overfitting: the model/estimator/classifier for the training data is too complex, making the training data high accuracy, but low accuracy in test data."+'\n'+"# Underfitting: the training data is too simple, making the training lower accuracy, but high accuracy in test data.")
# cv_results = cross_val_score(logreg, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")

# print("2.k-nearest neighbors model\n")
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# print("Training data:" + str(knn.score(X_train,y_train)))
# print("Testing data: "+ str(knn.score(X_test,y_test)))
# cv_results = cross_val_score(knn, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")

# print("2.1 tuning the model to select best hyperparameters\n")
# # ROC curve/ receiver operating characteristic curve
# # Area under the ROC curve (AUC)
# # larger area under the ROC curve = better model
# param_grid = {'n_neighbors': np.arange(1, 50)}
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# knn_cv.fit(X_train, y_train)
# print("best_params:" + str(knn_cv.best_params_))
# print("best_score:"+ str(knn_cv.best_score_)+"\n")
# 
# print("2.2 rerun the model\n")
# knn_new = KNeighborsClassifier(n_neighbors=11)
# knn_new.fit(X_train,y_train)
# y_pred = knn_new.predict(X_test)
# print("Training data:" + str(knn_new.score(X_train,y_train)))
# print("Testing data: "+ str(knn_new.score(X_test,y_test)))
# cv_results = cross_val_score(knn_new, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")
# 
# print("3.DecisionTreeClassifier model\n")
# # decision trees
# clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
# clf.fit(X_train, y_train)
# # Predict for 1 observation
# #clf.predict(X_test.iloc[0].values.reshape(1, -1)) 
# # Predict for multiple observations
# y_pred = clf.predict(X_test)
# print("Training data:" + str(clf.score(X_train,y_train)))
# print("Testing data: "+ str(clf.score(X_test,y_test)))
# cv_results = cross_val_score(clf, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")
# 
# print("3.1 tuning the model to select best hyperparameters\n")
# #decision tree, max_depth
# # List of values to try for max_depth:
# max_depth_range = list(range(1, 10))
# # List to store the average RMSE for each value of max_depth:
# accuracy = []
# for depth in max_depth_range:
#     clf = DecisionTreeClassifier(max_depth = depth,
#     random_state = 0)
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     #accuracy.append(score)
#     print("Max_depth:"+str(depth)+"     "+"Score:"+str(score)+"\n")
# 
# print("3.2 rerun the model\n")
# # decision trees
# clf_4 = DecisionTreeClassifier(max_depth = 4, random_state = 0)
# clf_4.fit(X_train, y_train)
# # Predict for 1 observation
# #clf.predict(X_test.iloc[0].values.reshape(1, -1)) 
# # Predict for multiple observations
# y_pred = clf_4.predict(X_test)
# print("Training data:" + str(clf_4.score(X_train,y_train)))
# print("Testing data: "+ str(clf_4.score(X_test,y_test)))
# cv_results = cross_val_score(clf_4, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")
# 
# print("4.SVM support vector machine\n")
# svc = SVC()
# svc.fit(X_train,y_train)
# y_pred = svc.predict(X_test)
# print("Training data:" + str(svc.score(X_train,y_train)))
# print("Testing data: "+ str(svc.score(X_test,y_test)))
# cv_results = cross_val_score(svc, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")
# 
# #print(y_pred[:10])
# 
# print("5.RF Random forest\n")
# clf = RandomForestClassifier(max_depth = 6, random_state = 0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Training data:" + str(clf.score(X_train,y_train)))
# print("Testing data: "+ str(clf.score(X_test,y_test)))
# cv_results = cross_val_score(clf, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")



print(" Supervised regression models\n")
print("1.Linear regression model\n")
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Training data:" + str(reg.score(X_train,y_train)))
print("Testing data: "+ str(reg.score(X_test,y_test)))
cv_results = cross_val_score(reg, X, y, cv=5)
print("5-fold cross validation" + str(cv_results)+"\n")
avg_score = sum(cv_results)/5
print("Average score: " + str(avg_score))

#print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

print("2. Ridge Linear regression model\n")
ridge = Ridge(alpha = 0.1)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
print("Training data:" + str(ridge.score(X_train,y_train)))
print("Testing data: "+ str(ridge.score(X_test,y_test)))
cv_results = cross_val_score(ridge, X, y, cv=5)
print("5-fold cross validation" + str(cv_results)+"\n")

#avg_score = sum(cv_result)/5
#print("Average score: " + str(avg_score))

#print(confusion_matrix(y_test, ridge_pred))
# print(classification_report(y_test, ridge_pred))
# 
print("3.Lasso Linear regression model\n")
lasso = Lasso(alpha=0.1,tol=1e-5) #, normalize=True
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
print("Training data:" + str(lasso.score(X_train,y_train)))
print("Testing data: "+ str(lasso.score(X_test,y_test)))
cv_results = cross_val_score(lasso, X, y, cv=5)
print("5-fold cross validation" + str(cv_results)+"\n")
avg_score = sum(cv_results)/5
print("Average score: " + str(avg_score))


# metrics from the confusion matrix
#print(confusion_matrix(y_test, lasso_pred))
# print(classification_report(y_test, lasso_pred))

#
# print(" Coefficient for all three Linear regression model\n")
# # reg_coef[reg_coef >0]
# reg_coef = reg.fit(X, y).coef_
# select_names = []
# select_coef = []
# for i in range(len(reg_coef)):
#     if abs(reg_coef[i]) > 0:
#         select_names.append(names[i])
#         select_coef.append(reg_coef[i])
# print(select_coef)
# _ = plt.plot(range(len(select_names)), select_coef)
# _ = plt.xticks(range(len(select_names)), select_names, rotation=40)
# _ = plt.ylabel('Linear regression coefficients')
# #plt.show()


outfile=open(sys.argv[2]+"_ridge_coef.txt",'w')
ridge_coef = ridge.fit(X, y).coef_
select_names = []
select_coef = []
for i in range(len(ridge_coef)):
# 0.1 is the threshold to view coefficients, users can relax the thresholds to view more.
    if abs(ridge_coef[i]) > 0.1: 
        select_names.append(names[i])
        select_coef.append(ridge_coef[i])
        outfile.write(str(names[i])+"\t"+str(ridge_coef[i])+"\n")
outfile.close()  # close file is very important!!!!
os.system("python3 coef2gene.py gene_location_info.txt " + sys.argv[2]+"_ridge_coef.txt " + sys.argv[2]+'/'+ sys.argv[2]+"_ridge_coef_out.txt" )
os.system("rm "+sys.argv[2]+"_ridge_coef.txt ")

_ = plt.plot(range(len(select_names)), select_coef)
_ = plt.xticks(range(len(select_names)), select_names, rotation=90, fontsize=5)
_ = plt.ylabel('Ridge regression coefficients')
#plt.show()
plt.savefig(sys.argv[2]+'/'+sys.argv[2]+"_ridge"+".png")
plt.clf() # clear the previous figure


outfile2=open(sys.argv[2]+"_lasso_coef.txt",'w')
lasso_coef = lasso.fit(X, y).coef_
select_names = []
select_coef = []
for i in range(len(lasso_coef)):
    if abs(lasso_coef[i]) > 0.1:
        select_names.append(names[i])
        select_coef.append(lasso_coef[i])
        outfile2.write(str(names[i])+"\t"+str(lasso_coef[i])+"\n")
outfile2.close()
os.system("python3 coef2gene.py gene_location_info.txt " + sys.argv[2] +"_lasso_coef.txt " + sys.argv[2]+'/'+sys.argv[2]+"_lasso_coef_out.txt" )
os.system("rm "+sys.argv[2]+"_lasso_coef.txt ")

_ = plt.plot(range(len(select_names)), select_coef)
_ = plt.xticks(range(len(select_names)), select_names, rotation=90, fontsize=5)
_ = plt.ylabel('Lasso regression coefficients')
#plt.show()
plt.savefig(sys.argv[2]+'/'+sys.argv[2]+"_lasso"+".png")

f.close()
# 
# print("2.SVM support vector machine lineaer model\n")
# svc = SVC(kernel='linear',probability=True)
# svc.fit(X_train,y_train)
# y_svc_pred = svc.predict(X_test)
# print("Training data:" + str(svc.score(X_train,y_train)))
# print("Testing data: "+ str(svc.score(X_test,y_test)))
# cv_results = cross_val_score(svc, X, y, cv=5)
# print("5-fold cross validation" + str(cv_results)+"\n")
# 
# #print(confusion_matrix(y_test, y_svc_pred))
# # print(classification_report(y_test, y_svc_pred))
# # 
# svc_coef = lasso.fit(X, y).coef_
# print(svc_coef)
# _ = plt.plot(range(len(names)), lasso_coef)
# _ = plt.xticks(range(len(names)), names, rotation=90)
# _ = plt.ylabel('Support vector machine lineaer coefficients')
# 
# #plt.show()
# # 
# print("3 performance evaluation\n")

# metrics from the confusion matrix

#ROC curve/ receiver operating characteristic curve
#Area under the ROC curve (AUC)
#larger area under the ROC curve = better model
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Logistic Regression ROC Curve')
# plt.show();
# # 
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# print(roc_auc_score(y_test, y_pred_prob))
# 
# print("4 multiple ROC performance evaluation\n")
# #set up plotting area
# plt.figure(0).clf()



# fit logistic regression model and plot ROC curve
# reg = LogisticRegression()
# reg.fit(X_train, y_train)
# y_pred = reg.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))
# 
# fit model and plot ROC curve
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="K-nearest neighbors, AUC="+str(auc))
# 
# fit decision trees model and plot ROC curve
# clf = DecisionTreeClassifier(max_depth = 6, random_state = 0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="decision trees, AUC="+str(auc))
# 
# fit Random forest model and plot ROC curve
# clf = RandomForestClassifier(max_depth = 6, random_state = 0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="Random forest, AUC="+str(auc))
#

# reg2 = LinearRegression()
# reg2.fit(X_train, y_train)
# y_pred = reg2.predict(X_test)
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="Linear Regression, AUC="+str(auc))
# 
# 
# 
# y_pred_prob = reg2.predict(X_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Linear Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Linear Regression ROC Curve')
# plt.show();
# # 
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# print(roc_auc_score(y_test, y_pred_prob))
# 
# print("4 multiple ROC performance evaluation\n")
# #set up plotting area
# plt.figure(0).clf()

#fit support vector machine model and plot ROC curve
# svc = SVC(probability=True)
# svc.fit(X_train, y_train)
# y_pred = svc.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="support vector machine, AUC="+str(auc))
# 
# #If you are in a regression setting, just replace predict_proba with predict.
# # If you are in a classification setting, you cannot use linear regression - try logistic regression instead 
# #fit Linear regression model and plot ROC curve


# reg3 = Ridge(alpha = 0.1, normalize = True)
# reg3.fit(X_train, y_train)
# y_pred = reg3.predict(X_test)
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="Linear Regression, AUC="+str(auc))
# 
# reg4 = Lasso(alpha=0.1, normalize=True)
# reg4.fit(X_train, y_train)
# y_pred = reg4.predict(X_test)
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
# auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
# plt.plot(fpr,tpr,label="Linear Regression, AUC="+str(auc))


#add legend
# plt.legend()
# plt.show();
