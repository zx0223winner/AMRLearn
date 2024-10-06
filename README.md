# AMR_Learn new

### 1. AMRLearn -- Pipeline for using supervised machine learning algorithm to characterize antimicrobial resistance associated with the SNPs within host evolution

### 2. What's AMRLearn?

>Explore the genome diversity and specificity of E.coli genome  to uncover Antimicrobial resistance(AMR) within-host evolution. Focus on a smaller number of samples to explore the potential for high-quality de novo assembly, Get a sense of how much diversity there is between the E. coli strains and publicly available E. coli genomes.

>The SNPs for each gene of 40 strains were calculated under different antibiotics, such as Lincomycin, Spectinomycin, Florfenicol, chloramphenicol, Doxycycline, Tigecycline, Ciprofloxacin, Ofloxacin, Cefotaxime, Ceftazidime, Polymyxin_B, Erythromycin, Rifampin, Trimethoprim. 

>Based on supervised machine learning algorithms (Linear regression, Ridge regression, Lasso regression, Support Vector Machine(SVM)), four different linear regression models were used to build and train SNPs as independent variables in 40 E.coli strains.

>Two of the machine learning models(Ridge and Lasso) were selected to extract features and coefficients for each strain. 
Those genes who contribute the antibiotic resistance most (e.g.,under each of the vaccine treatment) were selected and ranked. 

>Next step, those genes will be furtherly explored to look into the within-host evolution.​

2.1 generate the lcoation info from genbank file
```python3

# must clean the plasmid before running the genbank file.
# 1.gbff2tab.py
>Usage:python3 1.gbff2tab.py <gene_bank_file> <gene_output_file> 
#Example: python3 1.gbff2tab.py NZ_CP053080.gbff gene_location_info.txt")
```

2.2 generate snps count table from Parsnp result
```python3
# 2.vcf2snp.py
>Usage:python3 2.vcf2snp.py <gene location info> <parsnp vcf file> <output file>
#Example: python3 2.vcf2snp.py gene_location_info.txt parsnp.ggr.vcf vcf_snp_count.txt
```

2.3 generate the table for the regression coffeficents
```python3
# 3.feature2target.py
>Usage:python3 3.feature2target.py vcf_snp_count.txt Antibiotics_test.txt feature2target.txt
```
2.4 the main machine learning scripts
```python3
# 4.AMRLearn.py
>Usage:python3 4.AMRLearn.py <output_file_name> <species name>
#Example: python3 AMRLearn.py Spectinomycin.txt Spectinomycin
```

### 3. Example of the files
--------------------------

#### 3.1  Example of the gbff2tab.py output file
The python script gbff2tab.py generates one output file: 4-column spreadsheet integrating with the information of locus tag, gene name, start and end site.

*Example of the 4-column gene location info file:* e.g., gene_location_info.txt
```
locus_tag	gene_name	start_site	end_site
EC958_RS00005	thrL	190	255
EC958_RS00010	thrA	337	2799
EC958_RS00015	thrB	2801	3733
EC958_RS00020	thrC	3734	5020
```
Column header explanation:
1. `locus_tag` gene id (e.g. EC958_RS00005)
2. `gene_name` name info (e.g. thrL)
3. `start_site` location (e.g. 190)
4. `end_site` location (e.g., 255)

<a name="sec5"></a>


#### 3.2  Example of the Parsnp result file.
The python script vcf2snp.py generates one output file, the statistics count of snps based on gene location.

Parsnp was designed to align the core genome of hundreds to thousands of bacterial genomes. Input can be both draft assemblies and finished genomes, and output includes variant (SNP) calls, core genome phylogeny and multi-alignments.


*Example of the variants calling file(vcf) :* e.g.,parsnp.ggr.vcf
```
##INFO=<ID=CDS,Number=1,Type=String,Description="Coding sequence locus">																																										
##INFO=<ID=SYN,Number=0,Type=Flag,Description="All alternative alleles are synonymous in coding sequence">																																										
##INFO=<ID=AAR,Number=1,Type=String,Description="Reference amino acid in coding sequence">																																										
##INFO=<ID=AAA,Number=.,Type=String,Description="Alternate amino acid in coding sequence, one per alternate allele">																																										
##FILTER=<ID=IND,Description="Column contains indel">																																										
##FILTER=<ID=N,Description="Column contains N">																																										
##FILTER=<ID=LCB,Description="LCB smaller than 200bp">																																										
##FILTER=<ID=CID,Description="SNP in aligned 100bp window with < 50% column % ID">																																										
##FILTER=<ID=ALN,Description="SNP in aligned 100b window with > 20 indels">																																										
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	GCF_000285655.3_genomic.gbff.fna	D96_6_1_5.contigs.fasta	D96_8_1_2.contigs.fasta	D6_5_2_2.contigs.fasta	D96_8_2_2.contigs.fasta	D6_8_1_4.contigs.fasta	D96_5_2_1.contigs.fasta	D96_9_2_3.contigs.fasta	D96_6_1_3.contigs.fasta	D96_3_1_4.contigs.fasta	D6_1_1_5.contigs.fasta	D6_5_1_2.contigs.fasta	D96_4_2_4.contigs.fasta	D96_2_2_4.contigs.fasta	D96_1_2_1.contigs.fasta	D96_7_2_5.contigs.fasta	D96_4_2_2.contigs.fasta	D6_9_1_2.contigs.fasta	D6_9_1_4.contigs.fasta	D6_4_1_5.contigs.fasta	D6_9_1_1.contigs.fasta	D6_6_2_4.contigs.fasta	D96_1_1_5.contigs.fasta	D6_6_1_1.contigs.fasta	D6_4_2_1.contigs.fasta	D6_7_2_1.contigs.fasta	D6_3_1_2.contigs.fasta	D6_9_1_5.contigs.fasta	D6_9_2_4.contigs.fasta	D96_1_1_3.contigs.fasta	D6_6_2_5.contigs.fasta	D96_2_1_5.contigs.fasta	D96_1_1_1.contigs.fasta	D96_7_1_1.contigs.fasta
NZ_HG941718	58	AAAAAAGAGT.GTCTGATAGC	G	C	40	PASS	NA	GT	0	0	1	0	0	0	1	0	0	1	1	0	0	1	0	0	0	0	1	0	1	0	0	0	0	1	0	0	0	0	0	1	0	0
NZ_HG941718	64	GAGTGTCTGA.TAGCAGCTTC	T	C	40	PASS	NA	GT	0	0	1	0	0	0	1	0	0	1	1	0	0	1	0	0	0	0	1	0	1	0	0	0	0	1	0	0	0	0	0	1	0	0
NZ_HG941718	95	CCTGCCGTGA.TCAAATTAAA	T	G	40	PASS	NA	GT	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1
```
Column header explanation:
1. `CHROM` identifiers:  (e.g. NZ_HG941718)
2. `POS` location of the contig (e.g. 58)
3. `ID` region (e.g., AAAAAAGAGT.GTCTGATAGC)
4. `REF` reference The protein functional type (e.g., G)
5. `ALT` altered (e.g. C)
6. `QUAL` quality (e.g. 40)
7. `FILTER` (e.g., PASS)
8. `INFO` InterPro Entry Identifier (e.g. CDS=EC958_RS00010;AAR=A;AAA=V)
9. `FORMAT` InterPro Entry Description (e.g. GT)
10. `GCF_000285655.3_genomic.gbff.fna` SNPs# in contig with respect to location (e.g.,1)
<a name="sec5"></a>

#### 3.3  Example of the coef2gene.py output file
The python script coef2gene.py generates one output file: 3-column spreadsheet integrating with the information of locus tag, gene name, coef.

*Example of the 4-column gene location info file:* e.g., gene_location_info.txt
```
locus_tag	gene_name	coef
EC958_RS00140	lspA	0.5310724193574924
EC958_RS00435	ilvI	-1.0890877026034491
EC958_RS00760	panB	-0.15592212697442143
EC958_RS01025	accA	3.1440780394105383
```
Column header explanation:
1. `locus_tag` gene id (e.g. EC958_RS00140)
2. `gene_name` name info (e.g. lspA)
3. `coef` location (e.g. 0.5310724193574924)
<a name="sec5"></a>

### 4. AMR_Learn tutorial

4.1 user will need to install respective packages first
> pip3 install -U scikit-learn scipy matplotlib

```python3


#import dataframe, array, plot required packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC

#import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC


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
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import sys
if len(sys.argv)!=3: #if the input arguments not 4, showing the usage.
    print("Usage:python3 AMRLearn.py <output_file_name> <species name>\n Example: python3 AMRLearn.py Spectinomycin.txt Spectinomycin")
    sys.exit()

# loading data
hsd_data = pd.read_csv(sys.argv[1],sep='\t')
#print(hsd_data.head())

# creating features and arget arrays
#hsd_data = hsd_data.set_index('index')
# print(hsd_data.head())
X = hsd_data.drop(sys.argv[2], axis=1).values
names = hsd_data.drop(sys.argv[2], axis=1).columns
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

#print(confusion_matrix(y_test, ridge_pred))
# print(classification_report(y_test, ridge_pred))
# 
print("3.Lasso Linear regression model\n")
lasso = Lasso(alpha=0.1) #, normalize=True
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
print("Training data:" + str(lasso.score(X_train,y_train)))
print("Testing data: "+ str(lasso.score(X_test,y_test)))
cv_results = cross_val_score(lasso, X, y, cv=5)
print("5-fold cross validation" + str(cv_results)+"\n")

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
# 
outfile=open(sys.argv[2]+"_ridge_coef.txt",'w')
ridge_coef = ridge.fit(X, y).coef_
select_names = []
select_coef = []
for i in range(len(ridge_coef)):
    if abs(ridge_coef[i]) > 0.1:
        select_names.append(names[i])
        select_coef.append(ridge_coef[i])
        outfile.write(str(names[i])+"\t"+str(ridge_coef[i])+"\n")
outfile.close()  # close file is very important!!!!
os.system("python3 coef2gene.py gene_location_info.txt " + sys.argv[2]+"_ridge_coef.txt " + sys.argv[2]+"_ridge_coef_out.txt" )
os.system("rm "+sys.argv[2]+"_ridge_coef.txt ")

_ = plt.plot(range(len(select_names)), select_coef)
_ = plt.xticks(range(len(select_names)), select_names, rotation=90, fontsize=5)
_ = plt.ylabel('Ridge regression coefficients')
#plt.show()
plt.savefig(sys.argv[2]+"_ridge"+".png")
plt.clf() # clear the previous figure

print("new")
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
os.system("python3 coef2gene.py gene_location_info.txt " + sys.argv[2] +"_lasso_coef.txt " + sys.argv[2]+"_lasso_coef_out.txt" )
os.system("rm "+sys.argv[2]+"_lasso_coef.txt ")

_ = plt.plot(range(len(select_names)), select_coef)
_ = plt.xticks(range(len(select_names)), select_names, rotation=90, fontsize=5)
_ = plt.ylabel('Lasso regression coefficients')
#plt.show()
plt.savefig(sys.argv[2]+"_lasso"+".png")

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


```

### 5.Limitation
There is a steep learning curve for researchers with limited knowledge of bioinformatics, especially those who are not familiar with the basic command lines and dash shell in a Linux/Unix environment. At the present time, a “one-click solution” does not exist because of the desire to retain flexibility in the usage of our scripts for different purposes. That said, our tool is comparatively easier to use at current stage. At present there are very few tools that can execute the machine learning predicting highly similar duplicate gene data. AMRLearn thus fills a need for the bioinformatics and genomics community.

### 6. Reference
1.Xi Zhang*, Yining Hu. (2022). AMRLearn: Pipeline for using supervised machine learning algorithm to characterize antimicrobial resistance associated with the SNPs within host evolution. 


