##*************************************************************************##
##          Step9. deep learning model                                     ##          
##*************************************************************************##


import sys
if len(sys.argv)!=3: 
    print("Usage:python3 AMR_Learn_DL.py <feature to target processing.txt > <antibiotics name>  \n \n e.g., python3 AMR_Learn_DL.py feature2target_processing.txt Spectinomycin")
    sys.exit()

# Create a keras model
# model building steps: specify architecture; compile; fit; predict
# classification models
import pandas as pd
from pandas.core.frame import DataFrame
# Import necessary modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold,StratifiedKFold
# Import the SGD optimizer
from tensorflow.keras.optimizers import SGD
# Dropout rate between the fully connected layers (useful to reduce overfitting)
#from tensorflow.keras.layers import Dropout

#batch job for multiple antibiotics in the list.txt file
#while read line; do python3 AMR_Learn_DL.py df_all_antibiotics_snps_ribosomes_classification.txt $line;done <list.txt

#writing out the log file
f = open(sys.argv[2]+'_DL.log', 'w')
sys.stdout = f

# loading data
hsd_data = pd.read_csv(sys.argv[1],sep='\t').fillna(0) #in case empty lines, so should have fillna()
#print(hsd_data.head())

# Annotate this line to switch the antibiotics types from R,S to R,I,S, two categories to three categories
#hsd_data = hsd_data[~(hsd_data[sys.argv[2]]==0)]

#hsd_data = hsd_data.set_index('locus_tag')
#hsd_data.reset_index()

predictors = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).values
#sys.argv[2] = the one you want to remove from next line.
#manual correction
names = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).columns
#X = X[:, :2]

targets = to_categorical(hsd_data[sys.argv[2]].values)

# scale the data, preprocessing
# predictors = scale(predictors)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Create list of learning rates: lr_to_test
lr_to_test = [0.001, 0.01, 0.05, 0.1, 0.5]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    num_folds = 5
    model_history=[]
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(predictors, targets):

# Set up the model: model  : linear regression model,
        model = Sequential()
# Add the first layer
        model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
#model.add(Dropout(0.2))
# Add the second layer
        model.add(Dense(50,activation='relu'))
#model.add(Dropout(0.2))
# Add the output layer
        model.add(Dense(2, activation='softmax'))
# Create SGD optimizer with specified learning rate: my_optimizer
        my_optimizer = SGD(learning_rate=lr)
# Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=2)
# Fit the model
        train_model = model.fit(predictors[train],targets[train],validation_split=0.3,epochs=30, callbacks=[early_stopping_monitor],validation_data=(predictors[test], targets[test]))
# Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=2)
#save the training model results into a spreadsheet which is good for visualization in the future        
        model_history.append(train_model.history)
# print each fold
        print('--------------------------------')
        print(f'Training for fold {fold_no} ...')
        # Increase fold number
        fold_no = fold_no + 1

#explore the predication results
        predictions = model.predict(predictors)
        probability_true = predictions[:,0]

    #print(probability_true)
        print("probability	prediction"+"\n")
        print(predictions)

# save model history

model_out = DataFrame(model_history)
model_out.to_csv(sys.argv[2]+ "_DL_model_history.csv",index=False)

f.close()
