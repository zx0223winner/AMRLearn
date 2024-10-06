##*************************************************************************##
##          Step9. deep learning model                                     ##          
##*************************************************************************##


import sys
if len(sys.argv)!=3: 
    print("Usage:python3 AMR_Learn_DL.py <feature to target file > <antibiotics name>  \n \n e.g., python3 AMR_Learn_DL.py feature2target_processing.txt Spectinomycin")
    sys.exit()

# Create a keras model
# model building steps: specify architecture; compile; fit; predict
# classification models

# Import necessary modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.layers import Dropout
# Dropout rate between the fully connected layers (useful to reduce overfitting)
import pandas as pd
# Import the SGD optimizer
from tensorflow.keras.optimizers import SGD

#while read line; do python3 AMR_Learn_DL.py df_all_antibiotics_snps_ribosomes_classification.txt $line;done <list.txt


# loading data

hsd_data = pd.read_csv(sys.argv[1],sep='\t').fillna(0) #empty lines, so should have fillna()
#print(hsd_data.head())

# Annotate this line to switch the antibiotics types from R,S to R,I,S
#hsd_data = hsd_data[~(hsd_data[sys.argv[2]]==0)]

#hsd_data = hsd_data.set_index('locus_tag')
# hsd_data.reset_index()

predictors = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).values
#sys.argv[2] = the one you want to remove from next line.
#manual correction
names = hsd_data.drop(['Spectinomycin','Cefotaxime','Ceftazidime','locus_tag'], axis=1).columns
#X = X[:, :2]

target = to_categorical(hsd_data[sys.argv[2]].values)

# scale the data, preprocessing
# predictors = scale(predictors)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]


# Create list of learning rates: lr_to_test
lr_to_test = [0.001, 0.01, 0.05, 0.1, 0.5]


# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )

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
    #model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Define early_stopping_monitor
    early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
    model.fit(predictors,target,validation_split=0.3,epochs=30, callbacks=[early_stopping_monitor])

# Define early_stopping_monitor
    early_stopping_monitor = EarlyStopping(patience=2)


# Fit the model
#model.fit(predictors,target, epochs=30, validation_split=0.2)

    predictions = model.predict(predictors)

    probability_true = predictions[:,0]

    #print(probability_true)
    print("probability	prediction"+"\n")
    print(predictions)



