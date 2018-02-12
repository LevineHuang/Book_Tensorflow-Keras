# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:52:00 2018

@author: Vincent
"""

import numpy
import pandas as pd
from sklearn import preprocessing

numpy.random.seed(10)
all_df = pd.read_excel("data/titanic3.xls")
cols=['survived','name','pclass' ,'sex', 'age', 'sibsp',
      'parch', 'fare', 'embarked']
all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))

def PreprocessData(raw_df):
    df=raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])

    ndarray = x_OneHot_df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)

#Create Model
from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(units=40, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))

#Train model
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=30, 
                         batch_size=30,verbose=2)

#預測資料
Jack = pd.Series([0 ,'Jack',3, 'male'  , 23, 1, 0,  5.0000,'S'])
Rose = pd.Series([1 ,'Rose',1, 'female', 20, 1, 0, 100.0000,'S'])
JR_df = pd.DataFrame([list(Jack),list(Rose)],  
                  columns=['survived', 'name','pclass', 'sex', 
                   'age', 'sibsp','parch', 'fare','embarked'])
all_df=pd.concat([all_df,JR_df])
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict(all_Features)