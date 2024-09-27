# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 13:04:58 2022

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Input, Dense, Flatten
from keras.models import Model,Sequential
import tensorflow as tf
import sklearn.ensemble
from sklearn import preprocessing

data = pd.read_csv('D:\\divorce_data.csv')

#計算correlation
cor = data.corr(method="spearman")
#補空值
cor=cor.fillna(value=0)

#畫與divorce 的相關係數直方圖
corr_value=list(cor['Divorce'])
corr_index=list(cor['Divorce'].index)

plt.figure(figsize=(22,8))
plt.bar(corr_index, corr_value)

plt.xlabel('index')
plt.ylabel('value')
plt.title('corr')
plt.show()

#挑選重要變數
important_num_cols=list(cor["Divorce"][(cor["Divorce"]>0.8)].index)
final_col=important_num_cols
df=data[final_col]

#分資料
x=df.iloc[:,:39]
y=df.iloc[:,39:40]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,random_state=42)

#SVC
svc_model= SVC(kernel = "linear",probability=True)
svc_model.fit(x_train, y_train)
svc_y_pred=svc_model.predict(x)
svc_pred=pd.Series(svc_y_pred)

#LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
log_y_pred=log_model.predict(x)
log_pred=pd.Series(log_y_pred)

#DT
dt_model= DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
dt_y_pred=dt_model.predict(x)
dt_pred=pd.Series(dt_y_pred)

#合併結果
y_con=pd.concat([svc_pred,log_pred,dt_pred],axis = 1)

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(y_con)

#nn分資料
x_nntrain,x_nntest, y_nntrain, y_nntest = train_test_split(X ,y, test_size=0.3)

#nn結構
visible1 = Input(shape=(3,))
hidden1 = Dense(32, activation='relu')(visible1) 
hidden2 = Dense(32, activation='relu')(hidden1)
hidden3 = Dense(20, activation='relu')(hidden2)
output = Dense(2, activation='softmax')(hidden3)
model = Model(inputs=[visible1] , outputs=output)
opt = tf.keras.optimizers.Adam(learning_rate=0.0010)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()

hist= model.fit(x_nntrain, y_nntrain, epochs =100, batch_size =16, validation_data=(x_nntest, y_nntest), shuffle=True)

y_nnpred=model.predict(y_con)

#判斷結果
result=[]
for i in range(len(y_nnpred)):
    if y_nnpred[i][1]>y_nnpred[i][0]:
        result.append("divorce")
    elif y_nnpred[i][0]>y_nnpred[i][1]:
        result.append("not divorce")
    else:
        result.append("same")
print(result)

#畫圖
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('NN accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('NN loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

def models(xx):
    svc_y_pred=svc_model.predict(xx)
    svc_pred=pd.Series(svc_y_pred)
    log_y_pred=log_model.predict(xx)
    log_pred=pd.Series(log_y_pred)
    dt_y_pred=dt_model.predict(xx)
    dt_pred=pd.Series(dt_y_pred)
    y_con=pd.concat([svc_pred,log_pred,dt_pred],axis = 1)
    y_nnpred=model.predict(y_con)
    return(y_nnpred)

import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_train.columns,
    class_names=['not divorce', 'divorce'],
    mode='classification'
)

xx=np.array(x.iloc[100])
exp = explainer.explain_instance(
    data_row=xx, 
    predict_fn=models
)

exp.show_in_notebook(show_table=True)