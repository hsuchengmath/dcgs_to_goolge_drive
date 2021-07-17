import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as metrics 
from sklearn.metrics import confusion_matrix
import random


def transform_date_to_age(date_str, categorical=True):
    if date_str != 'None':
        age_val = 2021 - pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S').year
        if categorical is False:
            return age_val
        else:
            if age_val <= 30:
                return '0~30'
            elif age_val > 30 and age_val < 50:
                return '30~50'
            else:
                return '50~'   
    else:
        return 'None'


def train_model(train_data,label):
    train_data = np.array(train_data)
    rf = RandomForestRegressor()
    rf.fit(train_data, label)
    return rf


def predict_score(pred_prob, Y_test_array, binary_threshold=0.5):
    pred_one_hot = list()
    for i in range(pred_prob.shape[0]):
        if pred_prob[i] >= binary_threshold:
            pred_one_hot.append(1)
        else:
            pred_one_hot.append(0)
    print(metrics.classification_report(list(Y_test_array), pred_one_hot))
    print('---------------------------------------')
    print('Confusion Matrix')
    print(np.transpose(confusion_matrix(list(Y_test_array), pred_one_hot).T))
    print('---------------------------------------')
    print('positive label : 1 | negative label : 0')


def random_model_performance(ground_truth):
    ground_truth = list(ground_truth)
    pos,neg = 0,0
    for val in ground_truth:
        if int(val) == 1:
            pos +=1
        else:
            neg +=1
    random_pred = np.array([random.sample([1,0],1)[0] for _ in range(len(ground_truth))])
    predict_score(random_pred,ground_truth)