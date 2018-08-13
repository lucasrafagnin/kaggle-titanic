# coding: utf-8

import pandas as pd

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

outcomes = train['Survived']
ids = []


def predict(data):
    predictions = []
    for _, passenger in data.iterrows():
        ids.append(passenger['PassengerId'])
        if passenger['Sex'] == 'female':
            if passenger['Pclass'] == 3 and passenger['Age'] > 40:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if passenger['Pclass'] != 3:
                if passenger['Age'] < 10:
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)
    return pd.Series(predictions)


def score(truth, pred):
    if len(truth) == len(pred):
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)
    else:
        return "Number of predictions does not match number of outcomes!"


def export_to_kaggle():
    predictions = predict(test)
    submission = pd.DataFrame()
    submission['PassengerId'] = ids
    submission['Survived'] = predictions
    submission.to_csv('submission.csv', index=False)
    print('Kaggle submission.csv file exported')


def train_dataset():
    predictions = predict(train)
    print(score(outcomes, predictions))


train_dataset()
# export_to_kaggle()
