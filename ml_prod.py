import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

import warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

pred = pd.read_csv('Pred.csv')
print(f'Fechamento anterior: {pred["Close"][0]}')
print(f'Previsão anterior: {pred["target"][0]}')

base = pd.read_csv('Today.csv')

try:
    tomorrow = pd.read_csv('Future.csv')
    print(f'Fechamento atual: {tomorrow["Close"][0]}')
    base = pd.concat([base, tomorrow[:1]], sort=True)
    tomorrow = tomorrow.drop(tomorrow[:1].index, axis=0)
    base.to_csv('Today.csv', index=False)
    tomorrow.to_csv('Future.csv', index=False)
except Exception:
    print('O fechamento ainda não ocorreu.')
    pass

base['target'] = base['Close'][1:len(base)].reset_index(drop=True)
pred = base[-1::].drop('target', axis =1)
train = base.drop(base[-1::].index, axis=0)

train.loc[train['target'] > train['Close'], 'target'] = 1
train.loc[train['target'] != 1, 'target'] = 0


y = train['target']
x = train.drop('target', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = ExtraTreesClassifier()
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(f'Accuracy: {result}')

pred['target'] = model.predict(pred)
print(f'Fechamento de ontem: {pred["Close"][0]}')

if pred['target'][0] == 1:
    print('VAI SUBIR!!!')
else:
    print('Vai cair.')

pred.to_csv('Pred.csv', index=False)


