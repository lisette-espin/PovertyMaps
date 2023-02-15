from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import plot_confusion_matrix

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

############################################################
# Functions
############################################################

def runGradientBoostingDecisionTreeRegressor(X,y):
  ### scaling X
  scaler = MinMaxScaler()
  X_scale = scaler.fit_transform(X)

  ### Train, test
  X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=123, shuffle=True)

  ### Fit model
  learning_rates = [0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 1]

  best_r2 = 0
  best_n_estimators = 0
  best_learning_rate = 0
  max_depth = 2
  n_estimators = 100

  for learning_rate in learning_rates:
    regressor = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    regressor.fit(X_train, y_train)

    ### Evaluation / Validation
    r2s = [r2_score(y_test, y_pred) for y_pred in regressor.staged_predict(X_test)]
    best = np.argmax(r2s)
    r2 = r2s[best]

    if r2 > best_r2:
      best_r2 = r2
      best_n_estimators = best + 1
      best_learning_rate = learning_rate
      
  ### Best model
  print("best hyperparameters:")
  print('n_estimators: {}'.format(best_n_estimators))
  print('learning_rate: {}'.format(best_learning_rate))
  print('best_r2: {}'.format(best_r2))

  best_regressor = GradientBoostingRegressor(
      max_depth=max_depth,
      n_estimators=best_n_estimators,
      learning_rate=best_learning_rate
  )
  best_regressor.fit(X_train, y_train)

  ### Evaluation
  y_pred = best_regressor.predict(X_test)
  print("R-square: {}".format(r2_score(y_test, y_pred)))
  print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))

  ### Plot
  mx = np.concatenate((y_pred, y_test)).max()
  plt.scatter(y_test, y_pred)
  plt.plot([0,mx+1],[0,mx+1],ls='--',c='grey')
  _ = plt.xlabel('Ground Truth')
  _ = plt.ylabel('Prediction')

def runGradientBoostingDecisionTreeClassifier(X, y):
  ### scaling X
  scaler = MinMaxScaler()
  X_scale = scaler.fit_transform(X)
  X_train, X_val, y_train, y_val = train_test_split(X_scale, y.values.ravel(), random_state=0)

  ### validation (parameter tuning)
  learning_rates = [0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 1]
  best_lr = 0
  best_acc = 0
  for learning_rate in learning_rates:
      gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
      gb.fit(X_train, y_train)
      acc_train = gb.score(X_train, y_train)
      acc_test = gb.score(X_val, y_val)
      print("Learning rate: ", learning_rate)
      print("Accuracy score (training): {0:.3f}".format(acc_train))
      print("Accuracy score (validation): {0:.3f}".format(acc_test))
      print()

      if acc_test > best_acc:
        best_acc = acc_test
        best_lr = learning_rate

  ### evaluation
  print("best learning rate: {}".format(best_lr))
  gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=best_lr, max_features=2, max_depth=2, random_state=0)
  gb_clf2.fit(X_train, y_train)
  y_pred = gb_clf2.predict(X_val)

  print("Classification Report")
  print(classification_report(y_val, y_pred))

  ### confusion matrix plot
  plot_confusion_matrix(gb_clf2, X_val, y_val)  
  plt.show()  
  

