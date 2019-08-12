"""
xgb_handler.py: Utility script for XBoost modeling

__author__ = "Victor Marco Milli"
__version__ = "0.9.1"
__maintainer__ = "Victor Marco Milli"
__status__ = "Project/study script for project SWISS / Bise"

"""
import xgboost as xgb
import pickle
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
import time


def anticipate_start(df, steps):
    """Anticipates the positive event the number of steps (corresponding to intervals of 10 mins) ahead,
    positive event is then repeated till actual positive event which in turn is
    turned into a negative event.
    This is an upsampling technique when targeting start or end of event, and not the event itself"""
    
    df1 = df.copy()
    for index  in range (0, len(df1)):        
        if df1.iloc[index] == 0:
            future_ind = index + steps
            if future_ind >= len(df1):
                steps = len(df1) - index -1
                future_ind = index + steps
            if df1.iloc[future_ind] == 1:
                for i in range (0, steps):
                    df1.iloc[index + i] = 1
                df1.iloc[future_ind] = 0
    return df1

def run_xgboost_prediction_no_lag(df_train_x, df_train_y, df_test_x, df_test_y, dump_dir=None, **hyperparams):
    """Runs a single xgboost classification"""

    return run_xgboost_prediction(df_train_x, df_train_y, df_test_x, df_test_y, pred_period=None, dump_dir=dump_dir, hyperparams=hyperparams)


def run_xgboost_prediction(df_train_x, df_train_y, df_test_x, df_test_y, pred_period=None, dump_dir=None, **hyperparams):
    """Runs a single xgboost classification. If pred_period is set, the y_data will be shifted by pred_period rows"""

    reg = xgb.XGBClassifier()
    reg.set_params(**hyperparams)

    X_train = df_train_x
    y_train = df_train_y

    X_test = df_test_x
    y_test = df_test_y

    if pred_period:

        X_train = df_train_x[:-pred_period]
        y_train = df_train_y[pred_period:]

        X_test = df_test_x[:-pred_period]
        y_test = df_test_y[pred_period:]

    reg.fit(X_train, y_train, verbose=2)

    y_pred = reg.predict(X_test)

    prec_score, rec_score, f_score = get_scores(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    if dump_dir:
        file_path = join(dump_dir, 'xgb_no_tuning_single_pred' + str(time.time()) + '.dat')
        print(f'saving to {file_path}')
        pickle.dump(reg, open(file_path, "wb"))

    cm = None

    try:
        cm = confusion_matrix(y_test, y_pred)
    except:
        print('Confusion Matrix crash!!!')
        print(type(y_test), type(y_pred))


    return reg, prec_score, rec_score, f_score, cm, y_pred



def run_xgboost_predictions_no_tuning(df_train_x, df_train_y, df_test_x, df_test_y, longest_pred_period, dump_dir=None, dump_prefix='', **hyperparams):
    """Runs XGBoost with a fixed set of classifier hyperparameters, perfoming data delay to a maximum of 'longest_pred_period'
    steps (corresponding to 10 mins each)"""
    # initialise two empty dictionaries to store the models and scores
    xgboostmodels = {}
    xgboostscores = {}
    xgboostconfmatrixes = {}
    prediction_period = [i for i in
                         range(1, longest_pred_period + 1)]  # how far into the future to predict (10 minute segments)

    score = -1
    counter = 1
    for prediction in prediction_period:

        reg = xgb.XGBClassifier()
        reg.set_params(**hyperparams)
        X_train = df_train_x[:-prediction]  # everything except the last value
        y_train = df_train_y[prediction:]  # offset by 10mins

        X_test = df_test_x[:-prediction]  # everything except the last value
        y_test = df_test_y[prediction:]  # offset by 10mins

        key = 'xgboost_model_' + dump_prefix + str(prediction * 10)

        xgboostmodels[key] = reg.fit(X_train, y_train, verbose=2)  # store the model

        # score = np.mean(ms.cross_val_score(reg, x_train, y_train, scoring='roc_auc', cv=5))

        y_pred = reg.predict(X_test)

        prec_score, rec_score, f_score = get_scores(y_test, y_pred)
        xgboostscores[key] = rec_score  # store the model score

        print(classification_report(y_test, y_pred))
        print(counter, rec_score)
        # check on number true positives
        if prec_score[1] > score:
            if dump_dir:
                file_path = join(dump_dir, 'xgb_no_tuning_' + dump_prefix + str(time.time()) + '.dat')
                print(f'Model improved accuracy, saving to {file_path}')
                pickle.dump(reg, open(file_path, "wb"))
            score = prec_score[1]
        counter += 1

        cm = None
        try:
            cm = confusion_matrix(y_test, y_pred)
        except:
            print('Confusion Matrix crash!!!')
            print(type(y_test), type(y_pred))

        xgboostconfmatrixes[key] = cm
    return xgboostmodels, xgboostscores, xgboostconfmatrixes




def get_scores(y_test, y_pred):
    prec_score = precision_score(y_test.values, y_pred)
    print('precision score: {:.6f}'.format(prec_score))

    rec_score = recall_score(y_test.values, y_pred)
    print('recall score: {:.6f}'.format(rec_score))
    f_score = f1_score(y_test.values, y_pred)
    print('f1 score: {:.6f}'.format(f_score))

    return prec_score, rec_score, f_score

