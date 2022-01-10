import sklearn.metrics as metric
import pandas as pd

def results_analysis(gt, y_pred):
    df = dict()
    df['accuracy'] = [metric.accuracy_score(gt, y_pred)]
    df['f1score'] = [metric.f1_score(gt, y_pred)] #2 * (precision * recall) / (precision + recall)
    df['precision'] = [metric.precision_score(gt, y_pred)] #tp / (tp + fp)
    df['recall'] = [metric.recall_score(gt, y_pred)] #tp / (tp + fn)
    cm = metric.confusion_matrix(gt, y_pred)
    cm = {'tn': [cm[0, 0]], 'fp': [cm[0, 1]],
            'fn': [cm[1, 0]], 'tp': [cm[1, 1]]}

    return pd.DataFrame(df), pd.DataFrame(cm)
