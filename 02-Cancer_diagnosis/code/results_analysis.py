from configparser import NoSectionError
import sklearn.metrics as metric
import pandas as pd
from model import ModelHandler
from sklearn.model_selection import KFold
import pandas as pd
from Classifier import CancerClassifier
import torch
import torchvision.transforms as T
from torch.optim import AdamW


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

def results(X_data, y_data, MODEL, NSPLITS):

        kf = KFold(n_splits = NSPLITS, shuffle = True)
        data = {'tp':[] , 'fp': [] , 'tn': [] , 'fn': []}
        data2 = {'accuracy':[] , 'f1score': [] , 'recall': [] , 'precision': []}
        index = ['split'+str(x) for x in range(NSPLITS)]
        for idx, idy in kf.split(X_data, y_data):
                model = ModelHandler(X = X_data[idx], Y = y_data[idx], model = MODEL, n_splits = NSPLITS)
                model.fit(True,True)

                df, cm  = results_analysis( y_data[idy], model.predict(X_data[idy]))
                for x in data:
                    data[x].append(cm[x][0] )
                for x in data2:
                    data2[x].append(df[x][0])
        data = pd.DataFrame(data)
        data.index = index
        data2 = pd.DataFrame(data2)
        data2.index = index
        from IPython.display import display, HTML

        css = """
        .output {
        flex-direction: row;
        }
        """

        HTML('<style>{}</style>'.format(css))
        display(data)
        display(data2)


def results_deep(X_data, y_data, MODEL_PATH, NSPLITS,epochs=10,LR=1e-3):
        
        kf = KFold(n_splits = NSPLITS, shuffle = True)
        loss = torch.nn.BCELoss()
        data = {'tp':[] , 'fp': [] , 'tn': [] , 'fn': []}
        data2 = {'accuracy':[] , 'f1score': [] , 'recall': [] , 'precision': []}
        index = ['split'+str(x) for x in range(NSPLITS)]
        transform = T.Compose([T.Normalize(mean = (-640 ) , std=(380))])
        for idx, idy in kf.split(X_data, y_data):
                model = CancerClassifier(MODEL_PATH).cuda()
                optimizer = AdamW(model.parameters(), lr = LR)
                for k in range(epochs):
                       for img,gt in zip(X_data[idx],y_data[idx]):
                               optimizer.zero_grad()
                               gt = torch.Tensor(1)
                               t,y = transform(torch.from_numpy(img)).reshape(14,1,64,64).float().cuda(), torch.tensor(gt)
                               gt[0] = y
                               out = model(t)
                               #acc += 1 if (out>=0.5) == int(y) else 0 
                               loss_ = loss(out, gt.float().cuda())
                               loss_.backward()
                final = []
                for img,gt in zip(X_data[idy],y_data[idy]):
                        out = model(t)
                        final.append(out >= 0.5)
                df, cm  = results_analysis( y_data[idy], final)
                for x in data:
                    data[x].append(cm[x][0] )
                for x in data2:   
                    data2[x].append(df[x][0])
        data = pd.DataFrame(data)
        data.index = index
        data2 = pd.DataFrame(data2)
        data2.index = index
        from IPython.display import display, HTML

        css = """
        .output {
        flex-direction: row;
        }
        """

        HTML('<style>{}</style>'.format(css))
        display(data)
        display(data2)