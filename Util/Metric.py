from sklearn.metrics import f1_score

def macro_f1_score(true, pred):
    return f1_score(true, pred, average='macro')
