from diploma import *



def get_confusion_matrix(y_true,y_pred):
     return confusion_matrix(y_true, y_pred).ravel()

def recall(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return tp / (tp + fn)
    except:
        return 0 
def precision(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return tp / (tp + fp)
    except:
        return 0  
def true_negative_rate(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return tn / (tn + fp)
    except:
        return 0
def false_negative_rate(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return fn / (fn + tp)
    except:
        return 0
def false_positive_rate(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return fn / (fn + tp)
    except:
        return 0

def positive_likehood(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return recall(confusion_matrix) / false_positive_rate(confusion_matrix)
    except:
        return 0
def negative_likehood(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return false_negative_rate(confusion_matrix) / true_negative_rate(confusion_matrix)
    except:
        return 0
def MCC(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return ((tp*tn) - (fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1./2)
    except:
        return 0
def f1_score(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return (2*tp / (2*tp + fp + fn))
    except:
        return 0

def diagnostic_odds_ratio(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix
    try:
      return positive_likehood(confusion_matrix) / negative_likehood(confusion_matrix)
    except:
        return 0

def print_statistics(y_true,y_pred):
    confusion_matrix =  get_confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    print(f"recall : {round(recall(confusion_matrix),3)}")
    print(f"precision : {round(precision(confusion_matrix),3)}")
    print(f"true_negative_rate : {round(true_negative_rate(confusion_matrix),3)}")
    print(f"false_negative_rate : {round(false_negative_rate(confusion_matrix),3)}")
    print(f"false_positive_rate : {round(false_positive_rate(confusion_matrix),3)}")
    print(f"positive_likehood : {round(positive_likehood(confusion_matrix),3)}")
    print(f"negative_likehood : {round(negative_likehood(confusion_matrix),3)}")
    print(f"f1_score : {round(f1_score(confusion_matrix),3)}")
    print(f"MCC : {round(MCC(confusion_matrix),3)}")
    print(f"diagnostic_odds_ratio : {round(diagnostic_odds_ratio(confusion_matrix),3)}")

    print(classification_report(y_true, y_pred))

def get_statistics_from_cf(confusion_matrix):
    return {
    "confusion_matrix": confusion_matrix,
    "precision" : round(precision(confusion_matrix),3),
    "recall" : round(recall(confusion_matrix),3),
    "true_negative_rate" : round(true_negative_rate(confusion_matrix),3),
    "false_negative_rate" : round(false_negative_rate(confusion_matrix),3),
    "false_positive_rate" : round(false_positive_rate(confusion_matrix),3),
    "positive_likehood" : round(positive_likehood(confusion_matrix),3),
    "negative_likehood" : round(negative_likehood(confusion_matrix),3),
    "f1_score" : round(f1_score(confusion_matrix),3),
    "MCC" : round(MCC(confusion_matrix),3),
    "diagnostic_odds_ratio" : round(diagnostic_odds_ratio(confusion_matrix),3)
    }

def get_statistics(y_true,y_pred):
    confusion_matrix =  get_confusion_matrix(y_true, y_pred)
    return {
    "confusion_matrix": confusion_matrix,
    "precision" : round(precision(confusion_matrix),3),
    "recall" : round(recall(confusion_matrix),3),
    "true_negative_rate" : round(true_negative_rate(confusion_matrix),3),
    "false_negative_rate" : round(false_negative_rate(confusion_matrix),3),
    "false_positive_rate" : round(false_positive_rate(confusion_matrix),3),
    "positive_likehood" : round(positive_likehood(confusion_matrix),3),
    "negative_likehood" : round(negative_likehood(confusion_matrix),3),
    "f1_score" : round(f1_score(confusion_matrix),3),
    "MCC" : round(MCC(confusion_matrix),3),
    "diagnostic_odds_ratio" : round(diagnostic_odds_ratio(confusion_matrix),3)
    }
def save_results(results,modelSavingPath,model_dirname,train,test,val, stats,predictions):
    with open(f'{modelSavingPath + model_dirname}/trainingInfo.pickle', 'wb') as file:
        dill.dump({
            "test":test,
            "train":train, 
            "val":val,
            "results": results.history if results else None,
            "stats" : stats,
            "predictions": predictions
            }, file)
    res = tf.keras.utils.plot_model(model,to_file=f'{modelSavingPath + model_dirname}/model.png', show_shapes=True)
       