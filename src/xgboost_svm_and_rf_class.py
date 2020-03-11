import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

loaded_data = pd.read_csv("./Classification_and_Importance/features.csv")

col = 'class'
conditions = [loaded_data[col] == 1, loaded_data[col] == -1]
choices = ["1", "0"] #changing labels to 1 and 0 since that is required

loaded_data['class'] = np.select(conditions, choices, default=np.nan)
loaded_data['class'] = pd.to_numeric(loaded_data['class'])

label = loaded_data[['class']]
features = loaded_data.iloc[:, 2:]
features = features.drop(['has superlative', 'has comparative','has sup or comp'], axis =1)

#convert to array
label_arr = np.ravel(label)
features_arr = np.array(features)

#scoring to get accuracy, precision, recall, f1-score
scoring_tech = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}


## XGBoost
xgb_clf = xgb.XGBClassifier()
score_xgb = cross_validate(xgb_clf, features_arr, label_arr, cv=10, scoring=scoring_tech)
xgb_results = pd.DataFrame.from_dict(score_xgb)
y_pred_xgb = cross_val_predict(xgb_clf, features_arr, label_arr, cv=10)


## Suport Vector Classifier
svm_clf = SVC(gamma='auto') #all default parameters, gamma has to be explicitly stated
score_svm = cross_validate(svm_clf, features_arr, label_arr, cv=10, scoring=scoring_tech)
svm_results = pd.DataFrame.from_dict(score_svm)
y_pred_svm = cross_val_predict(svm_clf, features_arr, label_arr, cv=10)


## Random Forest
rf_clf = RandomForestClassifier(n_estimators=100) #default value
score_rf = cross_validate(rf_clf, features_arr, label_arr, cv=10, scoring=scoring_tech)
rf_results = pd.DataFrame.from_dict(score_rf)
y_pred_rf = cross_val_predict(rf_clf, features_arr, label_arr, cv=10)

### Confusion Matrix Plot

##XGBoost
cmtx_xgb = pd.DataFrame(
    confusion_matrix(label_arr, y_pred_xgb, labels=[1, 0]),
    index=['Clickbait', 'Non-Clickbait'],
    columns=['Clickbait', 'Non-Clickbait']
)
#plot
ax = sns.heatmap(cmtx_xgb, annot=True, fmt='d', cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for XGBoost")

##SVM
cmtx_svm = pd.DataFrame(
    confusion_matrix(label_arr, y_pred_svm, labels=[1, 0]),
    index=['Clickbait', 'Non-Clickbait'],
    columns=['Clickbait', 'Non-Clickbait']
)
#plot
ax = sns.heatmap(cmtx_svm, annot=True, fmt='d', cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for SVM")

##RBF
cmtx_rf = pd.DataFrame(
    confusion_matrix(label_arr, y_pred_rf, labels=[1, 0]),
    index=['Clickbait', 'Non-Clickbait'],
    columns=['Clickbait', 'Non-Clickbait']
)
#plot
ax = sns.heatmap(cmtx_rf, annot=True, fmt='d', cmap=sns.light_palette((210, 90, 60), input="husl"))
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix for Random Forest")

### ROC Curve
auc_xgb = roc_auc_score(label_arr, y_pred_xgb)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(label_arr, y_pred_xgb)

auc_svm = roc_auc_score(label_arr, y_pred_svm)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(label_arr, y_pred_svm)

auc_rf = roc_auc_score(label_arr, y_pred_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(label_arr, y_pred_rf)

plt.plot(fpr_xgb, tpr_xgb, label='XGBoost = %0.3f' % auc_xgb, color="#5dbcd2", linewidth=2)
plt.plot(fpr_svm, tpr_svm, label='SVM = %0.3f' % auc_svm, color="#ffcc00", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label='Random Forest = %0.3f' % auc_rf, color="#ff66cc", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color="#878787")  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Receiver Operating Characteristic', fontsize=18)
plt.legend(loc="lower right", title="Area Under Curve")


