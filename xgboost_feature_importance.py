import pandas as pd
import xgboost as xgb
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

loaded_data = pd.read_csv("./Classification_and_Importance/features.csv")

col = 'class'
conditions = [loaded_data[col] == 1, loaded_data[col] == -1]
choices = ["1", "0"]

loaded_data['class'] = np.select(conditions, choices, default=np.nan)
loaded_data['class'] = pd.to_numeric(loaded_data['class'])

### feature importance generation

label = loaded_data[['class']]
features = loaded_data.iloc[:, 2:]

data_matrix = xgb.DMatrix(features, label)

params = {'objective':'binary:logistic'} #params need to be explicity stated, binary:logistic chosen since that is default in API settings

trained = xgb.train(params, data_matrix)
#xgb.plot_importance(trained)
feature_importance = trained.get_score(importance_type='weight')
k = Counter(feature_importance)
feature_importance_table = pd.DataFrame.from_dict(k, orient='index').reset_index()
feature_importance_table.columns = ['Features', 'Weights']

#plot the feature importance
fts = feature_importance_table['Features']
weights = feature_importance_table['Weights']
y_pos = np.arange(len(weights))

plt.bar(y_pos, weights, color="#5dbcd2")
plt.xticks(y_pos, fts, rotation = 90)
plt.xlabel("Features", fontsize=15)
plt.ylabel("Weights", fontsize=15)
plt.savefig("./Plots/featureimp.png", bbox_inches='tight')
plt.show()
