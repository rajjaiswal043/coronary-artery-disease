## Data handling
import pandas as pd


## Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

## Preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ## Metrics
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import precision_recall_curve, average_precision_score
# from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_curve,auc


data = pd.read_csv('/home/raj/Documents/project/coronary artery disease/data/CAD.csv')
data.head()
# print(data)
# print(f'{data.shape[0]} rows and {data.shape[1]} columns')
# print(data.info())
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(' ', '_')
# print(data.columns)
# print(data.isnull().sum().any())
# print(f'{data.duplicated().sum()} duplicate row present')

num_cols = ['Age','Weight', 'Length','BMI', 'BP', 'PR', 'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC','Lymph', 'Neut', 'PLT', 'EF-TTE']

cat_cols = ['Sex', 'DM', 'HTN', 'Current_Smoker', 'EX-Smoker', 'FH', 'Obesity', 'CRF', 'CVA', 'Airway_disease', 'Thyroid_Disease',
            'CHF', 'DLP', 'Edema', 'Weak_Peripheral_Pulse', 'Lung_rales', 'Systolic_Murmur', 'Diastolic_Murmur', 'Typical_Chest_Pain',
            'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional_CP', 'LowTH_Ang', 'Q_Wave', 'St_Elevation', 'St_Depression', 'Tinversion',
            'LVH', 'Poor_R_Progression', 'Cath']

ord_cols = ['Function_Class', 'Region_RWMA', 'VHD']

# for cat_col in cat_cols:
#   print(f"* {cat_col} ==> {data[cat_col].unique()} ==> {data[cat_col].nunique()} unique values")

# for num_col in num_cols:
#   print(f"* {num_col}==> {data[num_col].unique()} ==> {data[num_col].nunique()} unique values")

# for ord_col in ord_cols:
#   print(f"* {ord_col}==> {data[ord_col].unique()} ==> {data[ord_col].nunique()} unique values")

# for ord_col in ord_cols:
#     print("* {} : {} Unique Values =>".format(ord_col, data[ord_col].nunique()), data[ord_col].unique())

df = data.copy()

vhd = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
sex = {"Male": "Male", "Fmale": "Female"}

data['VHD'] = data['VHD'].map(vhd)
data['Sex'] = data['Sex'].map(sex)

data.replace('N', 0, inplace=True)
data.replace('Y', 1, inplace=True)

X = data[num_cols]
y = data[cat_cols]

X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(X, y, random_state=0,test_size=0.2)
print('Percentage holdout data: {:.2f}%'.format(len(X_test_init)/len(X)))

X = data.drop('Cath', axis = 1)

y = data['Cath']

map_label = {'Cad':1,'Normal':0}

y = y.map(map_label)

cat_cols.remove('Cath')
# print(cat_cols)

preprocessor = ColumnTransformer(transformers = [('ohe',OneHotEncoder(handle_unknown = 'ignore',sparse_output = False),cat_cols),('scaler',StandardScaler(),num_cols)],remainder = 'passthrough',verbose_feature_names_out = False).set_output(transform = 'pandas')
X_prep = preprocessor.fit_transform(X)

# print(X_prep)
# print(f'Data size: {X_prep.shape[0]} rows and {X_prep.shape[1]} columns')

# symtoms found in the heart disease patient 
grouped_data = data.groupby(['Typical_Chest_Pain', 'Atypical', 'Nonanginal', 'Exertional_CP', 'LowTH_Ang'])['Cath'].value_counts(normalize=False).to_frame(name="Number of Patients")
styled_grouped_data = grouped_data.style.background_gradient(cmap='Blues')
print(styled_grouped_data.data)

# symtoms found in the patient in percentage 
group_data=data.groupby(['Typical_Chest_Pain', 'Atypical', 'Nonanginal', 'Exertional_CP', 'LowTH_Ang'])['Cath'].value_counts(normalize=True).mul(100).round(3).to_frame(name="Percent within group (%)")
styled_group_data=group_data.style.background_gradient(cmap='Blues')
print(styled_group_data.data)

skf = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 123)

# logisticregression model
lr = LogisticRegression(random_state = 123)
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results_lr = cross_validate(estimator=lr, X=X_prep, y=y, scoring=scoring, cv=skf, return_train_score=True)
print("Cross-validation results for Logistic Regression:")
for score in scoring:
    mean_score = cv_results_lr[f'test_{score}'].mean()
    print(f"{score}: {mean_score:.2f}")

# Randomforest classifier model
rf = RandomForestClassifier(random_state = 123)

cv_results_rf = cross_validate(estimator = rf,X = X_prep,y = y,scoring = ['accuracy', 'precision', 'recall', 'f1'],cv = skf,verbose = 1,return_train_score = True,error_score = 'raise')

cv_results_rf = cross_validate(estimator=rf, X=X_prep, y=y, scoring=scoring, cv=skf, return_train_score=True)
print("Cross-validation results for randomforest:")
for score in scoring:
    mean_score = cv_results_rf[f'test_{score}'].mean()
    print(f"{score}: {mean_score:.2f}")


# Light BGM model
lgbm = LGBMClassifier(random_state = 123, verbose = -1)

cv_results_lgbm = cross_validate(estimator = lgbm,X = X_prep,y = y,scoring = ['accuracy', 'precision', 'recall', 'f1'],cv = skf,verbose = 1,return_train_score = True,error_score = 'raise')

cv_results_lgbm = cross_validate(estimator=lgbm, X=X_prep, y=y, scoring=scoring, cv=skf, return_train_score=True)
print("Cross-validation results for Light BGM:")
for score in scoring:
    mean_score = cv_results_lgbm[f'test_{score}'].mean()
    print(f"{score}: {mean_score:.2f}")

# XGB classifier model
xgb = XGBClassifier(random_state = 123)

cv_results_xgb = cross_validate(estimator = xgb,X = X_prep,y = y,scoring = ['accuracy', 'precision', 'recall', 'f1'],cv = skf,verbose = 1,return_train_score = True,error_score = 'raise')

cv_results_xgb = cross_validate(estimator=xgb, X=X_prep, y=y, scoring=scoring, cv=skf, return_train_score=True)
print("Cross-validation results for XGB classifier:")
for score in scoring:
    mean_score = cv_results_xgb[f'test_{score}'].mean()
    print(f"{score}: {mean_score:.2f}")

# catboost model
cb = CatBoostClassifier(random_state = 123, verbose = 0)

cv_results_cb = cross_validate(estimator = cb,X = X_prep,y = y,scoring = ['accuracy', 'precision', 'recall', 'f1'],cv = skf,verbose = 1,return_train_score = True,error_score = 'raise')

cv_results_cb = cross_validate(estimator=cb, X=X_prep, y=y, scoring=scoring, cv=skf, return_train_score=True)
print("Cross-validation results for CatBoost:")
for score in scoring:
    mean_score = cv_results_cb[f'test_{score}'].mean()
    print(f"{score}: {mean_score:.2f}")

# round off the percentage for graph 
# logistic regression train and test data
metrics_train_lr = {'accuracy':round(cv_results_lr['train_accuracy'].mean(), 3),'precision':round(cv_results_lr['train_precision'].mean(), 3),'recall':round(cv_results_lr['train_recall'].mean(), 3),'f1':round(cv_results_lr['train_f1'].mean(), 3)}
metrics_test_lr = {'accuracy':round(cv_results_lr['test_accuracy'].mean(), 3),'precision':round(cv_results_lr['test_precision'].mean(), 3),'recall':round(cv_results_lr['test_recall'].mean(), 3),'f1':round(cv_results_lr['test_f1'].mean(), 3)}
# random forest train and test data 
metrics_train_rf = {'accuracy':round(cv_results_rf['train_accuracy'].mean(), 3),'precision':round(cv_results_rf['train_precision'].mean(), 3),'recall':round(cv_results_rf['train_recall'].mean(), 3),'f1':round(cv_results_rf['train_f1'].mean(), 3)}
metrics_test_rf = {'accuracy':round(cv_results_rf['test_accuracy'].mean(), 3),'precision':round(cv_results_rf['test_precision'].mean(), 3),'recall':round(cv_results_rf['test_recall'].mean(), 3),'f1':round(cv_results_rf['test_f1'].mean(), 3)}
# random forest train and test data 
metrics_train_lgbm = {'accuracy':round(cv_results_lgbm['train_accuracy'].mean(), 3),'precision':round(cv_results_lgbm['train_precision'].mean(), 3),'recall':round(cv_results_lgbm['train_recall'].mean(), 3),'f1':round(cv_results_lgbm['train_f1'].mean(), 3)}
metrics_test_lgbm = {'accuracy':round(cv_results_lgbm['test_accuracy'].mean(), 3),'precision':round(cv_results_lgbm['test_precision'].mean(), 3),'recall':round(cv_results_lgbm['test_recall'].mean(), 3),'f1':round(cv_results_lgbm['test_f1'].mean(), 3)}
# random forest train and test data 
metrics_train_xgb = {'accuracy':round(cv_results_xgb['train_accuracy'].mean(), 3),'precision':round(cv_results_xgb['train_precision'].mean(), 3),'recall':round(cv_results_xgb['train_recall'].mean(), 3),'f1':round(cv_results_xgb['train_f1'].mean(), 3)}
metrics_test_xgb = {'accuracy':round(cv_results_xgb['test_accuracy'].mean(), 3),'precision':round(cv_results_xgb['test_precision'].mean(), 3),'recall':round(cv_results_xgb['test_recall'].mean(), 3),'f1':round(cv_results_xgb['test_f1'].mean(), 3)}
# random forest train and test data 
metrics_train_cb = {'accuracy':round(cv_results_cb['train_accuracy'].mean(), 3),'precision':round(cv_results_cb['train_precision'].mean(), 3),'recall':round(cv_results_cb['train_recall'].mean(), 3),'f1':round(cv_results_cb['train_f1'].mean(), 3)}
metrics_test_cb = {'accuracy':round(cv_results_cb['test_accuracy'].mean(), 3),'precision':round(cv_results_cb['test_precision'].mean(), 3),'recall':round(cv_results_cb['test_recall'].mean(), 3),'f1':round(cv_results_cb['test_f1'].mean(), 3)}

# train set is used 
models_name = {'logistic_reg':metrics_train_lr,'random_forest':metrics_train_rf,'xgboost':metrics_train_xgb,'lgbm':metrics_train_lgbm,'catboost':metrics_train_cb}

df_train_metrics = pd.DataFrame.from_dict(models_name,orient='index')

df_train_metrics = df_train_metrics.sort_values('recall', ascending = False)

fig,ax = plt.subplots(figsize=(9,4.5))
sns.heatmap(df_train_metrics, annot=True, cmap = 'coolwarm', annot_kws = {'fontweight':'bold'},fmt = '.3f', ax = ax)
ax.xaxis.tick_top()
ax.set_ylabel('Models', fontsize = 11, fontweight = 'bold', color = 'blue')
ax.set_title('Metrics', fontsize = 11, fontweight = 'bold', color = 'blue')
plt.show()

# test set used 
models_name = {'logistic_reg':metrics_test_lr,'random_forest':metrics_test_rf,'xgboost':metrics_test_xgb,'lgbm':metrics_test_lgbm,'catboost':metrics_test_cb}

df_test_metrics = pd.DataFrame.from_dict(models_name,orient='index')

df_test_metrics = df_test_metrics.sort_values('recall', ascending = False)

fig,ax = plt.subplots(figsize=(9,4.5))
sns.heatmap(df_test_metrics, annot=True, cmap = 'coolwarm', annot_kws = {'fontweight':'bold'},fmt = '.3f', ax = ax)
ax.xaxis.tick_top()
ax.set_ylabel('Models', fontsize = 11, fontweight = 'bold', color = 'blue')
ax.set_title('Metrics', fontsize = 11, fontweight = 'bold', color = 'blue')
plt.show()


