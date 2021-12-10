# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import pandas as pd
import numpy as np
from helpers.eda import *
from helpers.data_prep import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.shape
df.columns
# 1. EDA
# Convert total charges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
# Label churn
df["Churn"] = df["Churn"].map({"Yes" : 1, "No" : 0})

check_df(df)

# Categorical and numerical variable summaries

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df,col,plot=True)

for col in num_cols:
    num_summary(df,col,plot=True,kde = True)

# Target summary with categorical and numerical variables

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col,plot = True)

for col in num_cols:
    target_summary_with_num(df,"Churn",col,plot = [True,"mean"])

for col in num_cols:
    box_plot(df,"Churn",col)


# 2. Data Preprocessing & Feature Engineering

# Create has internet service columns
df["HAS_INTERNET_SERVICE"] = df["InternetService"].apply(lambda x: "0" if x == "No" else "1")

# Create has phone service column
df["HAS_PHONE_SERVICE"] = df["PhoneService"].apply(lambda x: "0" if x == "No" else "1")

# Crate has both services columns
df.loc[(df["HAS_INTERNET_SERVICE"] == "1") & (df["HAS_PHONE_SERVICE"] == "1"),"HAS_BOTH_SERVICES"] = "1"
df["HAS_BOTH_SERVICES"].fillna("0",inplace = True)

df['TOTAL_SERVICES'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Convert numerical variables to categorical variables
df["TENURE_CAT"] = pd.qcut(df['tenure'], 4, labels=["D","C","B","A"])
df["TOTAL_CHARGES_CAT"] = pd.qcut(df['TotalCharges'], 4, labels=["D","C","B","A"])
df["MONTHLY_CHARGES_CAT"] = pd.qcut(df['MonthlyCharges'], 4, labels=["D","C","B","A"])


# Summary of senior citizens according to montly charges and  total charges

df.groupby("SeniorCitizen")["TotalCharges","MonthlyCharges","tenure"].mean()
df["tenure"].mean()

# Which services senior citizens are using

df[["HAS_INTERNET_SERVICE","HAS_PHONE_SERVICE","HAS_BOTH_SERVICES"]] = df[["HAS_INTERNET_SERVICE","HAS_PHONE_SERVICE","HAS_BOTH_SERVICES"]].apply(pd.to_numeric)
df.groupby("SeniorCitizen")["HAS_INTERNET_SERVICE","HAS_PHONE_SERVICE","HAS_BOTH_SERVICES"].sum()

# Online Backup,online security,device protection, tech support

# Analysis of 4 variable

# Select users that use all of this sevices
support = df.loc[(df["OnlineBackup"] == "Yes") & (df["OnlineSecurity"] == "Yes") & (df["DeviceProtection"] == "Yes") & (df["TechSupport"] == "Yes")]
support["Churn"].mean()
support.shape
# as we can see average churn is very low

# Select users that use any of this sevices
support = df.loc[(df["OnlineBackup"] == "Yes") | (df["OnlineSecurity"] == "Yes") | (df["DeviceProtection"] == "Yes") | (df["TechSupport"] == "Yes")]
support["Churn"].mean()
support.shape

# churn is rate 0.24 in this case

# Select users that don't use any of this sevices
support = df.loc[(df["OnlineBackup"] == "No") & (df["OnlineSecurity"] == "No") & (df["DeviceProtection"] == "No") & (df["TechSupport"] == "No")]
support["Churn"].mean()
support.shape

# churn is 0.54

# Making variable about this

df.loc[:,'NO_SUPPORT']= "1"
df.loc[(df['OnlineBackup'] != 'No') | (df['DeviceProtection'] != 'No') | (df['TechSupport'] != 'No')  | (df['OnlineSecurity'] != 'No') ,'NO_SUPPORT']= "0"

# Total Services and Churn Relationship

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.groupby("TotalServices")["Churn"].mean()
# As total the services increases churn rate is decreases

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df,col)

for col in num_cols:
    num_summary(df,col)

# Target summary with categorical and numerical variables

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)

for col in num_cols:
    target_summary_with_num(df,"Churn",col)

for col in num_cols:
    box_plot(df,"Churn",col)

correlation_matrix(df,df.columns)

# Drop Features
df = df.drop(columns = ['customerID','gender'])

df.head()

# Dataprerpcessing

# Outliers
for col in num_cols:
    print(col,check_outlier(df,col))

# Missing Values
missing_values_table(df)

# Impute missing values mean of each class on MONTHLY_CHARGES_CAT
df["TotalCharges"].fillna(df.groupby("MONTHLY_CHARGES_CAT")["TotalCharges"].transform("mean"),inplace = True)

df.drop("TOTAL_CHARGES_CAT",axis = 1, inplace= True)
df["TOTAL_CHARGES_CAT"] = pd.qcut(df['TotalCharges'], 4, labels=["D","C","B","A"])
missing_values_table(df)

# Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df,col)

# Analyze relationship between rare categories and dependent variable if there is rare categories

rare_analyser(df,"Churn",cat_cols)

# It looks like we don't have any significant rare categories
# We can apply one hot encoing to all categorical variables

# One hot encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df,ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Get Useless columns if there is

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis = None)]

# Since the dataset don't have any useless cols we don't need to drop
# any column

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

df.head()

def telco_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Convert total charges to numeric
    dataframe["TOTALCHARGES"] = pd.to_numeric(dataframe["TOTALCHARGES"], errors="coerce")
    # Label churn
    dataframe["CHURN"] = dataframe["CHURN"].map({"Yes": 1, "No": 0})

    # FEATURE ENGINEEERING
    # Create has internet service columns
    dataframe["HAS_INTERNET_SERVICE"] = dataframe["INTERNETSERVICE"].apply(lambda x: "0" if x == "No" else "1")
    # Create has phone service column
    dataframe["HAS_PHONE_SERVICE"] = dataframe["PHONESERVICE"].apply(lambda x: "0" if x == "No" else "1")
    # Crate has both services columns
    dataframe.loc[(df["HAS_INTERNET_SERVICE"] == "1") & (dataframe["HAS_PHONE_SERVICE"] == "1"), "HAS_BOTH_SERVICES"] = "1"
    dataframe["HAS_BOTH_SERVICES"].fillna("0", inplace=True)

    # Convert numerical variables to categorical variables
    dataframe["TENURE_CAT"] = pd.qcut(dataframe['TENURE'], 4, labels=["D", "C", "B", "A"])
    dataframe["MONTHLY_CHARGES_CAT"] = pd.qcut(dataframe["MONTHLYCHARGES"], 4, labels=["D", "C", "B", "A"])

    # Impute missing values mean of each class on MONTHLY_CHARGES_CAT and create TOTAL_CHARGES_CAT
    dataframe["TOTALCHARGES"].fillna(dataframe.groupby("MONTHLY_CHARGES_CAT")["TOTALCHARGES"].transform("mean"),inplace = True)
    dataframe["TOTAL_CHARGES_CAT"] = pd.qcut(dataframe['TOTALCHARGES'], 4, labels=["D", "C", "B", "A"])

    # Total services columns
    dataframe['TOTALSERVICES'] = (dataframe[['PHONESERVICE', 'INTERNETSERVICE', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                               'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES']] == 'Yes').sum(axis=1)

    # LABEL ENCODING
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]
    for col in binary_cols:
        label_encoder(dataframe, col)

    # ONE HOT ENCODING
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    # SCALING
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    # Drop Features
    dataframe = dataframe.drop(columns=['CUSTOMERID', 'GENDER'])

    return dataframe

df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = telco_data_prep(df)

y = df["CHURN"]
X = df.drop(["CHURN"], axis=1)

# Base Models

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print("***********{}*********".format(name))
        for score in cv_results:
            print(score + ": " + str(cv_results[score].mean()))



base_models(X, y, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

"""
***********LR*********
test_roc_auc: 0.8446869950664593

***********KNN*********
test_roc_auc: 0.7813284324290132

***********SVC*********
test_roc_auc: 0.7958159205166994

***********CART*********
test_roc_auc: 0.6497639903220079

***********RF*********
test_roc_auc: 0.817465436006292

***********Adaboost*********
test_roc_auc: 0.8453080223123987

***********GBM*********
test_roc_auc: 0.8458080570267552

***********XGBoost*********
test_roc_auc: 0.8165039856921078

***********LightGBM*********
test_roc_auc: 0.8321867798446448

***********CatBoost*********
test_roc_auc: 0.8383582755850701

"""

# Hyperparameter Optimization
logreg_params = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'class_weight': ['balanced']}

adaboost_params = {"learning_rate" : [0.0001, 0.001, 0.01, 0.1, 1.0],
                   "n_estimators" : [10, 50, 100, 500]}

gbm_params = {"learning_rate": [0.25,0.1,0.05,0.01],
              "n_estimators" : [1, 2, 4, 8, 16, 32, 64, 100, 200],
              "max_depth" : [3,8,10,12,14],
              "min_samples_split" : [2,3,4],
              }

lightgbm_params = {"learning_rate" : [0.25,0.1,0.05,0.01],
                   "max_depth" : [-1,1,2,3,4,5],
                   "num_leaves" : [10,20,30,40,50],
                   "n_estimators": [100, 250, 300, 350, 500, 1000],
                   "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1]}


catboost_params = {"iterations": [200, 300,400],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}



classifiers = [('LR', LogisticRegression(), logreg_params),
               ("Adaboost", AdaBoostClassifier(), adaboost_params),
               ("GBM", GradientBoostingClassifier(), gbm_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params),
               ('CatBoost', LGBMClassifier(), catboost_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

"""
########## LR ##########
roc_auc (Before): 0.8447
roc_auc (After): 0.8443
LR best params: {'class_weight': 'balanced', 'solver': 'liblinear'}

########## Adaboost ##########
roc_auc (Before): 0.8453
roc_auc (After): 0.847
Adaboost best params: {'learning_rate': 0.1, 'n_estimators': 500}

########## GBM ##########
roc_auc (Before): 0.8459
roc_auc (After): 0.8468
GBM best params: {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 4, 'n_estimators': 64}

########## LightGBM ##########
roc_auc (Before): 0.8322
roc_auc (After): 0.8491
LightGBM best params: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 1, 'n_estimators': 1000, 'num_leaves': 10}

########## CatBoost ##########
roc_auc (Before): 0.8322
roc_auc (After): 0.8322
CatBoost best params: {'depth': 3, 'iterations': 200, 'learning_rate': 0.1}
"""
best_models = hyperparameter_optimization(X, y)

best_models

"""
{'LR': LogisticRegression(class_weight='balanced', solver='liblinear'),
 'Adaboost': AdaBoostClassifier(learning_rate=0.1, n_estimators=500),
 'GBM': GradientBoostingClassifier(min_samples_split=4, n_estimators=64),
 'LightGBM': LGBMClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=1,
                n_estimators=1000, num_leaves=10),
 'CatBoost': LGBMClassifier(depth=3, iterations=200)}
"""

################################################
# 6. Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

lgbm_final = LGBMClassifier(colsample_bytree=0.8, learning_rate=0.05, max_depth=1,n_estimators=1000, num_leaves=10).fit(X,y)
adaboost_final = AdaBoostClassifier(learning_rate=0.1, n_estimators=500).fit(X,y)
gbm_final = GradientBoostingClassifier(min_samples_split=4, n_estimators=64).fit(X,y)

plot_importance(lgbm_final,X,num = 20,save = True)
plot_importance(adaboost_final,X,num = 20, save=True)
plot_importance(gbm_final,X,num = 20, save=True)

# Stacking & Ensemble Learning

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('LightGBM', best_models["LightGBM"]), ('Adaboost', best_models["Adaboost"]),
                                              ('GBM', best_models["GBM"])],
                                  voting='soft').fit(X, y)
    cv_results_clf = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
    print(f"Accuracy: {cv_results_clf['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results_clf['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results_clf['test_roc_auc'].mean()}")
    print(f"Precision: {cv_results_clf['test_precision'].mean()}")
    print(f"Recall: {cv_results_clf['test_recall'].mean()}")
    return voting_clf


voting_clf = voting_classifier(best_models, X, y)

"""
Accuracy: 0.8071839266094646
F1Score: 0.5928969937987071
ROC_AUC: 0.8491753982853968
Precision: 0.6741733499727741
Recall: 0.5291599785981808
"""
# Prediction

for i in range(1,100):
    random_customer = X.sample(1, random_state=i)
    print(voting_clf.predict(random_customer))









