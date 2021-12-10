import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from helpers.eda import *
from helpers.data_prep import *
import warnings
warnings.filterwarnings('ignore')
import os

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
    dataframe.loc[(dataframe["HAS_INTERNET_SERVICE"] == "1") & (dataframe["HAS_PHONE_SERVICE"] == "1"), "HAS_BOTH_SERVICES"] = "1"
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

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('LightGBM', best_models["LightGBM"]), ('Adaboost', best_models["Adaboost"]),
                                              ('GBM', best_models["GBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc","precision","recall"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    return voting_clf

# Pipeline Main Function

def main():
    df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = telco_data_prep(df)
    y = df["CHURN"]
    X = df.drop(["CHURN"], axis=1)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    # os.chdir("/Users/Fikri/Desktop/vbo_bootcamp/8hafta/")
    joblib.dump(voting_clf, "voting_clf_telco.pkl")
    print("Voting_clf has been created")
    return voting_clf

if __name__ == "__main__":
    main()



