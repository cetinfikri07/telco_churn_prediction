import joblib
import pandas as pd
from proje.telco_ml_pipeline import telco_data_prep
import warnings
warnings.filterwarnings('ignore')

def load(path):
    df = pd.read_csv(path)
    return df

df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = telco_data_prep(df)

y = df["CHURN"]
X = df.drop(["CHURN"], axis=1)

random_user = X.sample(1, random_state=45)
new_model = joblib.load("voting_clf_telco.pkl")
new_model.predict(random_user)

