import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cardio_df = pd.read_csv("data\\processed\cardio_data_processed.csv")
X = cardio_df.drop('cardio', axis=1)
y = cardio_df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

joblib.dump(logistic_model, "models\\trained_model_1.pkl")
joblib.dump(scaler, "models\scaler.pkl")

