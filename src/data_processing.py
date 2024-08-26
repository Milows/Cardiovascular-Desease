import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cardio_df = pd.read_csv("data\\raw\cardio.csv", sep=";", index_col="id")
cardio_df["age"] = (cardio_df["age"] / 365).round().astype("int")
cardio_df["gender"] = cardio_df["gender"].replace({1:0, 2:1})
cardio_df = cardio_df.drop_duplicates()
cardio_df.drop(cardio_df[cardio_df["ap_hi"] < 30].index, inplace=True)
cardio_df.drop(cardio_df[cardio_df["ap_hi"] > 220].index, inplace=True)
cardio_df.drop(cardio_df[cardio_df["ap_lo"] < 30].index, inplace=True)
cardio_df.drop(cardio_df[cardio_df["ap_lo"] > 160].index, inplace=True)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(cardio_df)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
cardio_df['cluster'] = clusters

os.makedirs("data\\processed", exist_ok=True)
cardio_df.to_csv("data\\processed\cardio_data_processed.csv", index=False)