import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page Config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("Mall Customer Segmentation Dashboard")
st.write("K-Means Clustering for Customer Segmentation")

# Load Data
data = pd.read_csv("Mall_Customers.csv")

# Convert to Rupees
data["Annual Income (₹)"] = data["Annual Income (k$)"] * 1000 * 83

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

with col2:
    st.subheader("Select Cluster Settings")
    k = st.slider("Number of Clusters", 2, 10, 5)

# Features
X = data[["Annual Income (₹)", "Spending Score (1-100)"]]

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
data["Cluster"] = kmeans.fit_predict(X)

# Graph
st.subheader("Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(10,6))

for i in range(k):
    ax.scatter(
        data[data["Cluster"] == i]["Annual Income (₹)"],
        data[data["Cluster"] == i]["Spending Score (1-100)"],
        label=f"Cluster {i+1}"
    )

# Centroids
centroids = kmeans.cluster_centers_

ax.scatter(
    centroids[:,0],
    centroids[:,1],
    s=300,
    c='black',
    label="Centroids"
)

ax.set_xlabel("Annual Income (₹)")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)

# Cluster Summary
st.subheader("Cluster Summary")

cluster_summary = data.groupby("Cluster")[[
    "Age",
    "Annual Income (₹)",
    "Spending Score (1-100)"
]].mean()
st.dataframe(cluster_summary)

# Download Button
st.subheader("Download Clustered Data")

csv = data.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="Customer_Segmentation.csv",
    mime="text/csv"
)