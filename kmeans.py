import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Customer Segmentation using K-Means",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Segmentation using K-Means")

st.markdown("""
**Name:** Aamina Hasan  
**Branch:** Computer Science Engineering (CSE)  
**Room No.:** 612
""")

st.write("This project groups customers based on their Age and Income using the K-Means Clustering algorithm.")

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("income.csv")

st.subheader("Dataset")
st.dataframe(df)

# -----------------------------------
# Original Scatter Plot
# -----------------------------------

st.subheader("Original Data")

fig, ax = plt.subplots()

ax.scatter(df["Age"], df["Income($)"])

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")

st.pyplot(fig)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["Income($)"] = scaler.fit_transform(df[["Income($)"]])

# -----------------------------------
# Select Number of Clusters
# -----------------------------------

st.subheader("Select Number of Clusters")

n_clusters = st.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

# -----------------------------------
# Apply K-Means
# -----------------------------------

km = KMeans(n_clusters=n_clusters, random_state=42)

df["Cluster"] = km.fit_predict(df[["Age", "Income($)"]])

# -----------------------------------
# Clustered Dataset
# -----------------------------------

st.subheader("Clustered Dataset")

st.dataframe(df)

# -----------------------------------
# Cluster Visualization
# -----------------------------------

st.subheader("K-Means Clustering")

fig, ax = plt.subplots()

colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "cyan",
    "gray",
    "olive"
]

for i in range(n_clusters):

    cluster = df[df["Cluster"] == i]

    ax.scatter(
        cluster["Age"],
        cluster["Income($)"],
        color=colors[i],
        label=f"Cluster {i+1}"
    )

ax.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    color="black",
    marker="*",
    s=250,
    label="Centroids"
)

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.legend()

st.pyplot(fig)

# -----------------------------------
# Elbow Method
# -----------------------------------

st.subheader("Elbow Method")

sse = []

k_range = range(1,11)

for k in k_range:

    model = KMeans(n_clusters=k, random_state=42)

    model.fit(df[["Age","Income($)"]])

    sse.append(model.inertia_)

fig, ax = plt.subplots()

ax.plot(k_range, sse, marker="o")

ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Sum of Squared Errors (SSE)")
ax.set_title("Elbow Method")

st.pyplot(fig)
