import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Simple CSS
# -----------------------------

st.markdown("""
<style>
.stApp{
    background-color:#F7F9FC;
}

h1{
    color:#1F4E79;
    text-align:center;
}

div.stButton > button{
    background:#1F77B4;
    color:white;
    border-radius:8px;
    border:none;
}

div.stButton > button:hover{
    background:#145A86;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("👩‍💻 Developer")

st.sidebar.write("**Aamina Hasan**")
st.sidebar.write("Computer Science Engineering")


st.sidebar.markdown("---")

st.sidebar.markdown(
"""
**LinkedIn**

https://www.linkedin.com/in/aamina-hasan-50a6a930b/
"""
)

# -----------------------------
# Title
# -----------------------------

st.title("📊 Customer Segmentation using K-Means")

st.write(
    "Group customers based on **Age** and **Income** using the K-Means Clustering algorithm."
)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("income.csv")

original_df = df.copy()

with st.expander("📄 View Dataset"):
    st.dataframe(df)

st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")

# -----------------------------
# Original Graph
# -----------------------------

st.subheader("Original Data")

fig, ax = plt.subplots()

ax.scatter(original_df["Age"], original_df["Income($)"])

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")

st.pyplot(fig)

# -----------------------------
# Scaling
# -----------------------------

scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["Income($)"] = scaler.fit_transform(df[["Income($)"]])

# -----------------------------
# Select Clusters
# -----------------------------

st.subheader("Choose Number of Clusters")

n_clusters = st.slider(
    "Clusters",
    2,
    10,
    3
)

# -----------------------------
# KMeans
# -----------------------------

km = KMeans(
    n_clusters=n_clusters,
    random_state=42
)

df["Cluster"] = km.fit_predict(df[["Age","Income($)"]])

st.success(f"{n_clusters} clusters created successfully.")

# -----------------------------
# Cluster Dataset
# -----------------------------

with st.expander("📄 Clustered Dataset"):
    st.dataframe(df)

# -----------------------------
# Cluster Plot
# -----------------------------

st.subheader("Cluster Visualization")

colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan"
]

fig, ax = plt.subplots()

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
    s=200,
    label="Centroids"
)

ax.set_xlabel("Scaled Age")
ax.set_ylabel("Scaled Income")

ax.legend()

st.pyplot(fig)

# -----------------------------
# Elbow Method
# -----------------------------

st.subheader("Elbow Method")

sse = []

for k in range(1,11):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(df[["Age","Income($)"]])

    sse.append(model.inertia_)

fig, ax = plt.subplots()

ax.plot(range(1,11), sse, marker="o")

ax.set_xlabel("K")
ax.set_ylabel("SSE")

st.pyplot(fig)

# -----------------------------
# Cluster Summary
# -----------------------------

# -----------------------------
# Cluster Summary
# -----------------------------

st.subheader("📊 Cluster Summary")

cluster_summary = df.groupby("Cluster").agg(
    Average_Age=("Age", "mean"),
    Average_Income=("Income($)", "mean"),
    Customers=("Cluster", "count")
)

st.dataframe(cluster_summary)

# -----------------------------
# Customers in Each Cluster
# -----------------------------

st.subheader("👥 Customers in Each Cluster")

for i in range(n_clusters):
    with st.expander(f"Cluster {i+1}"):
        st.dataframe(
            df[df["Cluster"] == i][["Name", "Age", "Income($)"]]
        )

# -----------------------------
# Project Description
# -----------------------------

st.markdown("---")

st.subheader("Project Description")

st.write(
"""
This project uses the **K-Means Clustering** algorithm to group customers
based on their **Age** and **Income**. Data is first normalized using
**MinMaxScaler**, and users can interactively choose the number of clusters.
The Elbow Method helps identify an appropriate value of K.
"""
)

st.subheader("Technologies Used")

st.write("""
- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
""")

st.markdown("---")

st.markdown(
"""
**Developed by:** Aamina Hasan

🔗 LinkedIn: https://www.linkedin.com/in/aamina-hasan-50a6a930b/
"""
)
