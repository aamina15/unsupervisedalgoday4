import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

# -----------------------------------
# Custom CSS
# -----------------------------------

st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#f5f7fa;
}

/* Main title */
h1{
    color:#0F4C81;
    text-align:center;
    font-weight:bold;
}

/* Sub headings */
h2,h3{
    color:#0F4C81;
}

/* Buttons */
div.stButton > button{
    background-color:#0F4C81;
    color:white;
    border-radius:10px;
    border:none;
    font-size:16px;
    font-weight:bold;
    padding:10px 18px;
}

div.stButton > button:hover{
    background-color:#155A8A;
}

/* Cards */
.info-box{
    background:white;
    padding:18px;
    border-radius:12px;
    border-left:6px solid #0F4C81;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Title
# -----------------------------------

st.title("📊 Customer Segmentation using K-Means Clustering")

st.markdown("""
<div class="info-box">

### 👩‍💻 Student Details

**Name:** Aamina Hasan

**Branch:** Computer Science Engineering (CSE)

**Room No.:** 612

</div>
""", unsafe_allow_html=True)

st.write(
    "This project groups customers into clusters based on their **Age** and **Income** using the K-Means Clustering algorithm."
)

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("income.csv")

# Keep original copy for first graph
original_df = df.copy()

with st.expander("📄 View Dataset"):
    st.dataframe(df)

# -----------------------------------
# Original Scatter Plot
# -----------------------------------

st.subheader("📈 Original Customer Distribution")

fig, ax = plt.subplots(figsize=(6,4))

ax.scatter(original_df["Age"], original_df["Income($)"])

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.set_title("Age vs Income")

st.pyplot(fig)

# -----------------------------------
# Feature Scaling
# -----------------------------------

scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["Income($)"] = scaler.fit_transform(df[["Income($)"]])

# -----------------------------------
# Cluster Selection
# -----------------------------------

st.subheader("⚙️ Choose Number of Clusters")

st.info("Move the slider to change the number of clusters.")

n_clusters = st.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=3
)

# -----------------------------------
# Train KMeans
# -----------------------------------

km = KMeans(
    n_clusters=n_clusters,
    random_state=42
)

df["Cluster"] = km.fit_predict(df[["Age","Income($)"]])

st.success(f"Successfully created **{n_clusters}** clusters.")

# -----------------------------------
# Clustered Dataset
# -----------------------------------

with st.expander("📄 View Clustered Dataset"):
    st.dataframe(df)

# -----------------------------------
# Cluster Plot
# -----------------------------------

st.subheader("🎯 Cluster Visualization")

fig, ax = plt.subplots(figsize=(6,5))

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
    marker="*",
    color="black",
    s=250,
    label="Centroids"
)

ax.set_xlabel("Scaled Age")
ax.set_ylabel("Scaled Income")

ax.legend()

st.pyplot(fig)

# -----------------------------------
# Elbow Method
# -----------------------------------

st.subheader("📉 Elbow Method")

sse = []

for k in range(1,11):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(df[["Age","Income($)"]])

    sse.append(model.inertia_)

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(range(1,11), sse, marker="o")

ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("SSE")
ax.set_title("Elbow Method")

st.pyplot(fig)

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")
st.caption("Machine Learning Project | K-Means Clustering | Aamina Hasan")
