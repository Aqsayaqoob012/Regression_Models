# =========================
# Salary Analysis App
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split



# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Salary Data Analysis",
    layout="wide"
)

# Hide Streamlit style (menu, header, footer)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}   /* top-right menu */
            footer {visibility: hidden;}     /* bottom footer */
            header {visibility: hidden;}     /* top header */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# =========================
# App Content
# =========================
st.title("Salary Data Analysis & Prediction App")
st.markdown("Exploratory Data Analysis, Feature Engineering & Regression Models Comparison")

# =========================
# Load Dataset
# =========================
st.header(" Load Dataset")

df = pd.read_csv("Salary_Data.csv")
st.success("Dataset Loaded Successfully!")

st.subheader(" Dataset Preview")
# Original head
df_head = df.head()

# Index 1 se start
df_head.index = df_head.index + 1

# Optional: Index column name
df_head.index.name = "S.No"

# Display in Streamlit
st.dataframe(df_head)

# =========================
# Dataset Information
# =========================



st.header("Dataset Information")

# -----------------------------
# Null Values & Data Types Table
# -----------------------------
info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes,
    "Non-Null Count": df.notnull().sum(),
    "Null Count": df.isnull().sum()
})

st.subheader("Data Types & Null Values")
st.dataframe(info_df)  # nice interactive table


st.subheader("Statistical Summary")
st.dataframe(df.describe())

# =========================
# Missing Values
# =========================
st.header(" Missing Value Analysis")

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    st.success("No missing values found 🎉")
else:
    st.warning("Missing Values Found")

    # Convert to DataFrame for better formatting
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Count"]

    # Use st.dataframe with max_column_width
    st.dataframe(
        missing_df.style.set_table_styles(
            [{
                'selector': 'th',
                'props': [('max-width', '100px')]
            }, {
                'selector': 'td',
                'props': [('max-width', '100px')]
            }]
        )
    )

# Drop missing rows
df = df.dropna()


# =========================
# Distribution Plots
# =========================
# ==============================
# Feature Distribution Section
# ==============================

st.header(" Feature Distributions")

fig, ax = plt.subplots(3, 1, figsize=(10, 10.5))


for a in ax:
    a.tick_params(axis='both', labelsize=4)  # ticks choty

# Age
sns.histplot(df['Age'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title("Age Distribution", fontsize=5)
ax[0].set_xlabel("Age", fontsize=5)
ax[0].set_ylabel("Count", fontsize=5)

# Salary
sns.histplot(df['Salary'], kde=True, ax=ax[1], color='lightgreen')
ax[1].set_title("Salary Distribution", fontsize=5)
ax[1].set_xlabel("Salary", fontsize=5)
ax[1].set_ylabel("Count", fontsize=5)

# Experience
sns.histplot(df['Years of Experience'], kde=True, ax=ax[2], color='salmon')
ax[2].set_title("Experience Distribution", fontsize=5)
ax[2].set_xlabel("Years of Experience", fontsize=5)
ax[2].set_ylabel("Count", fontsize=5)

plt.tight_layout()
st.pyplot(fig)


# VERY IMPORTANT for Streamlit



# =========================
# Education Level Analysis
# =========================
st.header("Education Level Analysis")

fig = plt.figure(figsize=(11,3.5))

sns.countplot(x='Education Level', data=df, palette='pastel', edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.6)
st.pyplot(fig)

# Remove PhD & Rename
df = df[df['Education Level'] != 'phD']
df['Education Level'] = df['Education Level'].replace({
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree"
})

# =========================
# Gender Analysis
# =========================
st.header(" Gender Distribution")

fig = plt.figure(figsize=(11,3.5))
sns.countplot(x='Gender', data=df, palette='pastel', edgecolor='black')
plt.grid(axis='y', linestyle='--', alpha=0.6)
st.pyplot(fig)

# =========================
# Job Title Analysis
# =========================

st.header(" Job Title Analysis") # Top 15 Job Titles Pie Chart
top_jobs = df['Job Title'].value_counts().head(15)
fig = plt.figure(figsize=(3,5))
 # Figure aur chhoti 
 # fig = plt.figure(figsize=(4,4))
 #  # square figure
plt.pie( top_jobs, labels=top_jobs.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette('pastel', 20),
          textprops={'fontsize': 3}, radius=0.7)
plt.title("Top 15 Job Titles", fontsize=6) 
plt.tight_layout() 
st.pyplot(fig)



# =========================
# Salary Relationships
# =========================
st.header("Salary Relationships")

fig = plt.figure(figsize=(11,3.5))
sns.lineplot(x='Years of Experience', y='Salary', data=df)
st.pyplot(fig)

fig = plt.figure(figsize=(11,3.5))
sns.lineplot(x='Age', y='Salary', data=df)
st.pyplot(fig)

# =========================
# Outlier Detection
# =========================
st.header(" Outlier Handling")

num_features = df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

st.success("Outliers handled using IQR clipping")

# =========================
# Data Transformation
# =========================
st.header(" Data Transformation")

df['Years of Experience'] += 1

df['Age'], _ = stats.boxcox(df['Age'])
df['Years of Experience'], _ = stats.boxcox(df['Years of Experience'])

pt = PowerTransformer(method='yeo-johnson')
df['Salary'] = pt.fit_transform(df[['Salary']])

# =========================
# Model Preparation
# =========================
st.header("Model Training")

y = df['Salary']
X = df.drop('Salary', axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Metrics storage
results = []

# =============================
# Train & Evaluate Models
# =============================
for name, reg in models.items():
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    results.append({
        "Model": name,
        "Train R²": f"{r2_train*100:.2f}%",
        "Test R²": f"{r2_test*100:.2f}%",
        "MAE": f"{mae:.2f}",
        "MSE": f"{mse:.2f}",
        "RMSE": f"{rmse:.2f}"
    })

# =============================
# Display in Streamlit
# =============================
st.header(" Regression Models Comparison")

# Metric cards
st.subheader("R² Score Overview")
cols = st.columns(len(models))
for i, res in enumerate(results):
    cols[i].metric(res["Model"], f"Test R²: {res['Test R²']}")

# Full table with index starting from 1
st.subheader("Detailed Metrics")
df_results = pd.DataFrame(results)

# Set index to start from 1
df_results.index = df_results.index + 1

st.dataframe(df_results)
