!pip install streamlit pandas plotly seaborn prophet transformers

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from transformers import pipeline

# Load Data
@st.cache_data
def load_data():
    df=pd.read_csv("Sales Data.csv",index_col=0)  # Update with your actual file path
    df['Order Date'] = pd.to_datetime(df['Order Date'],format='ISO8601')
    df['Month'] = df['Order Date'].dt.month
    df['Hour'] = df['Order Date'].dt.hour
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Data")
selected_city = st.sidebar.multiselect("Select City", df["City"].unique(), default=df["City"].unique())
df_filtered = df[df["City"].isin(selected_city)]

# KPIs
st.title("📊 Sales Dashboard")
st.metric("Total Sales", f"${df_filtered['Sales'].sum():,.2f}")
st.metric("Average Order Value", f"${df_filtered['Sales'].mean():,.2f}")

# Sales by Month
st.subheader("📅 Monthly Sales Trend")
monthly_sales = df_filtered.groupby("Month")["Sales"].sum().reset_index()
fig1 = px.line(monthly_sales, x="Month", y="Sales", markers=True, title="Sales Trend by Month")
st.plotly_chart(fig1)

# Sales by City
st.subheader("🏙️ City-wise Sales Distribution")
city_sales = df_filtered.groupby("City")["Sales"].sum().reset_index()
fig2 = px.bar(city_sales, x="City", y="Sales", title="Sales by City", color="Sales")
st.plotly_chart(fig2)

# Sales by Product
st.subheader("🛒 Top-Selling Products")
product_sales = df_filtered.groupby("Product")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False)
fig3 = px.bar(product_sales.head(10), x="Product", y="Sales", title="Top 10 Products")
st.plotly_chart(fig3)

# Forecasting using Prophet
st.subheader("🔮 Sales Forecasting")
df_forecast = df_filtered.groupby("Order Date").sum()["Sales"].reset_index()
df_forecast.columns = ["ds", "y"]

model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig4 = px.line(forecast, x="ds", y="yhat", title="Predicted Sales for Next 30 Days")
st.plotly_chart(fig4)

# AI Q&A System
st.subheader("🤖 Ask AI about Sales Insights")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = f"""
Total Sales: ${df_filtered['Sales'].sum():,.2f}
Average Order Value: ${df_filtered['Sales'].mean():,.2f}
Top-Selling Products: {', '.join(product_sales.head(5)['Product'])}
Peak Purchase Hours: {df_filtered['Hour'].value_counts().index[:3].tolist()}
"""

question = st.text_input("Ask a question about sales (e.g., 'Why are sales stagnant?')")
if question:
    response = qa_pipeline(question=question, context=context)
    st.write("**AI Answer:**", response["answer"])

# Run using: streamlit run app.py
