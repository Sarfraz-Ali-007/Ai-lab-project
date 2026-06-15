import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Salary Prediction",
    layout="wide"
)

model = joblib.load(
    "models/best_salary_model.pkl"
)

st.title("Employee Salary Prediction System")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    st.subheader("Salary Distribution")

    fig = px.histogram(
        df,
        x="Salary"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if "Years of Experience" in df.columns:

        st.subheader(
            "Experience vs Salary"
        )

        fig2 = px.scatter(
            df,
            x="Years of Experience",
            y="Salary"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

st.divider()

st.header("Predict Salary")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    [
        "Bachelor",
        "Master",
        "PhD"
    ]
)

job_title = st.text_input(
    "Job Title",
    "Data Analyst"
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=2
)

if st.button("Predict Salary"):

    sample = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience]
    })

    prediction = model.predict(sample)[0]

    st.success(
        f"Predicted Salary: ${prediction:,.2f}"
    )
