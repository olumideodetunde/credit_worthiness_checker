import json
import requests
import streamlit as st

st.title("Credit Worthiness Checker")
st.write("Please input the following details to check your credit worthiness")
gender = st.selectbox(label="Please choose one of the following options", options= ["M", "F"])
age = st.number_input(label="Please enter your age", min_value=16, max_value=150)
income = st.number_input(label="Please enter your income", min_value=0)
no_of_children = st.number_input(label="Please enter the number of children you have", min_value=0,
                                 max_value=20)

data = {
    "gender":gender,
    "age":age,
    "income":income,
    "no_of_children":no_of_children
}

if st.button("Check Credit Worthiness"):
    response = requests.post(url='https://credit-worthiness-checker-backend.fly.dev/predict',timeout=10, data=json.dumps(data))
    output = response.json()["prediction"]
    if output == 0:
        st.subheader(f"The model predicts {output} which means you are credit worthy. Congratulations!")
    else:
        st.subheader(f"The model predicts {output} which means you are not credit worthy. Sorry! You are not credit worthy")
else:
    pass
