#%%
import json
import requests
import streamlit as st

st.title("Credit Worthiness Checker")
st.write("Please input the following details to check your credit worthiness")
gender = st.selectbox(label="Please choose one of the following options", options= ["M", "F"])
age = st.number_input(label="Please enter your age", min_value=16, max_value=150)
income = st.number_input(label="Please enter your income", min_value=0)
no_of_children = st.number_input(label="Please enter the number of children you have", min_value=0, max_value=20)

data = {
    "gender":gender,
    "age":age,
    "income":income,
    "no_of_children":no_of_children
}

if st.button("Check Credit Worthiness"):
    response = requests.post(url='http://127.77.56.1:',timeout=10, data=json.dumps(data))
    output = response.text
    if output == "0":
        st.subheader("Congratulations! You are credit worthy")
    else:
        st.subheader("Sorry, you are not credit worthy")
    #st.subheader(f"Response from the API: {response.text}")
else:
    pass