<center>

# Credit Worthiness Checker

</center>


This repository holds programs files required to train and deploy a credit worthiness cheker as an ml service endpoint. The final frontend that uses the backend service can be accessed here [credit worthiness checker](credit_worthiness_checker.streamlit.app)

This project has been streamlined into 2 main parts:
1. The data science lifecycle - ending with an exported model

2. The deployment - further divided into 2 parts:
    - The backend service
    - The frontend
    
# Requirements
Python 3.8+

# Replicating the project
1. Clone this repository to your local machine using the command below:

```
git clone https://github.com/olumideodetunde/credit_worthiness_checker.git
```

2. Download the dataset from kaggle and place the data appropriately    

    - Click [here](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/data) to download the data & place it in the artifacts/data/raw directory.

3. Run the data science pipeline

    - Run the command leveraging makefile: this would prepare the data, generate defined visualisations, engineer features, train the model and save the model to the deploy directory.

        ```bash
        make machinelearningmodel
        ```

4. Deploy the backend service

    - This project used FastAPI to deploy the backend service, To host the backend service, this project leveraged the the dockerfile in the backend directory and follow this [link](https://fly.io/docs/languages-and-frameworks/dockerfile/).


5. Build the frontend
    
    - Leveraging the streamlit script in the frontend directory, you can build the frontend by following the steps found in this link [streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)


