import streamlit as st
import pandas as pd
import pickle
import numpy as np
import groq
import utils as ut
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Correctly retrieve API key from environment variables and initialize OpenAI client
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise Exception("GROQ_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Initialize Groq client
groq_client = groq.Client(api_key=api_key)

# Function to load pre-trained models from pickle files
def load_model(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load model {filename}: {str(e)}")
        return None

# Load all pre-trained models
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_clafier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEnginer_model = load_model('xgboost-featureEngineered.pkl')

# Function to prepare input data for prediction
def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# Function to make predictions using selected models
def make_predictions(input_df, input_dict):
    try:
        probability = {
            'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
            'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
            'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
        }
        avg_probability = np.mean(list(probability.values()))
        fig = ut.create_gauge_chart(avg_probability)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"The customer has a {avg_probability*100:.2%}% chance of churning.")
        with col2:
            fig_probs = ut.create_model_probability_chart(probability)
            st.plotly_chart(fig_probs, use_container_width=True)
        return avg_probability
    except Exception as e:
        st.error(f"Error in making predictions: {str(e)}")
        return None

# Function to explain the prediction based on customer information and model output
def explain_prediction(probability, input_dict, surname):
  # Construct a prompt for OpenAI model to generate an explanation
  prompt = f"""
  You are a senior data scientist at a bank specializing in customer behavior analysis and churn predictions. Your responsibility is to interpret and explain the predictions made by machine learning models using financial and demographic data of customers without stating the probabilities during your explanations.

  The model has calculated that **{surname}** has a **{round(probability*100, 1)}%** chance of churning. As an expert, explain why this customer is likely to stay or leave, considering their financial habits, engagement with the bank's products, and any relevant demographic factors. strickly avoid mentioning machine learning models or their features directly in the explanation.

  ### Customer Information:
  {input_dict}
  ### Top 10 Influencing Factors for Churn Prediction:
  
  | **Feature**            | **Importance** | 
  |------------------------|----------------|
  | NumOfProducts          | 0.323888       |
  | IsActiveMember         | 0.164146       |
  | Age                    | 0.109550       |
  | Geography_Germany      | 0.091373       |
  | Balance                | 0.052786       |
  | Geography_France       | 0.046463       |
  | Gender_Female          | 0.045283       |
  | Geography_Spain        | 0.036855       |
  | CreditScore            | 0.035005       |
  | EstimatedSalary        | 0.032655       |
  | Tenure                 | 0.029909       |
  | HasCrCard              | 0.028933       |
  | -----------------------|---------------|


  {pd.set_option ('display.max_columns', None)}

  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

### Instructions for generating the explanation:

  - If the customer's probability of churning is **above 40%**, explain in 3 sentences why they are at a higher risk of churning. Focus on the key demographic and financial behaviors that suggest disengagement or dissatisfaction.
  
  - If the customer's probability of churning is **below 40%**, explain in 3 sentences why they are likely to stay. Highlight their engagement with the bank's products and any positive demographic trends that indicate stability.

  - The explanation should be **personalized**, mentioning specific details about the customer without referring to the machine learning model or its features explicitly. The aim is to give a human-centered, data-backed rationale without directly pointing to the technical aspects of the prediction.

  ### General Customer Insights:
  {pd.set_option('display.max_columns', None)}

  Provide a concise and business-friendly explanation to communicate the prediction results in a clear, understandable way to business stakeholders or the customer support team.
  
  """

  print("EXPLANATION PROMPT", prompt)

  # Use OpenAI's chat completion API to generate an explanation
  raw_response = client.chat.completions.create (
    model="llama-3.2-3b-preview",
    messages=[{
        "role": "user",
        "content": prompt
      }],
  )
  return raw_response.choices[0].message.content

# Function to generate an email based on customer information and explanation
def generate_email(probability, input_dict, surname, explanation):
  
  # Construct a prompt for OpenAI model to generate an email
  prompt = f"""
  You are a customer relationship manager at XYZ Bank. Your primary goal is to ensure customer satisfaction and retention by offering personalized incentives and support based on their needs.

  You have identified that the customer **{surname}** has a significant risk of disengagement, and you've reviewed their financial profile closely. Without mentioning probabilities or predictive models, draft an email to address this customer’s potential concerns and offer targeted incentives to encourage them to continue banking with XYZ Bank.

  ### Customer Information:
  {input_dict}

  ### Explanation of Risk:
  Based on the analysis of {surname}'s banking habits and financial profile, there are a few areas where the customer may feel dissatisfied or could benefit from additional services:
  {explanation}

  Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

  Make sure to list out a st of incentives to stay based on their information, in bullets poitns format. don't ever mention the probability of churning or the machine learning model to the customer.


  """

  raw_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{
      "role": "user",
      "content": prompt
    }],
  )
  print("\n]nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content


# Streamlit UI code
st.title("Customer Churn Prediction")
try:
    df = pd.read_csv("churn.csv")
    # UI for selecting customers
    customers = {f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()}
    selected_customer_option = st.selectbox("Select a Customer", customers)

    if selected_customer_option:
        selected_customer_id = int(selected_customer_option.split(" - ")[0])
        selected_customer_row = selected_customer_option.split(" - ")[1]

        selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=int(selected_customer['CreditScore']))
            location = st.selectbox("Location", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(selected_customer['Geography']))
            gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == "Male" else 1)
            age = st.number_input("Age", min_value=18, max_value=100, value=int(selected_customer['Age']))
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=int(selected_customer['Tenure']))

        with col2:
            balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer['Balance']))
            num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts']))
            has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard']))
            is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember']))
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer['EstimatedSalary']))

    # Prepare input data
    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

    # Make predictions
    avg_probability = make_predictions(input_df, input_dict)

    # Get explanation from OpenAI model
    explanation = explain_prediction(input_df, input_dict, selected_customer['Surname'])

    st.markdown("---")

    # Display explanation section
    st.subheader("Explanation of Prediction")

    st.markdown(explanation)

    # Generate email based on explanation
    email = generate_email(avg_probability, input_dict, selected_customer['Surname'], explanation)
    
    st.markdown("---")
    
    # Display email section
    st.subheader("Email")
    
    st.markdown(email)
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
