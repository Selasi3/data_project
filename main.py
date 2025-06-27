import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model, preprocessor, selector
@st.cache_resource
def load_objects():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    return model, preprocessor, selector

model, preprocessor, selector = load_objects()

st.title("Bank Term Deposit Subscription Prediction")

def user_input_features():
    age = st.number_input('Age', 17, 95, 35)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
                               'self-employed', 'services', 'student', 'technician', 'unemployed'])
    marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
    education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                                           'professional.course', 'university.degree'])
    default = st.selectbox('Has credit in default?', ['no', 'yes'])
    housing = st.selectbox('Has housing loan?', ['no', 'yes'])
    loan = st.selectbox('Has personal loan?', ['no', 'yes'])
    contact = st.selectbox('Contact communication type', ['cellular', 'telephone'])
    month = st.selectbox('Last contact month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Last contact day of week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.number_input('Last contact duration (seconds)', 0, 4918, 180)
    campaign = st.number_input('Number of contacts during this campaign', 1, 43, 1)
    pdays = st.number_input('Days since last contact (-1 if not previously contacted)', -1, 999, 999)
    previous = st.number_input('Number of contacts before this campaign', 0, 7, 0)
    poutcome = st.selectbox('Outcome of previous campaign', ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.slider('Employment variation rate', -3.4, 1.4, 1.1, 0.1)
    cons_price_idx = st.slider('Consumer price index', 92.2, 94.8, 93.5, 0.001)
    cons_conf_idx = st.slider('Consumer confidence index', -50.8, -26.9, -40.0, 0.001)
    euribor3m = st.slider('3 month Euribor rate', 0.6, 5.0, 4.9, 0.001)
    nr_employed = st.slider('Number of employees', 4963, 5228, 5160, 1)

    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }
    features = pd.DataFrame(data, index=[0])

    # Create missing engineered features
    features['contact_ratio'] = features['campaign'] / (features['previous'] + 1)
    # Assuming no detailed data for success_ratio, set to 0 or adjust if you have info
    features['success_ratio'] = 0
    # y_num is target variable, not needed for prediction, so omit

    return features

input_df = user_input_features()

st.subheader('Client Input Features')
st.write(input_df)

if st.button('Predict'):
    try:
        # Preprocess input
        input_preprocessed = preprocessor.transform(input_df)
        input_selected = selector.transform(input_preprocessed)

        # Predict
        prediction = model.predict(input_selected)
        prediction_proba = model.predict_proba(input_selected)

        st.subheader('Prediction')
        if prediction[0] == 1:
            st.success(f"The client is likely to subscribe to the term deposit (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.warning(f"The client is unlikely to subscribe (Probability: {prediction_proba[0][0]:.2f})")

        st.subheader('Prediction Probabilities')
        st.write(f"Probability of No Subscription: {prediction_proba[0][0]:.4f}")
        st.write(f"Probability of Subscription: {prediction_proba[0][1]:.4f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Please ensure all required features are provided correctly.")
