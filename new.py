import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirement.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

# Call the function to install packages
install_requirements()

import streamlit as st
import joblib
import pandas as pd
import pickle
from datetime import datetime, date

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Assessment App",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Define the feature engineering function
def feature_engineering(df):
    df["Years_Employed"] = df["Employed_days"] // 365
    df["Birthday_count"] = df["Birthday_count"].abs()
    df["Income_per_Family_Member"] = df["Annual_income"] / df["Family_Members"]
    df["Children_per_Family_Member"] = df["CHILDREN"] / df["Family_Members"]
    df["Is_Employed"] = df["Employed_days"] > 0
    return df


# Load the pipeline
pipeline = joblib.load("preprocessing_pipeline_ulang.pkl")

with open("catboost_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
optimal_threshold = model_data["optimal_threshold"]


# Contoh penggunaan model dan threshold untuk prediksi baru
def predict_with_threshold(X_new):
    y_probs = model.predict_proba(X_new)[:, 1]
    y_pred = (y_probs >= optimal_threshold).astype(int)
    return y_pred, y_probs


# Function to get user input
def get_user_input():
    st.sidebar.header("Applicant Information")

    # New section to ask employment status
    employment_status = st.sidebar.selectbox(
        "Enter Current Employment Status", ["Employed", "Unemployed"]
    )

    # Capture input values
    birth_date = st.sidebar.date_input("Enter Birth Date")

    # Calculate `Birthday_count` in days
    today = date.today()
    birthday_count = (today - birth_date).days

    user_input = {
        "Employed_days": st.sidebar.number_input(
            "Enter Employed Days", min_value=0, step=1
        ),
        "Birthday_count": birthday_count,
        "Annual_income": st.sidebar.number_input(
            "Enter Annual Income", min_value=0.0, step=0.01
        ),
        "Family_Members": st.sidebar.number_input(
            "Enter Number of Family Members", min_value=1, step=1
        ),
        "CHILDREN": st.sidebar.number_input(
            "Enter Number of Children", min_value=0, step=1
        ),
        "Civil marriage": st.sidebar.selectbox(
            "Select Civil Marriage Status",
            ["Civil marriage", "Separated", "Widow", "Married"],
        ),
        "EDUCATION": st.sidebar.selectbox(
            "Select Education Level",
            [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ],
        ),
        "GENDER": st.sidebar.selectbox("Select Gender", ["F", "M"]),
        "Car_Owner": st.sidebar.selectbox("Select Car Ownership", ["Y", "N"]),
        "Type_Income": st.sidebar.selectbox(
            "Select Type of Income",
            ["Pensioner", "Commercial associate", "Working", "State servant"],
        ),
        "Work_Phone": st.sidebar.selectbox(
            "Select Work Phone Availability", ["Not Available", "Available"]
        ),
        "Phone": st.sidebar.selectbox(
            "Select Phone Availability", ["Not Available", "Available"]
        ),
        "EMAIL_ID": st.sidebar.selectbox(
            "Select Email ID Availability", ["Not Available", "Available"]
        ),
        "Marital_status": st.sidebar.selectbox(
            "Select Marital Status", ["Single", "Married", "Divorced", "Widowed"]
        ),
        "Housing_type": st.sidebar.selectbox(
            "Select Housing Type",
            [
                "House / apartment",
                "With parents",
                "Rented apartment",
                "Municipal apartment",
                "Co-op apartment",
                "Office apartment",
            ],
        ),
        "Type_Occupation": st.sidebar.selectbox(
            "Select Type of Occupation",
            [
                "Core staff",
                "Cooking staff",
                "Laborers",
                "Sales staff",
                "Accountants",
                "High skill tech staff",
                "Managers",
                "Cleaning staff",
                "Drivers",
                "Low-skill Laborers",
                "IT staff",
                "Waiters/barmen staff",
                "Security staff",
                "Medicine staff",
                "Private service staff",
                "HR staff",
                "Secretaries",
                "Realty agents",
            ],
        ),
    }

    # Mapping for 'Not Available' and 'Available' to 0 and 1
    availability_mapping = {"Not Available": 0, "Available": 1}
    user_input["Work_Phone"] = availability_mapping[user_input["Work_Phone"]]
    user_input["Phone"] = availability_mapping[user_input["Phone"]]
    user_input["EMAIL_ID"] = availability_mapping[user_input["EMAIL_ID"]]

    # Adjust 'Employed_days' based on employment status
    if employment_status == "No":
        user_input["Employed_days"] *= -1

    # Convert to DataFrame and ensure all expected features are present
    input_df = pd.DataFrame(user_input, index=[0])
    for i in range(56):
        if str(i) not in input_df.columns:
            input_df[str(i)] = 0  # or some default value

    return input_df


# Main function to run the app
def main():
    st.title("💳 Credit Risk Assessment App")
    st.write(
        """
        ### Welcome to the Credit Risk Assessment App!
        This application uses advanced machine learning models to assess the credit risk of an applicant.
        Fill in the applicant's details in the sidebar and click 'Predict' to get the risk assessment.
    """
    )

    st.image(
        "https://cdn.pixabay.com/photo/2019/02/22/12/04/investing-4013413_1280.jpg",
        use_column_width=True,
    )

    # Get user input
    input_df = get_user_input()

    if st.button("Predict"):
        # Preprocess the input data
        input_processed = pipeline.transform(input_df)

        # Make predictions
        y_pred, y_probs = predict_with_threshold(input_processed)

        # Display the prediction and probabilities
        st.subheader("Prediction Results")
        st.write(
            f"**Credit Decision:** {'Approve Credit' if y_pred[0] == 1 else 'Reject Credit'}"
        )
        st.write(f"**Prediction Probability:** {y_probs[0]:.2f}")

        # Display detailed insights
        if y_pred[0] == 1:
            st.success(
                "The applicant is likely to be a low-risk customer. Credit approved!"
            )
        else:
            st.error(
                "The applicant is likely to be a high-risk customer. Credit not approved."
            )

        # Display additional information
        st.sidebar.subheader("Risk Factors Considered")
        st.sidebar.markdown(
            """
        - Employment Status
        - Annual Income
        - Family Size
        - Number of Children
        - Marital Status
        - Education Level
        - Housing Type
        - Occupation Type
        - Availability of Work Phone, Personal Phone, and Email ID
        """
        )


if __name__ == "__main__":
    main()
