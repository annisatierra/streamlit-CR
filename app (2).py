import streamlit as st
import joblib
import pandas as pd


# Define the feature engineering function
def feature_engineering(df):
    df["Years_Employed"] = df["Employed_days"] // 365
    df["Birthday_count"] = df["Birthday_count"].abs()
    df["Income_per_Family_Member"] = df["Annual_income"] / df["Family_Members"]
    df["Children_per_Family_Member"] = df["CHILDREN"] / df["Family_Members"]
    df["Is_Employed"] = df["Employed_days"] > 0
    return df


# Load the pipeline
pipeline = joblib.load("preprocessing_pipeline.pkl")


# Function to get user input
def get_user_input():
    user_input = {
        "Employed_days": st.number_input("Enter Employed Days", min_value=0, step=1),
        "Birthday_count": st.number_input("Enter Birthday Count", min_value=0, step=1),
        "Annual_income": st.number_input(
            "Enter Annual Income", min_value=0.0, step=0.01
        ),
        "Family_Members": st.number_input(
            "Enter Number of Family Members", min_value=1, step=1
        ),
        "CHILDREN": st.number_input("Enter Number of Children", min_value=0, step=1),
        "Civil marriage": st.selectbox(
            "Select Civil Marriage Status", ["Civil marriage", "Separated", "Widow"]
        ),
        "EDUCATION": st.selectbox(
            "Select Education Level",
            [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ],
        ),
        "GENDER": st.selectbox("Select Gender", ["F", "M"]),
        "Car_Owner": st.selectbox("Select Car Ownership", ["Y", "N"]),
        "Type_Income": st.selectbox(
            "Select Type of Income",
            ["Pensioner", "Commercial associate", "Working", "State servant"],
        ),
        "Work_Phone": st.selectbox("Select Work Phone Availability", ["0", "1"]),
        "Phone": st.selectbox("Select Phone Availability", ["0", "1"]),
        "EMAIL_ID": st.selectbox("Select Email ID Availability", ["0", "1"]),
        "Marital_status": st.selectbox(
            "Select Marital Status", ["Single", "Married", "Divorced", "Widowed"]
        ),
        "Housing_type": st.selectbox(
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
        "Type_Occupation": st.selectbox(
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

    # Convert to DataFrame and ensure all expected features are present
    input_df = pd.DataFrame(user_input, index=[0])
    for i in range(56):
        if str(i) not in input_df.columns:
            input_df[str(i)] = 0  # or some default value

    return input_df


# Main function to run the app
def main():
    st.title("Prediction App")

    # Get user input
    input_df = get_user_input()

    # Preprocess the input data
    input_processed = pipeline.transform(input_df)

    # Load the model
    model = joblib.load("CatBoost_final_2.pkl")

    # Make predictions
    prediction = model.predict(input_processed)
    prediction_prob = model.predict_proba(input_processed)

    # Display the prediction and probabilities
    st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
    st.write(f"Prediction Probability: {prediction_prob[0]}")


if __name__ == "__main__":
    main()
