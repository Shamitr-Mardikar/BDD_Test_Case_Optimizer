import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the DataFrame
file_path = 'datasets/testcase_dataset.csv'
df = pd.read_csv(file_path)

# Vectorize the Text Data
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['Test Case Description'])
y = df['Optimization Level']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the Model
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the model and vectorizer
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# Streamlit App
st.title('BDD Test Case Optimizer')

model, vectorizer = load_model()

# Input: Test Case Description
test_case_description = st.text_area('Enter BDD Test Case Description:', '')

if st.button('Predict Optimization Level'):
    if test_case_description:
        # Vectorize the input
        new_x = vectorizer.transform([test_case_description])
        
        # Make prediction
        predictions = model.predict(new_x)
        
        # Display prediction
        if predictions == 'Properly Optimized':
            st.write("PROPERLY OPTIMIZED")
            st.write("Test cases are Properly Optimized with a good level of ambiguity and granularity score.")
        elif predictions == 'Moderately Optimized':
            st.write("MODERATELY OPTIMIZED")
            st.write("Ensure clarity and precision by refining language to avoid ambiguity.\nUse specific roles like 'a logged-in user' instead of generic terms.\nEnhance reusability by breaking down complex steps into smaller steps.\nExtract common actions into separate, reusable 'Given' steps.")
        elif predictions == 'Poorly Optimized':
            st.write("POORLY OPTIMIZED")
            st.write("Improve step granularity by splitting complex steps into smaller ones.\nEach step should represent a single action or assertion.\nIncrease maintainability by ensuring steps are relevant to the scenario.\nRemove unnecessary or redundant steps, focusing on core actions.")
        else:
            st.write("Invalid prediction")
    else:
        st.write('Please enter a BDD test case description.')

# Example of how to display the DataFrame if needed
if st.checkbox('Show DataFrame'):
    st.write(df)
