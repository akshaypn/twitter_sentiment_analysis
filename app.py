import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# Load the saved model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('model/sentiment_model')
# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') 
# Streamlit app
st.title("Sentiment Analysis App")

# Input text box for user to enter a tweet
user_input = st.text_area("Enter a tweet:", "")

if st.button("Predict"):
    if user_input:
        # Tokenize and preprocess the input text
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted sentiment
        predicted_class = torch.argmax(outputs.logits).item()
        probs = F.softmax(outputs.logits, dim=-1) 
        probability =  max(probs[0])
        # Display the prediction
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        st.write(f"Predicted sentiment: {sentiment}") 
        st.write(f"Sentiment Probablity Percentage : {probability*100:.2f}%")
