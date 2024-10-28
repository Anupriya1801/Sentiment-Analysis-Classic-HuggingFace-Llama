# Sentiment-Analysis-Classic-HuggingFace-Llama

**Step 1**: Unzip the dataset.
**Step 2**: Go to https://console.groq.com/keys, create an api key and add that key in .env file
**Step 3**: Install all the necessary libraries used in imports
**Step 4**: Run the train_model1 file separately
**Step 5**: Run the train_model2 file separately
**Step 6**: After the dataset is trained for logistic regression and huggingface transformers, now run the main.py by using the command "uvicorn main:app --reload"
**Step 7**: Go to the website http://127.0.0.1:8000/docs and check for different movie reviews

This webapp will provide whether the review is "Positive" or "Negative". There will be 3 methods.
    **Method 1** - Logistic Regression
    **Method 2** - HuggingFace Transformers
    **Method 3** - Llama 3 from Groq Cloud
