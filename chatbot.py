# -*- coding: utf-8 -*-

import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the text file
with open('E:\\AI.0.txt', 'r', errors='ignore') as f:
    raw = f.read().lower()

# Tokenizing the text into sentences
sent_tokens = nltk.sent_tokenize(raw)  # List of sentences

# Preprocessing: Lemmatization to reduce words to their base form
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Dictionary to remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# List of greeting inputs and corresponding responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi!", "Hey!", "Hello!", "Greetings!", "Hi there!", "Glad to chat with you!"]

# Function to check if a greeting is present in the user's input
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Function to generate responses based on similarity using TF-IDF
def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)  # Adding the user's input to the sentence list
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # Compute cosine similarity between the user's input and all the sentences
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]  # Get the index of the most similar sentence
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]  # Second most similar sentence score
    
    if req_tfidf == 0:
        chatbot_response = "I am sorry! I don't understand you."
    else:
        # Instead of returning the question, extract the corresponding answer
        if idx % 2 == 0:  # Assuming questions are at even indices
            chatbot_response = sent_tokens[idx + 1]  # Return the answer that follows the question
        else:
            chatbot_response = sent_tokens[idx - 1]  # Return the previous answer

    sent_tokens.remove(user_response)  # Remove user input after processing
    return chatbot_response


# Main loop to start the chatbot interaction
if __name__ == "__main__":
    print("Hello! My name is Bibo. I'm an AI chatbot here to answer your questions. Type 'bye' to exit.")
    flag = True
    while flag:
        user_response = input().lower()
        
        if user_response != 'bye':
            if user_response in ('thanks', 'thank you'):
                flag = False
                print("Bibo: You're welcome!")
            else:
                # Check for greetings
                if greeting(user_response) is not None:
                    print("Bibo: " + greeting(user_response))
                else:
                    # Generate and print the chatbot's response
                    print("Bibo: " + response(user_response))
        else:
            flag = False
            print("Bibo: Bye! Have a great day!")
