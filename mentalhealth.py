import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import streamlit as st 
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

data = pd.read_csv('Mental_Health_FAQ.csv', sep = ',')


data.drop('Question_ID', axis = 1, inplace = True)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['Questions'].apply(preprocess_text)
data['Questions'].apply(preprocess_text)

xtrain = data['tokenized Questions'].to_list()

#Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

#------------------STREAMLIT IMPLIMENTATION-----------------------------------------

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>MENTAL HEALTH CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Eme Ita</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

user_hist = []
reply_hist = []


robot_image, space1, space2,  chats = st.columns(4)  #----to create extra spaces to allow more columns, the space1 & 2 cn be renamed to choice name,this is done to create space between the robot and the chat, in this case 2 spaces wer created bwn the robot and chat
with robot_image:
    robot_image.image('pngwing.com (9).png', width = 500)


with chats:
    user_message = chats.text_input('Hello there you can ask your question: ')

    def responder(text):
        user_input_processed = preprocess_text(text)
        vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
        similarity_score = cosine_similarity(vectorized_user_input, corpus)
        argument_maximum = similarity_score.argmax()
        return (data['Answers'].iloc[argument_maximum])

    bot_greetings = ['hello user, I am a creation Eme Ita......Ask your question',
                'hy there, how can i assist you today?',
                'what can I assist you with today?',
                'hullo, whats popping at your end?',
                'whatsup my guy']

    bot_farewell = ['thanks for your time',
                'bye, chat with you some other time',
                'thanks for your time, enjoy the rest of your day',
                'we look forward to your next patronage',
                'Take good care of yourself']

    human_greetings = ['hi', 'hello there', 'hey', 'hello']

    human_exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye_bye', 'close']

    import random
    random_greeting = random.choice(bot_greetings)
    random_farewell = random.choice(bot_farewell)

    if user_message.lower() in human_exits:
        chats.write(f"\nChatbot: {random_farewell}!")
        user_hist.append(user_message)
        reply_hist.append(random_farewell)


    elif user_message.lower() in ['hi', 'hello', 'hey', 'hi there']:
        chats.write(f"\nChatbot: {random_greeting}!")
        user_hist.append(user_message)
        reply_hist.append(random_greeting)

    elif user_message == '':
            chats.write('')

    else:   
        response = responder(user_message)
        chats.write(f"\nChatbot:{response}")
        user_hist.append(user_message)
        reply_hist.append(response)

import csv
#Save history of user texts
with open('history.txt', 'a') as file:
    for item in user_hist:
        file.write(str(item) + '\n')

#Save history of bot reply
with open('reply.txt', 'a') as file:
    for item in reply_hist:
        file.write(str(item) + '\n')


#import the file to display it in front end
with open('history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('reply.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({'User Input': data1, 'Bot_Reply': data2})

#history = pd.series(data)
st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)
#st.sidebar.write(data2)


