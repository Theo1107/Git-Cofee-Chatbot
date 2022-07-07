import numpy as np
import string
import re
import joblib
from util import JSONParser
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Function for Processing Chat from user
def chat_processing(chat):
    # Transform Chat Into Lowercase
    chat = chat.lower()

    # Remove Punctuation From Chat
    chat = chat.translate(str.maketrans("","",string.punctuation))

    # Remove Digit From Chat
    chat = re.sub("[^A-Za-z\s']"," ", chat)

    # Remove Tab From Chat
    chat = chat.strip()

    # Stemmer Definition
    stemmer = StemmerFactory().create_stemmer()

    # Stemming Chat
    chat = stemmer.stem(chat)

    return chat

def response(chat, pipeline, jp):
    chat = chat_processing(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2:
        return "Mohon maaf nih kak, aku masih belum ngerti maksud kakak :(" , None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        return jp.get_response(pred_tag)

def start(update, context):
    update.message.reply_text("Selamat!, kakak telah terhubung dengan Gitcoff, sebuah Chatbot AI dari Git Coffee 😉")

def respons(update, context):
    chat = update.message.text
    res = response(chat, model, jp)
    update.message.reply_text(res)

# Main
# Load dataset Intents for Bot Responses
path = "dataset/intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# Initiate Bot Token from BotFather
token = 'YOUR TELEGRAM BOT API TOKEN'
updater = Updater(token, use_context=True)
dp = updater.dispatcher

# Load Chatbot Machine Learning Model
model = joblib.load("chatbot.pkl")

# Command
dp.add_handler(CommandHandler("start",start))

# Message Handler
dp.add_handler(MessageHandler(Filters.text, respons))

# Run Bot
updater.start_polling()
updater.idle()