from flask import Flask, request, render_template, redirect, url_for
import nltk
import pickle
from textblob import TextBlob
import urllib.request as urllib
import json
import random

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('index.html', searchgif="", input_text="")

@app.route("/", methods=["POST"])
def text():
    processed_text = get_url(request.form['text'])
    return render_template("index.html", searchgif=processed_text, input_text=request.form['text'].replace('%20', ' '))

def get_url(text):
    api = "http://api.giphy.com/v1/gifs/search?"
    api_key = "&api_key=Jj9EUV1Uh1MxAgnRgveju7TEC2JCIYn3"
    limit = "&limit=5"

    f = open('dialogue_act_model.pickle', 'rb')
    dialogue_act_model = pickle.load(f)
    f.close()

    dialogue_acts = {
            "whQuestion": create_question_query,
            "ynQuestion": create_yn_query,
            "yAnswer": create_yn_query,
            "nAnswer": create_yn_query,
            "Accept": create_yn_query,
            "Reject": create_yn_query,
            "Bye": create_bye_query,
            "Greet": create_greet_query,
            "Emphasis": create_emotion_query,
            "Emotion": create_emotion_query,
            "Other": create_other_query,
            "Statement": create_other_query,
            "Continuer": create_other_query,
            "System": create_other_query,
            "Clarify": create_other_query
            }

    query = "q="
    text = dialogue_acts[dialogue_act_model.classify(dialogue_act_features(text))](text)
    query += text.replace(" ", "+")
    results = json.loads(urllib.urlopen(api + query + api_key + limit).read())

    if len(results["data"]) > 5:
        list_index = random.randint(0, 4)
    elif len(results["data"]) > 0:
        list_index = random.randint(0, len(results["data"]) - 1)
    else:
        return "https://media.giphy.com/media/xuDHhHcCR0rew/giphy.gif"
    return results["data"][list_index]["images"]["downsized_medium"]["url"]

def create_question_query(text):
    tagged_words = tag_words(text)
    nouns = ""
    for tag in tagged_words:
        if tag[1][0] == 'N':
            nouns += tag[0] + " "
    if nouns:
        nouns = nouns[:-1]
        return nouns
    else:
        return text

def create_yn_query(text):
    ysynonyms = ["sure", "yeah", "yes", "alright", "indeed", "agreed", "certainly", "absolutely"]
    nsynonyms = ["no", "naw", "never", "nah", "not really", "nope"]
    if random.randint(0,1) == 1:
        return random.choice(ysynonyms)
    else:
        return random.choice(nsynonyms)

def create_bye_query(text):
    bye_synonyms = ["bye", "cherrio", "so long", "adios", "ciao", "hasta la vista"]
    return random.choice(bye_synonyms)

def create_greet_query(text):
    greet_synonyms = ["bonjour", "hello", "howdy", "what's up", "hey", "hi", "buenos dias"]
    return random.choice(greet_synonyms)

def create_emotion_query(text):
    return text

def create_other_query(text):
    polarity = 0
    sad_synonyms = ["sad", "heartbroken", "unhappy"]
    good_synonyms = ["good", "happy", "glad", "joyful"]
    blob = TextBlob(text)
    for sentence in blob.sentences:
        polarity += sentence.sentiment.polarity
    if polarity < 0:
        return random.choice(sad_synonyms)
    if polarity > 0:
        return random.choice(good_synonyms)
    else:
        return text

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

def tag_words(text):
    words = nltk.tokenize.word_tokenize(text)
    return nltk.pos_tag(words)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
