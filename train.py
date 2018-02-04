import nltk
import pickle

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

print("Training emotions")
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
dialogue_act_model = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(dialogue_act_model, test_set))

f = open('dialogue_act_model.pickle', 'wb')
pickle.dump(dialogue_act_model, f)
f.close()


