from datasets import load_dataset
dataset = load_dataset("go_emotions", "simplified")

import pandas as pd
#converting to a dataframe
df=dataset['train'].to_pandas()

#pre-processing the dataset

import re
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
import string

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
def pre_process(text):
  x_test_wemo=[emoji_pattern.sub(r'', i) for i in text]
  
  #removing punctuations
  
  X_test=[remove_punctuations(i) for i in x_test_wemo]
  #Tokenizing using nltk
  
  X_test_tokens = [nltk.word_tokenize(row) for row in X_test]
  # Remove stop words
  
  X_test_tokens_new=[]
  stop_words = set(stopwords.words('english'))
  
  for w in X_test_tokens:
    m=[]
    for i in w:
      if i not in stop_words:
        m.append(i)
    X_test_tokens_new.append(m)
  # Stem the tokens
  stemmer = PorterStemmer()
  
  X_test_tokens_new1=[]
  
  for i in X_test_tokens_new:
    k=[]
    for j in i:
      k.append(stemmer.stem(j.lower()))
    X_test_tokens_new1.append(k)
  # Lemmatize the tokens
  lemmatizer = WordNetLemmatizer()
  
  X_test_tokens_new2=[]
  
  for i in X_test_tokens_new1:
    k=[]
    for j in i:
      k.append(lemmatizer.lemmatize(j))
    X_test_tokens_new2.append(k)  
  return X_test_tokens_new2

#test dataset
df_test=dataset['validation'].to_pandas()

#pre-processing test datasets
X_test=pre_process(df_test['text'])
X_test_updated= [' '.join(i) for i in X_test]
df_test['text']=X_test_updated

#pre-processing train dataset
X_train=pre_process(df['text'])
X_train_updated= [' '.join(i) for i in X_train]
df['text']=X_train_updated

#mapping of test

# Filter to keep only rows with list length == 1
mask = df_test['labels'].apply(lambda x: len(x) == 1)
df_test = df_test[mask]

# Convert the list of lists to int    
df_test['labels'] = df_test['labels'].apply(lambda x: int(x[0]))


mask1 = df['labels'].apply(lambda x: len(x) == 1)
df = df[mask1]

# Convert the list of lists to int    
df['labels'] = df['labels'].apply(lambda x: int(x[0]))



label_mappings = {
    0: 'admiration',
    1: 'amusement',
    2: 'anger',
    3: 'annoyance',
    4: 'approval',
    5: 'caring',
    6: 'confusion',
    7: 'curiosity',
    8: 'desire',
    9: 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'grief',
    17: 'joy',
    18: 'love',
    19: 'nervousness',
    20: 'optimism',
    21: 'pride',
    22: 'realization',
    23: 'relief',
    24: 'remorse',
    25: 'sadness',
    26: 'surprise',
    27: 'neutral',
    28: 'other',
}
# Map the labels


#mapping of train


counts_of_labels_train=df['labels'].value_counts()


#top 14 labels by slicing counts_of_labels
sliced_counts_train=counts_of_labels_train[:14]
labels_processed_train=[]
for i in df['labels']:
  if i not in sliced_counts_train.index:
    labels_processed_train.append(28) #other label
  else:
    labels_processed_train.append(i)
df['labels']=labels_processed_train




labels_processed_test=[]


for a in df_test['labels']:
  if a in sliced_counts_train:
    labels_processed_test.append(a)
  else:
    labels_processed_test.append(28)

df_test['labels']=labels_processed_test

# Map the labels
df['emotions'] = df['labels'].map(label_mappings)
df_test['emotions'] = df_test['labels'].map(label_mappings)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)

# Fit the vectorizer on the preprocessed text data
vectorizer.fit(df['text'])

# Transform the preprocessed text data into vectors
vectorized_data = vectorizer.transform(df['text'])
vectorized_data_test=vectorizer.transform(df_test['text'])

# Split the preprocessed and vectorized data into training and testing sets
X_train=vectorized_data
X_test=vectorized_data_test
y_train=df['labels']
y_test=df_test['labels']
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Train an SVM model on the training data
svm_model = SVC(kernel='linear', C=0.1)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the F1 score, precision, and recall
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print the results
#print("Confusion Matrix:")
#print(cm)
#print("F1 Score:", f1)
#print("Precision:", precision)
#print("Recall:", recall)

new_sentences=["you are love"]




from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
app = Flask(__name__)

CORS(app)
# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

# Define the Sentence model
class Sentence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sentence = db.Column(db.String(250), nullable=False)
    response = db.Column(db.String(250), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.json['sentence']
        vectorized_new_data = vectorizer.transform([data])
        new_predictions = svm_model.predict(vectorized_new_data)
        a=int(new_predictions)
        sentence = Sentence(sentence=data, response=label_mappings[a])
        # Add the sentence object to the database session
        db.session.add(sentence)
        # Commit the changes to the database
        db.session.commit()
        return jsonify({'result': label_mappings[a].upper()})
    if request.method == 'GET':
        sentences = Sentence.query.all()
        results = [{'sentence': sentence.sentence, 'response': sentence.response, 'timestamp': sentence.timestamp} for sentence in sentences]
        return jsonify({'data': results})   
if __name__ == '__main__':
    with app.app_context():
            db.create_all()
        
    app.run(host='127.0.0.1', port=8080, debug=True)