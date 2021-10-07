from flask import Flask, render_template, request
import joblib

import pandas as pd
import nltk
import gensim
import re
from gensim.utils import simple_preprocess
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# load the models from disk
Tags_model = open('Tags_model.pkl','rb')
clf = joblib.load(Tags_model)

tfidf_model = open('tfidf_vectorizer.pkl','rb')
tfidf_vectorizer = joblib.load(tfidf_model)

mlb_model = open('mlb.pkl','rb')
mlb = joblib.load(mlb_model)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        # Convert the question to df in order to run the analysis
        df_new_text = pd.DataFrame()
        df_new_text['Title'] = data

        # Remove punctuation
        df_new_text['Title_processed'] = df_new_text['Title'].map(lambda x: re.sub('[,\.!?]', ' ', x))

        def extract_NN(sent):  
            grammar = r"""
            NP:
                {<NNP>} # proper noun, singular ex. 'Dupont'
                {<NNPS>} # proper noun, plural ex. 'Americans'
                {<NN>} # noun, singular 'desk'
                {<NNS>} # noun plural 'desks'
                {<FW>} # foreign word
            """
            chunker = nltk.RegexpParser(grammar)
            ne = []
            chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
            for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
                ne.append(' '.join([child[0] for child in tree.leaves()]))
            return ne
        df_new_text['Title_extracted'] = df_new_text.Title.apply(extract_NN)

        # Tokenize and Lemmatize words
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        data_words = list(sent_to_words(df_new_text.Title_extracted))


        def Text_preprocessor(texts):
            lemmatizer = WordNetLemmatizer() # Lemmatize all words
            texts = [[lemmatizer.lemmatize(token) for token in doc] for doc in texts]
            return texts
        data_words = Text_preprocessor(data_words)

        # Remove Stopwords
        stop_words = stopwords.words('english')
        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        data_words_nostops = remove_stopwords(data_words)

        # Data shaping for the analysis
        Texts = []
        for i in data_words_nostops:
            Texts.append(' '.join(map(str, i)))
        
        # Create the data on which I will do the analysis
        mydata = pd.DataFrame()
        mydata['Text'] = Texts

        # Vectorize processed text
        X_newdata = tfidf_vectorizer.transform(mydata['Text'].tolist())

        # Predict tags
        tags = clf.predict(X_newdata)
        Tags = pd.DataFrame(tags, columns=mlb.classes_).apply(lambda row: row[row == 1].index.values.tolist(), axis=1)
        my_prediction = [i[1] for i in enumerate(Tags[0])]

    return render_template("result.html", prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

        