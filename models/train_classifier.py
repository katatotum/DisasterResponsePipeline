
# import libraries
import sys
import numpy as np
import pandas as pd
import nltk
nltk.download('popular')
import sqlalchemy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    table_name = "messages_and_categories"
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    categories = df.columns.tolist()[4:]
    Y = df[categories].values
    return X, Y, categories


def tokenize(text):
    #normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize text
    words = word_tokenize(text)
    #remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Reduce words to their stems
    #stemmed = [PorterStemmer().stem(w) for w in words]
    return lemmed


def build_model():
    clf = MultiOutputClassifier(RandomForestClassifier())
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', clf)
                ])
    parameters = {
                'clf__estimator__n_estimators': [5, 10, 20]
                }

    cv = GridSearchCV(estimator = pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(Y_pred)
    y_test_df = pd.DataFrame(Y_test)
    for i in range(0,36):
        print(category_names[i] + "\n")
        print(classification_report(y_test_df[i],y_pred_df[i]))
    return Y_pred,Y_test


def save_model(model, model_filepath):
    # Save the trained model as a pickle string. 
    pickle.dump(model, open(model_filepath, "wb")) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()