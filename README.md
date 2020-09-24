# Disaster Response Pipeline Project

## Installation
Python 3.x   
Libraries: numpy, pandas, sqlalchemy, nltk, re, sklearn, pickle, json, plotly, flask

## Project Motivation
The analysis uses a data set containing real tweets and texts that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.    

The project also includes a web app where a user can input a new message and get classification results in all of the possible categories. The web app will also display visualizations of the data. This provides a way for emergency workers to see how the model is working for individual messages, but in practice, this would not be efficient during an actual disaster and instead the model would be run on many messages simultanouesly and perhaps even automatically forward messages to the proper agency based on the category.

## File Descriptions
<b>data/disaster_messages.csv</b>:  original dataset with columns for message id, message, original message, and genre.   
<b>data/disaster_categories.csv</b>: original dataset with columns for message id and message categories (hand labled).   
<b>data/process_data.py</b>: ETL pipeline that takes data from csv's (above), cleans it, and stores it in a sqlite database.   
<b>data/DisasterResponse.db</b>: The database created by process_data.py (above).   
   
<b>models/train_classifier.py</b>: ML pipeline that uses DisasterResponse.db (above) as input, then builds, trains, evaluates and saves a multioutput classifier supervised learning model to predict categories given a message.   
   
<b>app/templates</b>: webpage templates for run.py   
<b>app/run.py</b>: Creates flask web app that displays basic graphs of the original data and uses the ML model (above) to classify messages typed in at the webpage.

## How to Make it All Work
1. Install any necessary software from <b>Installation</b> section (above).

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

5. Follow the prompt on the web app to type in or paste in a disaster-related message to see what categories the model classifies it as.

## Acknowledgements
This project is part of Udacity's Data Scientist Nanodegree and as such uses some starter code and verbiage provided by Udacity. Additionally, as already mentioned, the original datasets are provided by [Figure Eight (Appen)](https://appen.com/).
