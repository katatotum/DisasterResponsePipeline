import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Return merged dataframe from two csvs"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id",how="inner")
    return df


def clean_data(df):
    """Return transformed and cleaned dataframe."""
#One hot encode categories column
    # create a dataframe of the 36 individual category columns
    categories_expanded = df['categories'].str.split(";",expand=True)
    # select the first row of the categories dataframe
    # and use it to extract a list of new column names for categories.
    row = categories_expanded.loc[0]
    category_colnames = row.apply(lambda x: x[0:-2])
    # rename the columns of `categories`
    categories_expanded.columns = category_colnames
    
    #convert category values to just 0 or 1
    for column in categories_expanded:
        # set each value to be the last character of the string
        categories_expanded[column] = categories_expanded[column].str[-1]
        # convert column from string to numeric
        categories_expanded[column] = categories_expanded[column].astype(int)
    
    # replace categories column in df with new cateogory columns
    df.drop('categories',inplace=True,axis=1)
    df = pd.concat([df,categories_expanded],axis=1)

#Remove duplicates
    df.drop_duplicates(inplace = True)
    
#Clean id column
    #make list of ids that appear more than one time
    multi_ids = df['id'].value_counts()[df['id'].value_counts()>1].index.tolist()
    #remove any rows with ids that appeared more than once because it's ambiguous which one was labeled properly and should be used
    df = (df[~df['id'].isin(multi_ids)]).copy()
    
#Clean message column
    #removing rows where message = "#NAME?" because clearly useless
    df = (df[~(df['message']=='#NAME?')]).copy()
    
#Clean related column
    #removing rows where related = 2 because not defined
    df = (df[df['related']!=2]).copy()
    
    return df

def save_data(df, database_filename):
    """Save dataframe to sqlite database file"""
    table_name = 'messages_and_categories'
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql(table_name, engine, index=False, if_exists = "replace")


def main():
    """Take arguments from command line and run load_data(), clean_data(), save_data()"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
