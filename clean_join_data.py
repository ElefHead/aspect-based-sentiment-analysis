import pandas as pd
from os import path
from utils import remove_stopwords

data_path = "./data"
electronics_data_file = "data 1_train.csv"
food_data_file = "data 2_train.csv"

elec_data = pd.read_csv(path.join(data_path, electronics_data_file))
food_data = pd.read_csv(path.join(data_path, food_data_file))

elec_data.rename(columns={' text':'text', ' aspect_term':'aspect_term', ' term_location': 'term_location',
                 ' class': 'class'}, inplace=True)
elec_data = elec_data.drop('example_id', axis=1)
food_data.rename(columns={' text':'text', ' aspect_term':'aspect_term', ' term_location': 'term_location',
                 ' class': 'class'}, inplace=True)
food_data = food_data.drop('example_id', axis=1)

elec_data['filtered_text'] = elec_data['text'].apply(remove_stopwords)
food_data['filtered_text'] = food_data['text'].apply(remove_stopwords)

joint_data = pd.concat([elec_data, food_data], ignore_index=True)

joint_data.to_csv('./data/joint_data.csv')
elec_data.to_csv('./data/elec_data_clean.csv')
food_data.to_csv('./data/food_data_clean.csv')