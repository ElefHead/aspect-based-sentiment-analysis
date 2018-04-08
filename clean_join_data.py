import pandas as pd
from os import path
from utils import remove_stopwords, split_left, split_right

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

# Split text
elec_data['left_text'] = elec_data[['text', 'aspect_term']].apply(split_left, axis=1)
elec_data['right_text'] = elec_data[['text', 'aspect_term']].apply(split_right, axis=1)
food_data['left_text'] = food_data[['text', 'aspect_term']].apply(split_left, axis=1)
food_data['right_text'] = food_data[['text', 'aspect_term']].apply(split_right, axis=1)

ml_elec_data = elec_data[['aspect_term', 'class']].copy()
ml_food_data = food_data[['aspect_term', 'class']].copy()

ml_elec_data['left_text'] = elec_data['left_text'].apply(remove_stopwords)
ml_elec_data['right_text'] = elec_data['right_text'].apply(remove_stopwords)
ml_food_data['left_text'] = food_data['left_text'].apply(remove_stopwords)
ml_food_data['right_text'] = food_data['right_text'].apply(remove_stopwords)
ml_elec_data['text'] = elec_data['text'].apply(remove_stopwords)
ml_food_data['text'] = food_data['text'].apply(remove_stopwords)

joint_data = pd.concat([elec_data, food_data], ignore_index=True)
ml_joint_data = pd.concat([ml_elec_data, ml_food_data], ignore_index=True)

joint_data.to_csv('./data/joint_data.csv')
elec_data.to_csv('./data/elec_data.csv')
food_data.to_csv('./data/food_data.csv')

ml_joint_data.to_csv('./data/ml_joint_data.csv')
ml_elec_data.to_csv('./data/ml_elec_data.csv')
ml_food_data.to_csv('./data/ml_food_data.csv')