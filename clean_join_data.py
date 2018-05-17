import pandas as pd
from os import path
from utils import remove_stopwords, split_left, split_right, remove_aspect, point_aspect
from sklearn.model_selection import train_test_split
import numpy as np

data_path = "./data"
electronics_data_file = "data 1_train.csv"
food_data_file = "data 2_train.csv"

laptop_test_file = "Data-1_test.csv"
food_test_file = "Data-2_test.csv"

elec_data = pd.read_csv(path.join(data_path, electronics_data_file))
food_data = pd.read_csv(path.join(data_path, food_data_file))

test_elec_data = pd.read_csv(path.join(data_path, laptop_test_file))
test_food_data = pd.read_csv(path.join(data_path, food_test_file))

test_elec_data.rename(columns={' text':'text', ' aspect_term':'aspect_term', ' term_location': 'term_location'}, inplace=True)
test_food_data.rename(columns={' text':'text', ' aspect_term':'aspect_term', ' term_location': 'term_location'}, inplace=True)

elec_data.rename(columns={' text':'text', ' aspect_term':'aspect_term', ' term_location': 'term_location',
                 ' class': 'class'}, inplace=True)
elec_data = elec_data.drop('example_id', axis=1)
food_data.rename(columns={' text':'text', ' aspect_term':'aspect_term', ' term_location': 'term_location',
                 ' class': 'class'}, inplace=True)
food_data = food_data.drop('example_id', axis=1)

dl_elec_data = elec_data[['aspect_term', 'class']].copy()
dl_food_data = food_data[['aspect_term', 'class']].copy()

joint_data = pd.concat([elec_data, food_data], ignore_index=True)
joint_data['token_text'] = joint_data[['text', 'aspect_term']].apply(point_aspect, axis=1)
dl_elec_data['token_text'] = elec_data[['text', 'aspect_term']].apply(point_aspect, axis=1)
dl_food_data['token_text'] = food_data[['text', 'aspect_term']].apply(point_aspect, axis=1)

joint_data.to_csv('./data/joint_data.csv')
dl_elec_data.to_csv('./data/dl_elec_data.csv')
dl_food_data.to_csv('./data/dl_food_data.csv')

# Split text
elec_data['left_text'] = elec_data[['text', 'aspect_term']].apply(split_left, axis=1)
elec_data['right_text'] = elec_data[['text', 'aspect_term']].apply(split_right, axis=1)
food_data['left_text'] = food_data[['text', 'aspect_term']].apply(split_left, axis=1)
food_data['right_text'] = food_data[['text', 'aspect_term']].apply(split_right, axis=1)

ml_elec_data = elec_data[['aspect_term', 'class']].copy()
ml_food_data = food_data[['aspect_term', 'class']].copy()

ml_food_data['noaspect_text'] = food_data[['text', 'aspect_term']].apply(remove_aspect, axis=1)
ml_elec_data['noaspect_text'] = elec_data[['text', 'aspect_term']].apply(remove_aspect, axis=1)
ml_elec_data['noaspect_text'] = ml_elec_data['noaspect_text'].apply(remove_stopwords)
ml_food_data['noaspect_text'] = ml_food_data['noaspect_text'].apply(remove_stopwords)

test_elec_data['noaspect_text'] = test_elec_data[['text', 'aspect_term']].apply(remove_aspect, axis=1)
test_food_data['noaspect_text'] = test_food_data[['text', 'aspect_term']].apply(remove_aspect, axis=1)
test_food_data['noaspect_text'] = test_food_data['noaspect_text'].apply(remove_stopwords)
test_elec_data['noaspect_text'] = test_elec_data['noaspect_text'].apply(remove_stopwords)

ml_elec_data['left_text'] = elec_data['left_text'].apply(remove_stopwords)
ml_elec_data['right_text'] = elec_data['right_text'].apply(remove_stopwords)
ml_food_data['left_text'] = food_data['left_text'].apply(remove_stopwords)
ml_food_data['right_text'] = food_data['right_text'].apply(remove_stopwords)
ml_elec_data['text'] = elec_data['text'].apply(remove_stopwords)
ml_food_data['text'] = food_data['text'].apply(remove_stopwords)


ml_elec_data_train, ml_elec_data_test = train_test_split(ml_elec_data, train_size=0.3,
                                                         stratify=ml_elec_data['class'].values.astype(np.float32),
                                                         random_state=0)
ml_food_data_train, ml_food_data_test = train_test_split(ml_food_data, train_size=0.3,
                                                         stratify=ml_food_data['class'].values.astype(np.float32),
                                                         random_state=0)

elec_data_train, elec_data_test = train_test_split(elec_data, test_size=0.3,
                                                   stratify=elec_data['class'].values.astype(np.float32),
                                                   random_state=0)
food_data_train, food_data_test = train_test_split(food_data, test_size=0.3,
                                                   stratify=food_data['class'].values.astype(np.float32),
                                                   random_state=0)

joint_data_train = pd.concat([elec_data_train, food_data_train], ignore_index=True)
joint_data_test = pd.concat([elec_data_test, food_data_test], ignore_index=True)
ml_joint_data_train = pd.concat([ml_elec_data_train, ml_food_data_train], ignore_index=True)
ml_joint_data_test = pd.concat([ml_elec_data_test, ml_elec_data_test], ignore_index=True)

# joint_data = pd.concat([elec_data, food_data], ignore_index=True)



joint_data_train.to_csv('./data/joint_data_train.csv')
joint_data_test.to_csv('./data/joint_data_test.csv')

elec_data.to_csv('./data/elec_data.csv')
food_data.to_csv('./data/food_data.csv')

ml_joint_data_train.to_csv('./data/ml_joint_data_train.csv')
ml_joint_data_test.to_csv('./data/ml_joint_data_test.csv')

ml_elec_data.to_csv('./data/ml_elec_data.csv')
ml_food_data.to_csv('./data/ml_food_data.csv')

test_elec_data.to_csv('./data/ml_test_elec.csv')
test_food_data.to_csv('./data/ml_test_food.csv')