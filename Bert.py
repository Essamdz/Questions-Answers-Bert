import pandas as pd
import re
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import transformers as ppb
from numpy.linalg import norm
import xml.etree.ElementTree as ET

from preprocessing import read_xml,convert_xml_to_dataframe,add_label_column_to_dataframe,\
    add_list_of_answers_as_new_columns,import_Bert_model,new_column_for_Answers_BERT_features,\
        new_column_for_ReferenceAnswers_BERT_features,add_new_column_contain_highest_similarity,\
            concatenate_all_features
######################################################################################


root=read_xml()
df_all_data=convert_xml_to_dataframe(root)
df_all_data=add_label_column_to_dataframe(df_all_data)
df_all_data=add_list_of_answers_as_new_columns(df_all_data)

print("done df")

df_all_data=new_column_for_Answers_BERT_features(df_all_data)

print("done Bert")
df_all_data=new_column_for_ReferenceAnswers_BERT_features(df_all_data)
df_all_data=add_new_column_contain_highest_similarity(df_all_data)
df_all_data=concatenate_all_features(df_all_data)

#################################################################################

train_features, test_features, train_labels, test_labels = train_test_split(list(df_all_data.all_features), list(df_all_data.Label),test_size=0.2)

classifier = RandomForestClassifier()
classifier = MultinomialNB( ) 
classifier.fit(train_features, train_labels)
predictions=classifier.predict(test_features)
score=f1_score(test_labels, predictions,  average='macro')
print("Random Forest Classifier F1 Score=",score)

