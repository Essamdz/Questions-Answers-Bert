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

def read_xml():
    xml_data = open(r"grade_data.xml", 'r').read()  # Read file
    root = ET.XML(xml_data)  # Parse XML
    return root

def convert_xml_to_dataframe(root):
    all_data=[]
    Columns_name=[]
    for i, child in enumerate(root):
        data_in_one_row=[]
        for subchild in child:
            if subchild.tag=="MetaInfo":
                data_in_one_row.append(subchild.get('TaskID'))
            elif subchild.tag=="Annotation":
                data_in_one_row.append(subchild.get('Label'))
            else:
                data_in_one_row.append(subchild.text)
            
            if i==0:
                Columns_name.append(subchild.tag)
        all_data.append(data_in_one_row)
        
    df_all_data = pd.DataFrame(all_data)  # Write in DF and transpose it
    df_all_data .columns = Columns_name  # Update column names
    
    return df_all_data 

def extract_labels_from_Annotation_column(annotation:str):
    annotation_to_numbers=re.findall("\d",annotation)
    return annotation_to_numbers.index("1")+1


def add_label_column_to_dataframe(df_all_data):
    df_all_data['Label']=df_all_data.Annotation.apply(extract_labels_from_Annotation_column)
    return df_all_data

def convert_ReferenceAnswers_as_list_of_answers(ReferenceAnswers:str):
    list_of_answers=re.sub("\d:","",ReferenceAnswers)[1:-1].split("\n")
    return list_of_answers

def add_list_of_answers_as_new_columns(df_all_data):
    df_all_data['list_of_answers']=df_all_data.ReferenceAnswers.apply(convert_ReferenceAnswers_as_list_of_answers)
    return df_all_data


def import_Bert_model():
    Bert_model_class, Bert_tokenizer_class, Bert_pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    Bert_tokenizer = Bert_tokenizer_class.from_pretrained(Bert_pretrained_weights)
    Bert_model = Bert_model_class.from_pretrained(Bert_pretrained_weights)
    return Bert_tokenizer,Bert_model
        
def text_to_BERT_Features(text:str):

    tk=Bert_tokenizer.encode(text, add_special_tokens=True)
    
    max_len = 100   #this number depends on the length of the answers
    pad=np.array([tk + [0]*(max_len-len(tk))])
    attention_mask = np.where(pad != 0, 1, 0)
    
    input_ids = torch.tensor(pad)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = Bert_model(input_ids.to(torch.long), attention_mask=attention_mask)
    features = last_hidden_states[0][:,0,:].numpy()
    return features[0]

def list_of_text_to_BERT_Features(list_of_text:list):
    feature_list=[]
    for text in list_of_text:
        feature_list.append(text_to_BERT_Features(text))
    return feature_list

def new_column_for_Answers_BERT_features(df_all_data):
    df_all_data['Answers_Bert_Features']=df_all_data.Answer.apply(text_to_BERT_Features)
    return df_all_data

def new_column_for_ReferenceAnswers_BERT_features(df_all_data):  
    df_all_data['list_of_ReferenceAnswers_Bert_Features']=df_all_data.list_of_answers.apply(list_of_text_to_BERT_Features)
    return df_all_data   


def cosine_similarity(array1,array2):
    return np.dot(array1,array2)/(norm(array1)*norm(array2))

def find_the_highest_similarity_answer(list_of_array,array1):
    highest_similarity=0
    for array2 in list_of_array:
        new_similarity=cosine_similarity(array1,array2)
        if new_similarity>highest_similarity:
            highest_similarity=new_similarity
            highest_array=array2
    return highest_array
        
def add_new_column_contain_highest_similarity(df_all_data):
        list_of_highest=[]
        list_answers=list(df_all_data.list_of_ReferenceAnswers_Bert_Features)
        for i, array1 in enumerate(df_all_data.Answers_Bert_Features):
            the_highest=find_the_highest_similarity_answer(list_answers[i], array1)
            list_of_highest.append(the_highest)
        df_all_data['highest_answers']=list(list_of_highest)
        return df_all_data 
    
def concatenate_all_features(df_all_data):
    list_of_concatenated_arrays=[]
    for i in range(len(df_all_data)):
        concatenated_arrays=np.concatenate((df_all_data.Answers_Bert_Features[i],df_all_data.highest_answers[i]), axis = 0)
        list_of_concatenated_arrays.append(concatenated_arrays)
    df_all_data['all_features']=list_of_concatenated_arrays
    return df_all_data


Bert_tokenizer,Bert_model=import_Bert_model()