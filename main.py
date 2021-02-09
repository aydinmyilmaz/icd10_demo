import streamlit as st
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


st.title ('ICD10 Recommendation Engine')

#st.write ('''
## Sub-Title 1
#''')

# import train and test data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# merge train and test data into a new df
merged_df = train_df.append(test_df)
merged_df = merged_df.dropna().reset_index().drop('index',axis=1)

# generate new columns to for downstream tasks; icd10_codes and icd10_list
merged_df['icd10_codes'] = merged_df.icd.apply(lambda x: ','.join(x.split(',')))
merged_df['icd10_list'] = merged_df.icd.apply(lambda x: x.split(','))
merged_df = merged_df.drop('icd',axis=1)

# get tf-idf vectors
tfidfvectoriser = TfidfVectorizer()
tfidfvectoriser.fit(merged_df['icd10_codes'])
tfidf_vectors = tfidfvectoriser.transform(merged_df['icd10_codes'])

# custom function to get icd10 recommendation
def most_similar(idx, number_of_similar_ids, recommended_icd10_num=5):
  
    # idx : index number of target case, 
    
    print('target case given icd10 codes :', merged_df.iloc[idx,2],'\n')
    
    # get cosine_similarities for given case 
    cosine_similarities = linear_kernel(tfidf_vectors[idx], tfidf_vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[0:]]

    # generate df with calculated similarity scores
    df_similarity = pd.DataFrame(document_scores, columns=['score'])
    update_df = df_similarity.drop([idx])
      
    
    # select #n most similar case indexes from similarity df 
    most_similar_df = update_df.sort_values(by='score',ascending=False).head(number_of_similar_ids)
    similar_indexes = most_similar_df.index.values
    
    # get icd10 codes of #n most similar cases from merged df
    icd_10_codes = merged_df['icd10_list'][merged_df.index.isin(similar_indexes)].values

    
    # get set of icd10 codes for most similar #n cases
    nearest_icd_code_list = {item for items in icd_10_codes for item in items}
    
    # get common icd10 codes between target case and most similar n case
    common_list = [x for x in merged_df.iloc[idx,2] if x in nearest_icd_code_list ]
    
    #print('common_codes_list:', common_list,'\n')
    
    # count the number of icd10 codes for most similar #n cases
    from collections import Counter
    c = Counter([item for items in icd_10_codes for item in items])
    count_common_icd10 = c.most_common()
    
    #print( 'count_codes:',count_common_icd10,'\n')
    
    # select top m = recommended_icd10_num from counted_common_icd10 list gathered from #n most similar cases
    recommendation_list = []
    for code in count_common_icd10:
        # if most common codes are already in target case icd10 list include them in recomendation list
        if code[0] in merged_df.iloc[idx,2]:
            recommendation_list.append(code)      
    
    #print('recommendation_list:',recommendation_list)
    # recommend additional icd10 codes if target icd10 code list has less than 5 codes
    
    if len(recommendation_list) < recommended_icd10_num :
        
        l = len(recommendation_list)
        
        recommendation_list.extend([x for x in count_common_icd10 if x not in recommendation_list][0:recommended_icd10_num-l])
        
        #print('recommended_codes:', recommendation_list,'\n') 
     
    
    return recommendation_list[0:recommended_icd10_num]#most_similar_df, icd_10_codes

#get query parameters from user

# index of query case

# similar cases to evaluate given case similarity
sim_case_num = st.sidebar.slider('Number of Similar Cases to Consider',10,500, value=10, step=10)

# number of recommending icd10 codes
rcmd_icd_num = st.sidebar.slider('Number of ICD10 Codes to Recommend',5,50, value=5,step=5)

# index number for query case
df_len = merged_df.shape[0]

case_input_type = st.sidebar.selectbox('Case Selection Type',('Unique_ID', 'Index'))
#if st.sidebar.button('Enter Case Unique ID'):
if case_input_type == 'Unique_ID':
    case_id = st.sidebar.text_input('Case Unique ID', value='b7d38462-fe4b-41fe-96d6-f7988a4dc6fd') 
    try:
        query_index_number = merged_df[merged_df.case_id==case_id].index.values[0]
        #st.write('All Parameters are given')
        st.write('Please make selection for Recommendation')
    except IndexError:
        st.error('Please enter a valid input')
else :
    query_index_number = int(st.sidebar.number_input(f'Case Index Number 0-{df_len}', value=50000))
    #st.write('All Parameters are given')
    #st.write('Please make selection for Recommendation')
    if  df_len<query_index_number<0 :
        st.write("Please enter a valid input")
    

#query_index_number = st.sidebar.slider('Query Index Number',0,df_len)

# Add Action Button for ICD10 Recommendations
if st.sidebar.button(f'Give Me {rcmd_icd_num} ICD10 Recommendation'):

    st.write (''' ## Recommended ICD10 codes for given parameters : ''')

    st.write (f'Number of Similar Cases to Consider :  {sim_case_num}')
    st.write (f'Query Index Number : {query_index_number}')
    st.write (f'Number of ICD10 Codes to Recommend :  {rcmd_icd_num}')

    recommendation_list = [x[0] for x in most_similar(query_index_number,sim_case_num,rcmd_icd_num)]

    # print results on screen
    st.write(recommendation_list)

# Add Action button for ICD10 Similarities
if st.sidebar.button(f'Give Me ICD10 Definitions and Similarity Scores   '):
    

    st.write (''' ## Similarity Scores and Definitions for Recommended ICD10 Codes : ''')

    # Import gensim Word2Vec for word embeddings model
    from gensim.models import Word2Vec

    #create word corpus
    sent_corpus =merged_df.icd10_list.tolist() 

    #train model
    model = Word2Vec(sent_corpus, size=100, window=4, min_count=2, workers=4)

    #save and load model 
    model.save("model_icd10.model")
    model_1 = Word2Vec.load("model_icd10.model")

    # Get similarity scores for ICD10 codes
    icd_similarities_dict = dict()
    recommendation_list = [x[0] for x in most_similar(query_index_number,sim_case_num,rcmd_icd_num)]
    for item in recommendation_list:
        icd_similarities_dict[item] = []

        df_ = pd.DataFrame(model_1.wv.most_similar(item, topn=5500), columns=['icd10','sim_score'])
        sub_list = recommendation_list[:]
        sub_list.remove(item)
        for code in sub_list:
            icd_similarities_dict[item].append(df_[df_.icd10==code].reset_index().set_index(['icd10']).T.to_dict())
        
    #st.write(icd_similarities_dict)
    
    # Check icd10 definitions from oficial source


    df = pd.read_csv('data/icd10gm_def_1.txt', delimiter=';', header=None)
    df.columns= ['code','definiton']
    df['code'] = df.code.astype(str)

    # convert given icd10 codes to right format to compare in icd10gm_2021
    def code(code):
        converted_code = str(code)[0:3]+'.'+str(code)[3:]
        if converted_code[-1] == '.':
            converted_code = converted_code+'0'
        return converted_code

    code_list = [code(x) for x in recommendation_list]
    fixed_codes = [x[0:3] for x in code_list if x[-2:]=='.0']
    code_list.extend(fixed_codes)
    code_list_2 = list(set(code_list))

    recom_df = df[df.code.isin(code_list_2)]

    code_list_3 = []
    for idx in recom_df.index:
        if (df.loc[[idx],'code'].item()[0] == df.loc[[idx+1],'code'].item()[0]) and  (eval(df.loc[[idx], 'code'].item()[1:]) != eval(df.loc[[idx+1], 'code'].item()[1:])) :
            code_list_3.append(df.loc[[idx], 'code'].item())

    recom_df = recom_df[recom_df .code.isin(code_list_3)]

    icd_similarities_dict = dict()
    for item in recommendation_list:
        icd_similarities_dict[item] = []
        df_ = pd.DataFrame(model_1.wv.most_similar(item, topn=5500), columns=['icd10','sim_score'])
        sub_list = recommendation_list[:]
        sub_list.remove(item)
        for code in sub_list:
            icd_similarities_dict[item].append(df_[df_.icd10==code].reset_index().set_index(['icd10']).T.to_dict())
            
    sim_list = []
    for code in recom_df.code.values:
        for k,v in icd_similarities_dict.items():
            if (str(code).replace('.','')) == k :
                sim_list.append(v)
                break

    recom_df['icd10_similarities'] = sim_list

    # generate 2 new columns to average values from similarity evaluation
    list_similarity_index = []
    list_average_score = []

    for data in recom_df['icd10_similarities'].values:
        average_indexes = []
        average_score = 0
        #print('len data:',len(data))
        for item in data:
            average_indexes.append(int((list(item.values())[0])['index']))
            average_score += ((list(item.values())[0])['sim_score'])
        
        list_average_score.append(average_score/len(data))
        list_similarity_index.append(average_indexes)
            
    recom_df['list_similarity_indexes'] = list_similarity_index
    recom_df['average_similarity_score'] = list_average_score

    recom_df['icd10_similarities'] = sim_list

    st.write(recom_df)







