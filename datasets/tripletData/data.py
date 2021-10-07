import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# file_path = './first_stage/train.json'
file_path = './exercise_contest/data_train.json'

# 读取json文件
lines=[]
with open(file_path, encoding="utf-8") as raw_input:
        for line in raw_input:
            lines.append(line.strip())
lines = list(set(lines))

# 读取json中的fact accusation relevant_articles 
fact = []
accusation = []
relevant_articles = []
article_num = []
for line in lines:
    item = json.loads(line,encoding='utf-8')
    fact.append(item['fact'])
    accusation.append(item['meta']['accusation'])
    relevant_articles.append(list(set(item['meta']['relevant_articles']))) #法条中会有重复的，去重
    article_num.append(len(set(item['meta']['relevant_articles'])))  #计算法条的数量，后面会用到

# 利用pandas进行数据筛选，选择单罪名的数据  选择fact在150~600字的数据
legal_data = pd.DataFrame(fact,index=list(range(len(fact))),columns=['fact'])
legal_data['accusation']=accusation
legal_data['relevant_articles']=relevant_articles
legal_data['article_num'] = article_num

drop_index=[]
single_accusation=[]
for i in range(legal_data.shape[0]):
    accusations = legal_data.loc[i,['accusation']][0]
    fact_len = len(legal_data.loc[i,'fact'])
    if(len(accusations)==1 and (fact_len<=600 and fact_len>=150)):
        single_accusation.append(accusations[0])
    else:
        drop_index.append(i)
slegal_data=legal_data.drop(drop_index)  #去掉非单罪名的数据  shape (118321,3)
slegal_data['accusation'] = single_accusation  #去掉单罪名数据中的 []
slegal_data=slegal_data.drop_duplicates(subset='fact', keep='first',inplace=False) #去掉重复的fact

# 罪名词典  建立关于同一罪名的数据下标字典  统计法条数量
accusation_dict = list(set(single_accusation))
accusation_count = {}
for accusation in accusation_dict:
    accusation_count[accusation] = len(slegal_data[slegal_data['accusation']==accusation].index)

accusation_count = sorted(accusation_count.items(),key = lambda kv:(kv[1],kv[0]))
accusation_count.reverse()

""" 
    使用slegal_data，accusation_dict，accsation_dict_index，参考accusation_num进行数据构造
    目标：构造50000条(A,B,C)数据
    1. 第一大类  2
        1）AB 罪名一致
        2）AC 罪名不一
    
    2. 第二大类  3
        1）AB 罪名一致，发条一致  单法条  单法条
        2）AC 罪名一致，发条不一  法条数量
 """
 
# 第一大类数据的构造
output_file = './tripletData/category1_data.json'
with open(output_file,mode='w',encoding='utf-8') as output:
    for i in range(len(accusation_dict)):
        data_index = list(slegal_data[slegal_data['accusation']==accusation_dict[i]].index)
        data_num = len(data_index)
        
        if(data_num<3):
            continue
        
        temp_slegal_data=slegal_data.drop(data_index)  #把属于该accusation种的数据去掉
        c_data = temp_slegal_data.sample(n=data_num)   #从剩余数据集中抽取data_num数量的数据
        c_data = c_data.reset_index()


        for j in range(data_num):
            a = slegal_data.loc[data_index[j],'fact']
            b = slegal_data.loc[data_index[(j+1)%data_num],'fact']
            c = c_data.loc[j,'fact']

            line = json.dumps({'accusation':accusation_dict[i],'A':a,'B':b,'C':c},ensure_ascii=False).strip()
            
            output.write(line)
            output.write('\n')

 # 第二大类数据的构造
output_file = './tripletData/data2.json'
with open(output_file,mode='w',encoding='utf-8') as output:
    for accusation in accusation_dict:
        single_article_index = list(slegal_data[(slegal_data['article_num']==1)&(slegal_data['accusation']==accusation)].index)
        multi_articles_index = list(slegal_data[(slegal_data['article_num']>1)&(slegal_data['accusation']==accusation)].index)
        
        single_num = len(single_article_index)
        multi_num = len(multi_articles_index)

        if(multi_num==0 or single_num<3):
            continue

        for i in range(single_num):
            a = slegal_data.loc[single_article_index[i],'fact']
            b = slegal_data.loc[single_article_index[(i+1)%single_num],'fact']
            c = slegal_data.loc[multi_articles_index[i%multi_num],'fact']

            line = json.dumps({'accusation':accusation,'A':a,'B':b,'C':c},ensure_ascii=False).strip()
            
            output.write(line)
            output.write('\n')




