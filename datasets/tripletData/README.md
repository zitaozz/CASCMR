## 一、 数据简介

数据集从CAIL2018-small中生成，数据形式为(A,B,C)，用于SCM任务。数据集共分为两个文件（category1.json, category2.json），分别包含92322、69741条数据，共计162,063条数据。数据的生成方式以及两个类别间的差别下面进行讲述

## 二、数据生成

CAIL2018-small为处理前包含10个属性，共151254条数据，我们只取其中的fact、accusation、relevant_articles三个属性，之后对数据进行筛选：1）去掉多罪名数据 2）选择fact字数在150~600的数据 3）对数据进行去重。

筛选后，数据包含190个罪名，共计9,2325条数据。完成上面的操作后，我们自定义了两类数据，具体的规则和生成方法如下：

### 1. 第一大类  category1.json  92322
   #### ABC的匹配原则
    1）AB 罪名一致
    2）AC 罪名不一

生成第一类数据时，我们将同罪名数据进行聚合，之后同罪名全部数据same_accusation_data作为A，之后把数据same_accusation_data错一位后作为B，e.g. (A,B) = (same_accusation_data[i],same_accusation_data[(i+1)%len(same_accusation_data)])。从全部数据all_data中去掉same_accusation_data，之后从(all_data-same_accusation_data)中抽取 len(same_accusation_data) 数量的数据作为C，从而生成了(A,B,C)的数据对

### 2. 第二大类  category2.json  69741
   #### ABC的匹配原则 
    1）AB 罪名一致，发条一致  单法条  
    2）AC 罪名一致，发条不一  多法条

生成第二类数据时，我们同样对数据进行聚合，之后对同罪名数据same_accusation_data进行拆分，拆分为单法条数据single_article_data 和多法条数据multi_articles_data。把全部single_article_data作为A，之后把数据single_article_data错一位后作为数据B, e.g. (A,B) = (single_article_data[i],single_article_data[(i+1)%len(single_article_data)])。之后把multi_articles_data作为数据C，从而生成了(A,B,C)的数据对
