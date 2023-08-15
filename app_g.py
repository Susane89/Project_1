#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 


#for hypotheses
from statsmodels import robust
from pathlib import Path

from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

import statsmodels.formula.api as smf
from statsmodels.stats import power

from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


#функция для построения гистограммы
def gist(col1, col2):
    #load dataset
    gisto=open('dataset.pkl','rb')
    data_gr=joblib.load(gisto)
    data=data_gr
    fig = plt.figure(figsize=(10, 5))
    if col1=='Student_ID':
        if col2=='Test_1':
            x=list(data.Student_ID)
            y=list(data.Test_1)
        elif col2=='Test_2':
            x=list(data.Student_ID)
            y=list(data.Test_2)
        elif col2=='Test_3':
            x=list(data.Student_ID)
            y=list(data.Test_3)
        elif col2=='Test_4':
            x=list(data.Student_ID)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Student_ID)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Student_ID)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Student_ID)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Student_ID)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Student_ID)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Student_ID)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Student_ID)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Student_ID)
            y=list(data.Test_12)
            
    elif col1=='Test_1':
        if col2=='Student_ID':
            x=list(data.Test_1)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_1)
            y=list(data.Test_2)
        elif col2=='Test_3':
            x=list(data.Test_1)
            y=list(data.Test_3)
        elif col2=='Test_4':
            x=list(data.Test_1)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_1)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_1)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_1)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_1)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_1)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_1)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_1)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_1)
            y=list(data.Test_12)
            
    elif col1=='Test_2':
        if col2=='Student_ID':
            x=list(data.Test_2)
            y=list(data.Student_ID)
        elif col2=='Test_1':
            x=list(data.Test_2)
            y=list(data.Test_1)
        elif col2=='Test_3':
            x=list(data.Test_2)
            y=list(data.Test_3)
        elif col2=='Test_4':
            x=list(data.Test_2)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_2)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_2)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_2)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_2)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_2)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_2)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_2)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_2)
            y=list(data.Test_12)
            
    elif col1=='Test_3':
        if col2=='Student_ID':
            x=list(data.Test_3)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_3)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_3)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_3)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_3)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_3)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_3)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_3)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_3)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_3)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_3)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_3)
            y=list(data.Test_12)
            
    elif col1=='Test_4':
        if col2=='Student_ID':
            x=list(data.Test_4)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_4)
            y=list(data.Test_2)
        elif col2=='Test_3':
            x=list(data.Test_3)
            y=list(data.Test_4)
        elif col2=='Test_1':
            x=list(data.Test_4)
            y=list(data.Test_1)
        elif col2=='Test_5':
            x=list(data.Test_4)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_4)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_4)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_4)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_4)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_4)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_4)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_4)
            y=list(data.Test_12)
    
    elif col1=='Test_5':
        if col2=='Student_ID':
            x=list(data.Test_5)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_5)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_5)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_5)
            y=list(data.Test_4)
        elif col2=='Test_1':
            x=list(data.Test_5)
            y=list(data.Test_1)
        elif col2=='Test_6':
            x=list(data.Test_5)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_5)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_5)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_5)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_5)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_5)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_5)
            y=list(data.Test_12)
    
    elif col1=='Test_6':
        if col2=='Student_ID':
            x=list(data.Test_6)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_6)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_6)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_6)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_6)
            y=list(data.Test_5)
        elif col2=='Test_3':
            x=list(data.Test_6)
            y=list(data.Test_3)
        elif col2=='Test_7':
            x=list(data.Test_6)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_6)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_6)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_6)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_6)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_6)
            y=list(data.Test_12)
    
    elif col1=='Test_7':
        if col2=='Student_ID':
            x=list(data.Test_7)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_7)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_7)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_7)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_7)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_7)
            y=list(data.Test_6)
        elif col2=='Test_3':
            x=list(data.Test_7)
            y=list(data.Test_3)
        elif col2=='Test_8':
            x=list(data.Test_7)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_7)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_7)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_7)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_7)
            y=list(data.Test_12)
    
    elif col1=='Test_8':
        if col2=='Student_ID':
            x=list(data.Test_8)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_8)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_8)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_8)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_8)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_8)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_8)
            y=list(data.Test_7)
        elif col2=='Test_3':
            x=list(data.Test_8)
            y=list(data.Test_3)
        elif col2=='Test_9':
            x=list(data.Test_8)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_8)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_8)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_8)
            y=list(data.Test_12)
    
    elif col1=='Test_9':
        if col2=='Student_ID':
            x=list(data.Test_9)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_9)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_9)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_9)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_9)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_9)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_9)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_9)
            y=list(data.Test_8)
        elif col2=='Test_3':
            x=list(data.Test_9)
            y=list(data.Test_3)
        elif col2=='Test_10':
            x=list(data.Test_9)
            y=list(data.Test_10)
        elif col2=='Test_11':
            x=list(data.Test_9)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_9)
            y=list(data.Test_12)
            
    elif col1=='Test_10':
        if col2=='Student_ID':
            x=list(data.Test_10)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_10)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_10)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_10)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_10)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_10)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_10)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_10)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_10)
            y=list(data.Test_9)
        elif col2=='Test_3':
            x=list(data.Test_10)
            y=list(data.Test_3)
        elif col2=='Test_11':
            x=list(data.Test_10)
            y=list(data.Test_11)
        elif col2=='Test_12':
            x=list(data.Test_10)
            y=list(data.Test_12)
    
    elif col1=='Test_11':
        if col2=='Student_ID':
            x=list(data.Test_11)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_11)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_11)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_11)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_11)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_11)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_11)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_11)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_11)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_11)
            y=list(data.Test_10)
        elif col2=='Test_3':
            x=list(data.Test_11)
            y=list(data.Test_3)
        elif col2=='Test_12':
            x=list(data.Test_11)
            y=list(data.Test_12)
    
    elif col1=='Test_12':
        if col2=='Student_ID':
            x=list(data.Test_12)
            y=list(data.Student_ID)
        elif col2=='Test_2':
            x=list(data.Test_12)
            y=list(data.Test_2)
        elif col2=='Test_1':
            x=list(data.Test_12)
            y=list(data.Test_1)
        elif col2=='Test_4':
            x=list(data.Test_12)
            y=list(data.Test_4)
        elif col2=='Test_5':
            x=list(data.Test_12)
            y=list(data.Test_5)
        elif col2=='Test_6':
            x=list(data.Test_12)
            y=list(data.Test_6)
        elif col2=='Test_7':
            x=list(data.Test_12)
            y=list(data.Test_7)
        elif col2=='Test_8':
            x=list(data.Test_12)
            y=list(data.Test_8)
        elif col2=='Test_9':
            x=list(data.Test_12)
            y=list(data.Test_9)
        elif col2=='Test_10':
            x=list(data.Test_12)
            y=list(data.Test_10)
        elif col2=='Test_3':
            x=list(data.Test_12)
            y=list(data.Test_3)
        elif col2=='Test_11':
            x=list(data.Test_12)
            y=list(data.Test_11)
    
    
    #x=list(data.Student_ID)
    #y=list(data.Test_1)
    
    # creating the bar plot
    plt.barh(x, y, color='blue')

    # Add labels
    plt.xlabel(col2)
    plt.ylabel(col1)
    st.pyplot(fig)


# In[ ]:


#Функция для проверки гипотезы
def hypot(col1, col2, hyp):
    #load dataset
    gisto=open('dataset.pkl','rb')
    data_gr=joblib.load(gisto)
    data=data_gr
    
    #'T-test', 'Mann-Whitney U-test', 'Bootstraping'
    # Когда мы имеем в одной колонке Student_ID и в другой очки за тест, 
    # то проверяем значения тестов >=90 и ==100.
    
    # В случае с 2 колонками, состоящих из баллов за тесты, я проверяю значения 
    # обоих тестов >=90
    
    if col1=='Student_ID':
        if col2=='Test_1':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_1 == 100].Test_1))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_1 == 100].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_2 == 100].Test_2))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_2 == 100].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_3 == 100].Test_3))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_3 == 100].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_4 == 100].Test_4))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_4 == 100].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_5 == 100].Test_5))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_5 == 100].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_6 == 100].Test_6))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_6 == 100].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_7 == 100].Test_7))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_7 == 100].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_8 == 100].Test_8))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_8 == 100].Test_8)
                st.write(bootdata)
        elif col2=='Test_9':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_9 == 100].Test_9))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_9 == 100].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_10 == 100].Test_10))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_10 == 100].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_11 == 100].Test_11))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_11 == 100].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_12 == 100].Test_12))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_12 == 100].Test_12)
                st.write(bootdata)
           
        
    elif col1=='Test_1':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_1 == 100].Test_1))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_1 == 100].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
            
            
    elif col1=='Test_2':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_2 == 100].Test_2))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_2 == 100].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
            
            
    elif col1=='Test_3':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_3 == 100].Test_3))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_3 == 100].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
            
    elif col1=='Test_4':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_4 == 100].Test_4))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_4 == 100].Test_4)
                st.write(bootdata)
                
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
                
    
    elif col1=='Test_5':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_5 == 100].Test_5))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_5 == 100].Test_5)
                st.write(bootdata)
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
    
    
    elif col1=='Test_6':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_6 == 100].Test_6))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_6 == 100].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
    
    
    elif col1=='Test_7':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_7 == 100].Test_7))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_7 == 100].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_1 >= 90].Test_1, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_1 >= 90].Test_1, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_1 >= 90].Test_1, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_7 >= 90].Test_7, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_7 >= 90].Test_7, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_7 >= 90].Test_7, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
            
            
    elif col1=='Test_8':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_8 == 100].Test_8))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_8 == 100].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_2 >= 90].Test_2, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_2 >= 90].Test_2, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_2 >= 90].Test_2, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_8 >= 90].Test_8, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_8 >= 90].Test_8, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_8 >= 90].Test_8, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
            
            
    elif col1=='Test_9':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_9 == 100].Test_9))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_9 == 100].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_3 >= 90].Test_3, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_3 >= 90].Test_3, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_9 >= 90].Test_9, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_9 >= 90].Test_9, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_3 >= 90].Test_3, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_9 >= 90].Test_9, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
            
   
    elif col1=='Test_10':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_10 == 100].Test_10))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_10 == 100].Test_10)
                st.write(bootdata)
                
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_4 >= 90].Test_4, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_4 >= 90].Test_4, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_4 >= 90].Test_4, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_10 >= 90].Test_10, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_10 >= 90].Test_10, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_10 >= 90].Test_10, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
                
    
    elif col1=='Test_11':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_11 == 100].Test_11))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_11 == 100].Test_11)
                st.write(bootdata)
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_6 >= 90].Test_6,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_6 >= 90].Test_6))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_6 >= 90].Test_6)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_8 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_5 >= 90].Test_5, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_5 >= 90].Test_5, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_5 >= 90].Test_5, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_12':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_11 >= 90].Test_11, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_11 >= 90].Test_11, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_11 >= 90].Test_11, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)
    
    
    elif col1=='Test_12':
        if col2=='Student_ID':
            #1
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_12 == 100].Test_12))
            #2  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_12 == 100].Test_12)
                st.write(bootdata)
                
        elif col2=='Test_2':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_2 >= 90].Test_2,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_2 >= 90].Test_2))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_2 >= 90].Test_2)
                st.write(bootdata)
                
        elif col2=='Test_1':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_1 >= 90].Test_1,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_1 >= 90].Test_1))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_1 >= 90].Test_1)
                st.write(bootdata)
                
        elif col2=='Test_4':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_4 >= 90].Test_4,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_4 >= 90].Test_4))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_4 >= 90].Test_4)
                st.write(bootdata)
                
        elif col2=='Test_5':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_5 >= 90].Test_5,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_5 >= 90].Test_5))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_5 >= 90].Test_5)
                st.write(bootdata)
               
        
        elif col2=='Test_3':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_3 >= 90].Test_3,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_3 >= 90].Test_3))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_3 >= 90].Test_3)
                st.write(bootdata)
                
        elif col2=='Test_7':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_7 >= 90].Test_7,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_7 >= 90].Test_7))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_7 >= 90].Test_7)
                st.write(bootdata)
                
        elif col2=='Test_8':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_8 >= 90].Test_8,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_8 >= 90].Test_8))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_6, data[data.Test_12 >= 90].Test_8)
                st.write(bootdata)
                
        elif col2=='Test_9':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_9 >= 90].Test_9,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_9 >= 90].Test_9))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_9 >= 90].Test_9)
                st.write(bootdata)
                
        elif col2=='Test_10':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_10 >= 90].Test_10,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_10 >= 90].Test_10))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_10 >= 90].Test_10)
                st.write(bootdata)
                
        elif col2=='Test_11':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_12 >= 90].Test_12, 
                                         data[data.Test_11 >= 90].Test_11,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_12 >= 90].Test_12, 
                             data[data.Test_11 >= 90].Test_11))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_12 >= 90].Test_12, data[data.Test_11 >= 90].Test_11)
                st.write(bootdata)
                
        elif col2=='Test_6':
            #1. T-test
            if hyp=='T-test':
                t_test = stats.ttest_ind(data[data.Test_6 >= 90].Test_6, 
                                         data[data.Test_12 >= 90].Test_12,
                                         equal_var=False)
                st.write(f'p-value: {t_test.pvalue / 2:.4f}')
            #2
            if hyp == 'Mann-Whitney U-test':
                st.write(mannwhitneyu(data[data.Test_6 >= 90].Test_6, 
                             data[data.Test_12 >= 90].Test_12))
            #3  
            elif hyp == 'Bootstraping':
                def get_bootstrap(
                    data_column_1, 
                    data_column_2,
                    boot_it = 1000, # boot iterations
                    statistic = np.mean, # what we try to find
                    bootstrap_conf_level = 0.95 # confidence level
                ):
                    boot_len = max([len(data_column_1), len(data_column_2)])
                    boot_data = []
                    for i in tqdm(range(boot_it)):
                        samples_1 = data_column_1.sample(
                            boot_len, 
                            replace = True 
                        ).values
                        
                        samples_2 = data_column_2.sample(
                            boot_len,
                            replace = True
                        ).values
                        
                        boot_data.append(statistic(samples_1-samples_2)) 
                    pd_boot_data = pd.DataFrame(boot_data)
                    
                    left_quant = (1 - bootstrap_conf_level)/2
                    right_quant = 1 - (1 - bootstrap_conf_level) / 2
                    quants = pd_boot_data.quantile([left_quant, right_quant])
                    p_1 = norm.cdf(
                        x = 0, 
                        loc = np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_2 = norm.cdf(
                        x = 0, 
                        loc = -np.mean(boot_data), 
                        scale = np.std(boot_data)
                    )
                    p_value = min(p_1, p_2) * 2
                    
                    return {"p-value": p_value}
                bootdata = get_bootstrap(data[data.Test_6 >= 90].Test_6, data[data.Test_12 >= 90].Test_12)
                st.write(bootdata)


# Главная функция _run_ выводит на веб-странице streamlit выпадающие меню, кнопки с гистограммой и результатом проверки гипотезы.
# 
# Сначала выбираются 2 разные колонны, 

# In[ ]:


def run():
    html_temp=""""""
    st.markdown(html_temp)
    col1=st.selectbox(
        'Choose the first column', ('Student_ID','Test_1','Test_2','Test_3',
         'Test_4','Test_5','Test_6','Test_7','Test_8','Test_9','Test_10',
         'Test_11','Test_12'))
    st.write('You selected:', col1)
    
    st.markdown(html_temp)
    col2=st.selectbox(
        'Choose the second column', ('Student_ID','Test_1','Test_2','Test_3',
         'Test_4','Test_5','Test_6','Test_7','Test_8','Test_9','Test_10',
         'Test_11','Test_12'))
    st.write('You selected:', col2)
    
    #show histogram
    if st.button('Histogram') and col1!=col2:
        gist(col1,col2)
        st.success("Гистограмма из колонок 1 и 2, выбранных вами.".format(''))
    
    
    #choosing and checking hypothes
    st.markdown(html_temp)
    if col1=='Student_ID' or col2=='Student_ID':
        hyp = st.selectbox( 'Выберите проверку гипотезы:', 
                        ('Mann-Whitney U-test', 'Bootstraping'))
    else:
        hyp = st.selectbox( 'Выберите проверку гипотезы:', ('T-test', 
                        'Mann-Whitney U-test', 'Bootstraping'))
    st.write('You selected:', hyp)
    
    if st.button('Result'):
        hypot(col1, col2, hyp)

run()

