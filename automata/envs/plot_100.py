# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:20:53 2021

@author: Lucas
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#path = "C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs/dados_lucas_100/out1.csv"
path = "C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs/dados_lucas_100/out4.csv"

df = pd.read_csv (path)

sns.lineplot(data=df)
#plt.savefig('C:/Users/Lucas/Google Drive/Pesquisa/TCC/automata_gym_cont/automata/envs/Plots/General/100_random_instances_table.eps', dpi=300)


#x=xlabel_name, y="Reworks/Cars Produced", style="Event",  hue="Method", data=a, markers=True,  err_style=None)