import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def matriz (n):
    array_individuo = []
    for i in range(1,6):
        df = pd.read_csv(f"Part 0/part{n}dev{i}.csv")
        array_individuo.append (df.values)
    matriz = np.vstack(array_individuo) 
    return matriz

dados = matriz (0)
print(dados.shape)
print (dados)

atividades = []
for n in dados:
    valor = n[11]
    if valor not in atividades:
        atividades.append (valor)
            
print (atividades)
num_atividades = len (atividades)
print (num_atividades)


contagem = np.zeros ([16,1])
for i in range (len(dados)):
    atividade = int(dados [i][11])
    contagem [atividade - 1] += 1



