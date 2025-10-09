import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def matriz ():
    array_individuo = []
    for part in range(0, 15):
        for device in range(1,6):
            df = pd.read_csv(f"Part {part}/part{part}dev{device}.csv")
            df["Part"] = int(part+1)
            array_individuo.append (df.values)
        matriz = np.vstack(array_individuo) 
    return matriz

dados = matriz ()
print(dados)
print(dados.shape)
atividades = []
for n in dados:
    valor = n[11]
    if valor not in atividades:
        atividades.append (valor)
            
print (atividades)
num_atividades = len (atividades)
print (num_atividades)






