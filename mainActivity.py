import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import math

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


def modulo ():
    for i in range (dados.shape [0]):
        print (dados[i])
        modulo_acc = math.sqrt (int (dados[i][1])**2 + int (dados[i][2])**2 + int (dados[i][3])**2)
        modulo_gyr = math.sqrt (int (dados[i][4])**2 + int (dados[i][5])**2 + int (dados[i][6])**2)
        modulo_mag = math.sqrt (int (dados[i][7])**2 + int (dados[i][8])**2 + int (dados[i][9])**2)
        dados[i].add (modulo_acc)
        dados[i].add (modulo_gyr)
        dados[i].add (modulo_mag)
        

modulo ()
print (dados [0][13])
print (dados [0][14])
print (dados [0][15])

print (dados [560][13])
print (dados [560][14])
print (dados [560][15])
