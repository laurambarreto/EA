import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import math

## EXERCÍCIO ...
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

atividades = np.unique(dados[:, 11].astype(int))
print(atividades)

def modulo ():
    novas_linhas = []
    for i in range (dados.shape [0]):
        modulo_acc = math.sqrt (int (dados[i][1])**2 + int (dados[i][2])**2 + int (dados[i][3])**2)
        modulo_gyr = math.sqrt (int (dados[i][4])**2 + int (dados[i][5])**2 + int (dados[i][6])**2)
        modulo_mag = math.sqrt (int (dados[i][7])**2 + int (dados[i][8])**2 + int (dados[i][9])**2)
        # Adicionar os módulos no final da linha
        nova_linha = np.append(dados[i], [modulo_acc, modulo_gyr, modulo_mag])
        novas_linhas.append(nova_linha)

    # Converter a lista de linhas de volta para array numpy
    return np.array(novas_linhas)

dados = modulo ()


def boxplots():
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colunas_modulos = [13, 14, 15]         # Índices dos módulos

    # Uma figura para cada device
    for i in range (1, 6): 
        
        # Filtrar só os dados deste device
        device_data = dados[dados[:, 0] == i]

        # Criar uma figura com 3 subplots lado a lado
        plt.figure(figsize=(12, 5))
        plt.suptitle(f'Device {i} - Boxplots por Módulos e Atividades', fontsize = 14, fontweight = 'bold')

        # Para cada um dos 3 módulos
        for i, (nome_modulo, col_modulo) in enumerate(zip(nomes_modulos, colunas_modulos), start = 1):
            # Criar listas de valores por atividade (para este módulo)
            valores_boxplot = [device_data[device_data[:, 11] == a, col_modulo] for a in atividades]

            # Subplot desse módulo com as 16 atividades
            plt.subplot(1, 3, i)
            plt.boxplot(valores_boxplot, showfliers = True)
            plt.title(nome_modulo)
            plt.xlabel('Atividade (1-16)')
            plt.ylabel('Módulo')
            plt.xticks(range(1, len(atividades) + 1), atividades, rotation = 45)
            plt.grid(True, linestyle = '--', alpha = 0.6) # grelhas a tracejado e ligeiramente transparente

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

#boxplots()


def densidade_outliers_por_modulo_atividade():
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colunas_modulos = [13, 14, 15]         # Índices dos módulos

    # Filtrar só os dados do device 2
    device_data = dados[dados[:, 0] == 2]

    for (nome_modulo, col_modulo)  in zip(nomes_modulos, colunas_modulos):
        print(f'\nDensidade de Outliers por atividade - {nome_modulo} (Device 2)')

        for atividade in atividades:
            # Filtrar os dados para a atividade atual
            atividade_data = device_data[device_data[:, 11] == atividade, col_modulo]
            
            # Calcular Q1, Q3 e IQR
            Q1 = np.percentile(atividade_data, 25)
            Q3 = np.percentile(atividade_data, 75)
            IQR = Q3 - Q1
            
            # Definir limites para outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Contar outliers a cima ou a baixo dos limites
            outliers = atividade_data[(atividade_data < lower_bound) | (atividade_data > upper_bound)]
            num_outliers = len(outliers)
            total_points = len(atividade_data)
            
            # Calcular densidade de outliers
            densidade_outlier = num_outliers / total_points * 100
            if atividade < 10:
                print(f'Atividade {atividade}:  {densidade_outlier:.2f} %')
            else: 
                print(f'Atividade {atividade}: {densidade_outlier:.2f} %')
       
#densidade_outliers_por_modulo_Atividade()


def identifica_outliers(array, k):
    # Calcular média e desvio padrão
    mean = np.mean(array)
    std = np.std(array)

    # Calcular Z-scores
    zscores = np.abs((array - mean) / std)

    # Identificar os indíces dos outliers
    indices_outliers = np.where(zscores > k)[0]

    return indices_outliers

#outliers = identifica_outliers(dados[13], 1.5)
#print(outliers)


def plot_outliers_porModulo_eAtividade_numDevice(k = 3.5):
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colunas_modulos = [13, 14, 15]

    # Uma figura para cada device
    for device_id in range(1, 6): 
        # Filtrar só os dados deste device
        device_data = dados[dados[:, 0] == device_id]

        # Criar uma figura com 3 subplots lado a lado
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'Device {device_id} - Outliers (Z-score > {k})', fontsize = 14, fontweight = 'bold')

        # Para cada um dos 3 módulos
        for subplot_idx, (nome_modulo, col_modulo) in enumerate(zip(nomes_modulos, colunas_modulos), start = 1):
            plt.subplot(1, 3, subplot_idx)

            for atividade in atividades:
                # Filtrar dados dessa atividade
                atividade_data = device_data[device_data[:, 11].astype(int) == atividade, col_modulo].astype(float)

                # Identificar outliers nesta atividade
                indices_outliers = identifica_outliers(atividade_data, k)

                # Posições no eixo X (para separar visualmente as atividades)
                x_vals = np.full(len(atividade_data), atividade)

                # Pontos normais (azuis)
                plt.scatter(x_vals, atividade_data, color = 'blue', s = 7, alpha=0.6)

                # Se houver Outliers (vermelhos)
                if len(indices_outliers) > 0:
                    plt.scatter(x_vals[indices_outliers], atividade_data[indices_outliers], color = 'red', s = 10, label = 'Outlier')

            plt.title(nome_modulo)
            plt.xlabel('Atividade (1-16)')
            plt.ylabel('Módulo')
            plt.xticks(atividades)
            plt.grid(True, linestyle = '--', alpha =0.6)

        plt.tight_layout(rect = [0, 0, 1, 0.93])
        plt.show()

plot_outliers_porModulo_eAtividade_numDevice(k = 3.5)

#def kmeans(n):
