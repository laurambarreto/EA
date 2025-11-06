## -- IMPORT DAS BIBLIOTECAS NECESSÁRIAS -- ##
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt, colormaps
import math
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.stats import kstest, zscore
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import ranksums
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skrebate import ReliefF
from sklearn.feature_selection import f_classif  
import matplotlib.patches as mpatches

## ------------ EXERCÍCIO 2 ----------- ##
## -- DADOS DE TODOS OS PARTICIPANTES NUMA MATRIZ -- ##
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

# Verificar a quantidade de atividades distintas
atividades = np.unique(dados[:, 11].astype(int))
print(atividades)

## ------------ EXERCÍCIO 3 ------------ ##
## -- MÓDULOS DAS MEDIÇÕES DOS 3 SENSORES -- ##
def modulo ():
    # Cria lista com os módulos
    novas_linhas = []
    for i in range (dados.shape [0]):
        # Cálculo dos módulos (√(sensor_x^2 + sensor_y^2 + sensor_z^2))
        modulo_acc = math.sqrt (float (dados[i][1])**2 + float (dados[i][2])**2 + float (dados[i][3])**2)
        modulo_gyr = math.sqrt (float (dados[i][4])**2 + float (dados[i][5])**2 + float (dados[i][6])**2)
        modulo_mag = math.sqrt (float (dados[i][7])**2 + float (dados[i][8])**2 + float (dados[i][9])**2)

        # Adiciona os módulos no final da linha
        nova_linha = np.append(dados[i], [modulo_acc, modulo_gyr, modulo_mag])

        # Adiciona nova linha à lista novas_linhas
        novas_linhas.append(nova_linha)

    # Converter a lista de linhas de volta para array numpy
    return np.array(novas_linhas)

dados = modulo ()

# Ler CSV
df_X = pd.read_csv("Matriz_Features.csv")
df_y = pd.read_csv("Lista_Atividades.csv")

df_X['Atividade'] = df_y['Atividade']

print(df_X.groupby('id_part')['Atividade'].unique())
print(df_X)

# Converter para arrays NumPy
X_total = df_X.values       # matriz sem nomes das colunas
y_total = df_y.values.ravel()  # vetor 1D

print(X_total.shape)  # (n_amostras_filtradas, n_features + 1) -> inclui 'Participante'
print(y_total.shape)  # (n_amostras_filtradas,)


## ------------ EXERCÍCIO 1.1 ------------ ##
# Verificar balanceamento das atividades

# Obter atividades únicas e suas contagens
atividades_unicas, contagens = np.unique(y_total, return_counts = True)
# Plot gráfico de barras
plt.figure(figsize = (10, 5))
plt.bar(atividades_unicas, contagens, color = 'skyblue', edgecolor = 'black', zorder = 3)
plt.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4, zorder = 0)
plt.title("Distribuição das Atividades", fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold') 
plt.xlabel("Atividade")
plt.ylabel("Contagem")  
plt.show()


## ------------ EXERCÍCIO 1.2 ------------ ##
# Função para criar k novas amostras sintéticas numa atividade
def gerar_amostras_sinteticas(X, y, n_novos, k_vizinhos, atividade, participante = None):
    if participante is not None:
        X_atividade = X[(X[:, -1] ==  atividade) & (X[:, -2] ==  participante)]
    else:
        X_atividade = X[y ==  atividade]

    n_amostras = X_atividade.shape[0]
    X_sinteticas = []

    for _ in range(n_novos):
        idx = np.random.randint(0, n_amostras)
        amostra_base = X_atividade[idx]
        
        # Calcular distâncias entre a amostra e todas as outras da mesma atividade
        dic = {}
        for vizinho in range(n_amostras):
            if vizinho !=  idx:
                distancia = np.linalg.norm(amostra_base - X_atividade[vizinho])
                dic[vizinho] = distancia  

        # Ordenar pontos pela distância crescentemente 
        distancias_ordenadas = dict(sorted(dic.items(), key = lambda item: item[1]))

        # Obter os k vizinhos mais próximos 
        vizinhos_proximos = list(distancias_ordenadas.keys())[:k_vizinhos]

        # Escolher um vizinho aleatório
        vizinho_aleatorio = np.random.choice(vizinhos_proximos)

        # Obter o vetor entre a amostra base e o vizinho aleatório
        vetor = X_atividade[vizinho_aleatorio] - amostra_base

        # Gerar nova amostra sintética
        fator = np.random.rand()
        nova_amostra = amostra_base + fator * vetor 

        X_sinteticas.append(nova_amostra)

    return np.array(X_sinteticas)

gerar_amostras_sinteticas(X_total, y_total, n_novos = 3, k_vizinhos = 5, atividade = 4, participante = 3)


# ------------ EXERCÍCIO 1.3 ------------ #
# Plot 2D das amostras originais e sintéticas (com as duas primeiras features no eixo X e Y)
def plot_amostras_2D(X, y, atividade, n_sinteticas, k_vizinhos, participante = None):
    # Gerar amostras sintéticas para a atividade
    X_sinteticas = gerar_amostras_sinteticas(
        X, y,
        n_novos = n_sinteticas,
        k_vizinhos = k_vizinhos,
        atividade = atividade,
        participante = participante
    )
    
    # Obter as atividades únicas
    atividades_unicas = np.unique(y)
    
    # Paleta de 7 cores para as atividades
    cores = [
        "#FF6F61",  # 1 - Coral
        "#FF69C1",  # 2 - Roxo
        "#EF75FF",  # 3 - Verde médio
        "#B2FF66",  # 4 - Verde claro (atividade a destacar)
        "#FFA500",  # 5 - Laranja
        "#ECF665",  # 6 - Verde água
        "#8067FD"   # 7 - Rosa claro
    ]
    
    plt.figure(figsize = (10, 6))
    # Plot de cada atividade com cor diferente
    for i, act in enumerate(atividades_unicas):
        X_act = X[y ==  act]
        plt.scatter(X_act[:, 0], X_act[:, 1], color = cores[i], label = f'Atividade {int(act)}', alpha = 0.4)
    
    # Plot das amostras sintéticas (verde escuro, mais visíveis)
    plt.scatter(X_sinteticas[:, 0], X_sinteticas[:, 1],
                color = 'darkgreen', marker = 'X', s = 100,
                label = f'Amostras Sintéticas (Atividade {atividade})')
    
    # Estilo e legendas
    plt.title(f'Amostras Originais e Sintéticas - Atividade {atividade}',
              fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4)
    plt.show()

plot_amostras_2D(X_total, y_total, atividade = 4, n_sinteticas = 3, k_vizinhos = 5, participante = 3)