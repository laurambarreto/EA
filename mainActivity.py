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
from scipy.stats import ranksums
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler

## ------------ EXERCÍCIO 2 ------------ ##
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

def modulo ():
    novas_linhas = []
    for i in range (dados.shape [0]):
        modulo_acc = math.sqrt (float (dados[i][1])**2 + float (dados[i][2])**2 + float (dados[i][3])**2)
        modulo_gyr = math.sqrt (float (dados[i][4])**2 + float (dados[i][5])**2 + float (dados[i][6])**2)
        modulo_mag = math.sqrt (float (dados[i][7])**2 + float (dados[i][8])**2 + float (dados[i][9])**2)
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
    for device in range (1, 6): 
        
        # Filtrar só os dados deste device
        device_data = dados[dados[:, 0] == device]

        # Criar uma figura com 3 subplots lado a lado
        plt.figure(figsize = (12, 5))
        plt.suptitle(f'Device {device} - Boxplots por Módulos e Atividades', fontsize = 14, fontweight = 'bold')

        # Para cada um dos 3 módulos
        for i, (nome_modulo, col_modulo) in enumerate(zip(nomes_modulos, colunas_modulos), start = 1):
            # Criar listas de valores por atividade (para este módulo)
            valores_boxplot = [device_data[device_data[:, 11] == a, col_modulo] for a in atividades]

            # Subplot desse módulo com as 16 atividades
            plt.subplot(1, 3, i)
            plt.boxplot(valores_boxplot, showfliers = True)
            plt.boxplot(valores_boxplot, showfliers = True)
            plt.title(nome_modulo)
            plt.xlabel('Atividade (1-16)')
            plt.ylabel('Módulo')
            plt.xticks(range(1, len(atividades) + 1), atividades, rotation = 45)
            plt.grid(True, linestyle = '--', alpha = 0.6) # grelhas a tracejado e ligeiramente transparente
            plt.grid(True, linestyle = '--', alpha = 0.6) # grelhas a tracejado e ligeiramente transparente

        plt.tight_layout(rect = [0, 0, 1, 0.93])
        plt.show()

#boxplots()


def densidade_outliers_por_modulo_atividade():
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colunas_modulos = [13, 14, 15]  # Índices dos módulos

    # Filtrar só os dados do device 2
    device_data = dados[dados[:, 0] == 2]

    # Guardar as densidades para plot
    densidades_por_modulo = []

    for (nome_modulo, col_modulo) in zip(nomes_modulos, colunas_modulos):
        print(f'\nDensidade de Outliers por atividade - {nome_modulo} (Device 2)')

        densidades = []
        for atividade in atividades:
            # Filtrar os dados para a atividade atual
            atividade_data = device_data[device_data[:, 11] == atividade, col_modulo].astype(float)

            # Calcular Q1, Q3 e IQR
            Q1 = np.percentile(atividade_data, 25)
            Q3 = np.percentile(atividade_data, 75)
            IQR = Q3 - Q1

            # Definir limites para outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Contar outliers
            outliers = atividade_data[(atividade_data < lower_bound) | (atividade_data > upper_bound)]
            num_outliers = len(outliers)
            total_points = len(atividade_data)

            # Calcular densidade (%)
            densidade_outlier = (num_outliers / total_points * 100) if total_points > 0 else 0
            densidades.append(densidade_outlier)

            print(f'Atividade {atividade}: {densidade_outlier:.2f}%')

        densidades_por_modulo.append(densidades)

    fig, axs = plt.subplots(1, 3, figsize = (14, 5), sharey = True)

    for i, ax in enumerate(axs):
        ax.set_axisbelow(True)
        ax.grid(True, which = 'major', linestyle = '--', linewidth = 0.7, alpha = 0.4)
        ax.bar(range(len(atividades)), densidades_por_modulo[i], color = 'pink', edgecolor = 'black', linewidth = 0.5)
        ax.set_ylim(0, 100)
        ax.set_title(nomes_modulos[i], fontsize = 12, fontweight = 'bold')
        ax.set_xlabel('Atividade')
        if i == 0:
            ax.set_ylabel('Densidade de Outliers (%)')
        ax.set_xticks(range(len(atividades)))
        ax.set_xticklabels(atividades, rotation = 45)

    fig.suptitle('Densidade de Outliers por Módulo e Atividade (Device 2)', fontsize = 14, fontweight = 'bold')
    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.show()

#densidade_outliers_por_modulo_atividade()


def identifica_outliers(array, k):
    # Calcular média e desvio padrão
    mean = np.mean(array)
    std = np.std(array)

    # Calcular Z-scores
    zscores = np.abs((array - mean) / std)

    # Identificar os indíces dos outliers
    indices_outliers = np.where(zscores > k)[0]

    return indices_outliers

#outliers_1 = identifica_outliers(dados[13], 1.5)
#print (outliers_1)


def plot_outliers_porModulo_eAtividade_numDevice(k):
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colunas_modulos = [13, 14, 15]

    # Uma figura para cada device
    for device_id in range(1, 6): 
        # Filtrar só os dados deste device
        device_data = dados[dados[:, 0] == device_id]

        # Criar uma figura com 3 subplots lado a lado
        plt.figure(figsize = (15, 6))
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
                plt.scatter(x_vals, atividade_data, color = 'blue', s = 7, alpha = 0.6)

                # Se houver Outliers (vermelhos)
                if len(indices_outliers) > 0:
                    plt.scatter(x_vals[indices_outliers], atividade_data[indices_outliers], color = 'red', s = 10, label = 'Outlier')
                    plt.scatter(x_vals[indices_outliers], atividade_data[indices_outliers], color = 'red', s = 10, label = 'Outlier')

            plt.title(nome_modulo)
            plt.xlabel('Atividade (1-16)')
            plt.ylabel('Módulo')
            plt.xticks(atividades)
            plt.grid(True, linestyle = '--', alpha = 0.6)

        plt.tight_layout(rect = [0, 0, 1, 0.93])
        plt.tight_layout(rect = [0, 0, 1, 0.93])
        plt.show()

#plot_outliers_porModulo_eAtividade_numDevice(3)
#plot_outliers_porModulo_eAtividade_numDevice(3.5)
#plot_outliers_porModulo_eAtividade_numDevice(4)


def kmeans (dados, n, max_iter = 100):
    # escolher n centroides iniciais aleatoriamente do conjunto de dados
    indices = np.random.choice (dados.shape[0], n, replace = False)
    centroides = dados [indices]

    for _ in range (max_iter):
        # calcular a distância de cada ponto a todos os centroides
        distancias = np.linalg.norm (dados [:, np.newaxis] - centroides, axis = 2)
        # atribuir cada ponto ao centroide mais próximo
        labels = np.argmin (distancias, axis = 1)
        # guardar centroides da iteração anterior
        centroides_old = centroides.copy ()
        # atualizar os centroides como a média dos pontos atribuídos a cada cluster
        for i in range (n):
            pontos_cluster = dados [labels == i]
            centroides [i] = np.mean (pontos_cluster, axis = 0)
        # se os centroides não mudaram significativamente, parar
        if np.allclose (centroides, centroides_old):
            break
    
    outliers_idx = []
    for i in range(n):
        pontos_cluster = dados[labels == i]
        centroid = centroides[i]
        # Distâncias dos pontos ao respetivo centróide
        distancias_cluster = np.linalg.norm(pontos_cluster - centroid, axis=1)
        # Calcular Q1, Q3, IQR
        Q1 = np.percentile(distancias_cluster, 25)
        Q3 = np.percentile(distancias_cluster, 75)
        IQR = Q3 - Q1
        # Limites para outliers
        lower = Q1 - 4 * IQR
        upper = Q3 + 4 * IQR
        # Índices globais dos outliers
        cluster_indices = np.where(labels == i)[0]
        outlier_indices = cluster_indices[(distancias_cluster < lower) | (distancias_cluster > upper)]
        outliers_idx.extend(outlier_indices)

    return centroides, labels, np.array(outliers_idx)


def kmeans_outliers_porDevice_Atividades_Modulos(device_id, n_clusters):
    nomes_modulos = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colunas_modulos = [13, 14, 15]

    # Filtrar só os dados deste device
    device_data = dados[dados[:, 0] == device_id]
    
    # Criar uma figura com 3 subplots lado a lado
    plt.figure(figsize = (15, 6))
    plt.suptitle(f'Device {device_id} - Outliers (kmeans: k = {n_clusters})', fontsize = 14, fontweight = 'bold')

    # Para cada um dos 3 módulos
    for subplot_idx, (nome_modulo, col_modulo) in enumerate(zip(nomes_modulos, colunas_modulos), start = 1):
        plt.subplot(1, 3, subplot_idx)

        for atividade in atividades:
            # Filtrar dados dessa atividade
            atividade_data = device_data[device_data[:, 11].astype(int) == atividade, col_modulo].astype(float)

            # Identificar outliers nesta atividade
            centroides, labels, indice_outliers = kmeans(atividade_data.reshape(-1, 1), n_clusters)

            # Posições no eixo X (para separar visualmente as atividades)
            x_vals = np.full(len(atividade_data), atividade)

            # Pontos normais (azuis)
            plt.scatter(x_vals, atividade_data, color = 'blue', s = 7, alpha = 0.6)

            # Se houver Outliers (vermelhos)
            if len(indice_outliers) > 0:
                plt.scatter(x_vals[indice_outliers], atividade_data[indice_outliers], color = 'red', s = 10, label = 'Outlier')

        plt.title(nome_modulo)
        plt.xlabel('Atividade (1-16)')
        plt.ylabel('Módulo')
        plt.xticks(atividades)
        plt.grid(True, linestyle = '--', alpha =0.6)

    plt.tight_layout(rect = [0, 0, 1, 0.93])
    plt.show()

#kmeans_outliers_porDevice_Atividades_modulos(device_id = 2, n_clusters = 4)


# Função auxiliar: clarear cor
def clarear_cor(cor, fator = 1.6):
    """
    Clareia uma cor misturando-a com branco.
    fator > 1 -> mais clara
    """
    cor_rgb = np.array(mcolors.to_rgb(cor))
    cor_clara = 1 - (1 - cor_rgb) / fator
    return np.clip(cor_clara, 0, 1)

def kmeans_outliers_3D_porDeviceAtividade(device_id, activity, n_clusters, ):
    # Filtrar só os dados deste device
    device_data = dados[(dados[:, 0] == device_id) & (dados[:, 11] == activity)]

    device_modulos = device_data[:, 13:16].astype(float)

    # normalizar os módulos 
    mean = np.mean(device_modulos, axis=0)
    std = np.std(device_modulos, axis=0)
    device_modulos_norm = (device_modulos - mean) / std

    # confirmar se o z-score funcionou e deu média aproximada zero
    print(np.min(device_modulos_norm, axis=0))
    print(np.max(device_modulos_norm, axis=0))
    print(np.mean(device_modulos_norm, axis=0))

    centroides, labels, indice_outliers = kmeans(device_modulos_norm, n_clusters)

    # Paleta de cores fortes
    cores_base = list(mcolors.TABLEAU_COLORS.values())
    if n_clusters > len(cores_base):
        cores_base = list(mcolors.CSS4_COLORS.values())[:n_clusters]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection = '3d')

    # Plot de cada cluster
    for i in range(n_clusters):
        cluster_mask = labels == i
        cor_base = cores_base[i % len(cores_base)]

        # Gerar cor clara para os outliers
        cor_outlier = clarear_cor(cor_base, fator = 2.0)

        # Pontos normais (cores vivas e sólidas)
        normais_idx = np.setdiff1d(np.where(cluster_mask)[0], indice_outliers)
        if len(normais_idx) > 0:
            ax.scatter(device_modulos_norm[normais_idx, 0],
                       device_modulos_norm[normais_idx, 1],
                       device_modulos_norm[normais_idx, 2],
                       c = [cor_base], s = 5, alpha = 0.9, marker = 'o', label = f'Cluster {i+1}')

        # Outliers (mesma cor, mas mais clara e translúcida)
        outliers_cluster = np.intersect1d(np.where(cluster_mask)[0], indice_outliers)
        if len(outliers_cluster) > 0:
            ax.scatter(device_modulos_norm[outliers_cluster, 0],
                       device_modulos_norm[outliers_cluster, 1],
                       device_modulos_norm[outliers_cluster, 2],
                       c = [cor_outlier], s = 15, alpha = 0.4, marker = 'o', edgecolors = 'black',
                       linewidths = 0.3, label = f'Outliers C{i+1}')

    # Centróides
    ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2],
               c = 'black', s = 90, marker = 'P', label = 'Centroides')

    # Estética geral
    ax.set_title(f"Device {device_id} - Atividade {activity} (k = {n_clusters})")
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    ax.legend(loc = 'upper right')
    plt.show()
    
#for i in range(4,17):
 #   kmeans_outliers_3D_porAtividade(device_id = 2, activity = i, n_clusters = 2)


def dbscan_outliers_3D(device_id, eps=5, min_samples=10):
    # Filtrar só os dados deste device e atividade
    #device_activity_data = dados[(dados[:, 0] == device_id) & (dados[:, 11] == activity)]
    # Filtrar só device 
    device_data = dados[dados[:, 0] == device_id]
    # Extrair colunas dos módulos (X, Y, Z)
    device_modulos = device_data[:, 13:16].astype(float)
    '''
    # Normalizar (z-score)
    mean = np.mean(device_modulos, axis=0)
    std = np.std(device_modulos, axis=0)
    device_modulos_norm = (device_modulos - mean) / std
    '''
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(device_modulos)

    # Identificar outliers
    mask_inliers = labels != -1
    mask_outliers = labels == -1

    # Número de clusters (sem contar os outliers)
    unique_clusters = np.unique(labels[mask_inliers])
    n_clusters = len(unique_clusters)

    # Criar colormap viridis
    cmap = colormaps.get_cmap('viridis')
    # Gerar cores igualmente espaçadas no viridis
    colors = cmap(np.linspace(0, 1, n_clusters))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot de cada cluster com uma cor fixa
    for i, cluster_id in enumerate(unique_clusters):
        cluster_points = device_modulos[labels == cluster_id]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            color=colors[i],
            s = 25,
            label = f'Cluster {cluster_id}'
        )

    # Outliers em vermelho
    if np.any(mask_outliers):
        ax.scatter(
            device_modulos[mask_outliers, 0],
            device_modulos[mask_outliers, 1],
            device_modulos[mask_outliers, 2],
            c = 'red',
            s = 20,
            alpha = 0.5,
            label = 'Outliers'
        )

    # Rótulos e legenda
    ax.set_xlabel('Módulo ACC')
    ax.set_ylabel('Módulo GYR')
    ax.set_zlabel('Módulo MAG')
    ax.set_title(f'DBSCAN 3D — Device {device_id}')
    ax.legend()
    plt.show()

#dbscan_outliers_3D(device_id = 2, activity = 2, eps = 0.5, min_samples = 5)
#dbscan_outliers_3D(device_id = 2, eps = 0.5, min_samples = 5)


def kmeans_outliers_3D_apenasPorDevice(device_id, n_clusters):
    # Filtrar só os dados deste device
    device_data = dados[dados[:, 0] == device_id]
    device_modulos = device_data[:, 13:16].astype(float)

    # Normalizar os módulos (z-score)
    mean = np.mean(device_modulos, axis=0)
    std = np.std(device_modulos, axis=0)
    device_modulos_norm = (device_modulos - mean) / std

    # Confirmar normalização
    print("Min:", np.min(device_modulos_norm, axis=0))
    print("Max:", np.max(device_modulos_norm, axis=0))
    print("Mean:", np.mean(device_modulos_norm, axis=0))
    
    # Aplicar K-means 
    centroides, labels, indice_outliers = kmeans(device_modulos_norm, n_clusters)

    # Paleta de cores fortes
    cores_base = list(mcolors.TABLEAU_COLORS.values())
    if n_clusters > len(cores_base):
        cores_base = list(mcolors.CSS4_COLORS.values())[:n_clusters]

    # ======== PLOT 3D DOS CLUSTERS ========
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = device_modulos_norm[cluster_mask]
        cor_base = cores_base[i % len(cores_base)]
        cor_clara = clarear_cor(cor_base, fator=2.0)

        # Pontos normais
        normais_idx = np.setdiff1d(np.where(cluster_mask)[0], indice_outliers)
        if len(normais_idx) > 0:
            ax.scatter(device_modulos_norm[normais_idx, 0],
                       device_modulos_norm[normais_idx, 1],
                       device_modulos_norm[normais_idx, 2],
                       c = [cor_clara], s = 2, alpha = 0.2, marker = 'o',
                       label = f'Cluster {i+1}')

        # Outliers
        outliers_cluster = np.intersect1d(np.where(cluster_mask)[0], indice_outliers)
        if len(outliers_cluster) > 0:
            ax.scatter(device_modulos_norm[outliers_cluster, 0],
                       device_modulos_norm[outliers_cluster, 1],
                       device_modulos_norm[outliers_cluster, 2],
                       c = [cor_base], s = 10, alpha = 0.7, marker = 'P',
                       label = f'Outliers C{i+1}')

    # Centróides
    #ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c = 'black', s = 90, marker = 'P', label = 'Centroides')

    ax.set_title(f"Device {device_id} (k = {n_clusters})")
    ax.set_xlabel('Módulo ACC')
    ax.set_ylabel('Módulo GYR')
    ax.set_zlabel('Módulo MAG')
    ax.legend(loc='upper right')
    plt.show()

    # ======== PLOT 2D POR ATIVIDADE E DOS 3 MÓDULOS NO DEVICE ESCOLHIDO ========
    atividades = device_data[:, 11].astype(int)
    mod_acc = device_data[:, 13].astype(float)
    mod_gyr = device_data[:, 14].astype(float)
    mod_mag = device_data[:, 15].astype(float)

    mods = [mod_acc, mod_gyr, mod_mag]
    nomes = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']

    labels_outliers = np.zeros(len(device_data), dtype = bool)
    labels_outliers[indice_outliers] = True

    fig, axes = plt.subplots(1, 3, figsize = (15, 5), sharey = False)

    for i in range(3):
        ax = axes[i]
        ax.grid(True, linestyle = '--', alpha = 0.6)
        # Não outliers
        ax.scatter(atividades[~labels_outliers], mods[i][~labels_outliers], color = 'blue', s = 10, alpha = 0.6, label = 'Normal')
        # Outliers
        ax.scatter(atividades[labels_outliers], mods[i][labels_outliers], color = 'red', s = 10, alpha = 0.7, label = 'Outlier')
        ax.set_xlabel('Atividade')
        plt.xticks(np.unique(dados[:, 11].astype(int)))
        ax.set_ylabel('Módulo ' + nomes[i])
        ax.set_title(nomes[i])

    axes[0].legend()
    fig.suptitle(f'Outliers por Atividade com o Kmeans — Device {device_id}', fontsize = 13)
    plt.tight_layout()
    plt.show()

#kmeans_outliers_3D_apenasPorDevice(device_id = 2, n_clusters = 45)

'''
kmeans_outliers_3D_apenasPorDevice(1, 6)
kmeans_outliers_3D_apenasPorDevice(3, 6)
kmeans_outliers_3D_apenasPorDevice(4, 6)
kmeans_outliers_3D_apenasPorDevice(5, 6)
'''

def heatmap_normalidade_por_atividade(device_id):
    # Filtrar dados do device
    device_data = dados[dados[:, 0] == device_id]
    
    # Extrair colunas relevantes
    atividades = device_data[:, 11].astype(int)
    mod_acc = device_data[:, 13].astype(float)
    mod_gyr = device_data[:, 14].astype(float)
    mod_mag = device_data[:, 15].astype(float)

    # Lista de módulos e nomes
    mods = [mod_acc, mod_gyr, mod_mag]
    nomes = ['ACC', 'GYR', 'MAG']
    atividades_unicas = np.unique(atividades)

    # Matriz de p-values [módulo, atividade]
    p_matrix = np.zeros((3, len(atividades_unicas)))

    for i, mod in enumerate(mods):
        for j, act in enumerate(atividades_unicas):
            dados_atividade = mod[atividades == act]
            # Normalizar com z-score antes do teste
            dados_norm = zscore(dados_atividade)
            stat, p_value = kstest(dados_norm, 'norm')
            p_matrix[i, j] = p_value

    # Criar heatmap
    plt.figure(figsize = (12, 4))
    sns.heatmap(
        p_matrix, annot = True, fmt = ".3f",
        xticklabels = atividades_unicas, yticklabels = nomes,
        cmap = "coolwarm", cbar_kws={'label': 'p-value (Kolmogorov-Smirnov)'},
        vmin = 0, vmax = 1,
        linewidths = 0.4, linecolor = 'black'
    )
    plt.axhline(1, color = 'black', linewidth = 0.5)
    plt.title(f'Normalidade (KS-Test) — Device {device_id}')
    plt.xlabel('Atividade')
    plt.ylabel('Módulo')
    plt.tight_layout()
    plt.show()

#heatmap_normalidade_por_atividade(device_id = 2)

# ------------ 4.1 ------------ #
def testar_significancia_kruskal(device_id):
    # Filtrar apenas os dados do device
    device_data = dados[dados[:, 0] == device_id]

    # Extrair colunas relevantes
    atividades = device_data[:, 11].astype(int)
    mod_acc = device_data[:, 13].astype(float)
    mod_gyr = device_data[:, 14].astype(float)
    mod_mag = device_data[:, 15].astype(float)

    # Nomes e módulos
    mods = [mod_acc, mod_gyr, mod_mag]
    nomes = ['ACC', 'GYR', 'MAG']
    atividades_unicas = np.unique(atividades)

    # Guardar p-values
    p_vals = []

    print(f"\nTeste de Kruskal-Wallis para aferir se as médias das atividade nos módulos são estatisticamente significantes — Device {device_id}")
    print("---------------------------------------------------")

    for nome, mod in zip(nomes, mods):
        # Criar lista com os valores do módulo por atividade
        grupos = [mod[atividades == act] for act in atividades_unicas]

        # Aplicar o teste de Kruskal–Wallis
        stat, p_value = kruskal(*grupos)
        p_vals.append(p_value)

        # Imprimir resultado e interpretação
        interpretacao = "Diferenças significativas" if p_value < 0.05 else "Sem diferenças significativas"
        print(f"{nome}: H = {stat:.3f}, p = {p_value:.5f} → {interpretacao}")

#testar_significancia_kruskal(device_id = 1)

def heatmaps_modules_nonparam(dev_id):
    cols = [13, 14, 15]
    module_names = ["Acceleration", "Gyroscope", "Magnetometer"]

    # Sempre todas as 16 atividades
    atividades_unicas = np.arange(1,17)
    n_activities = len(atividades_unicas)

    for col, mod_name in zip(cols, module_names):
        activities_groups = [dados[(dados[:,11] == act) & (dados[:,0] == dev_id), col] for act in atividades_unicas]

        p_values = np.full((n_activities, n_activities), np.nan)

        for i in range(n_activities):
            for j in range(i+1, n_activities):
                if len(activities_groups[i]) > 0 and len(activities_groups[j]) > 0:
                    p = ranksums(activities_groups[i], activities_groups[j]).pvalue
                    p_values[i,j] = p
                    p_values[j,i] = p  # simetria

        df = pd.DataFrame(p_values, index = atividades_unicas.astype(int), columns = atividades_unicas.astype(int))

        plt.figure(figsize = (12,10))
        ax = sns.heatmap(df, annot = True, fmt = ".3f", linewidths = 0.5,
                         cmap = "coolwarm", vmin = 0, vmax = 1, cbar_kws = {'label': 'p-value'})

        # Adicionar contorno às células com p > 0.05
        for i in range(n_activities):
            for j in range(n_activities):
                if not np.isnan(p_values[i,j]) and p_values[i,j] > 0.05:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill = False, edgecolor = 'black', lw = 1))

        plt.title(f"Non-parametric p-values - {mod_name}")
        plt.show()

#heatmaps_modules_nonparam(dev_id = 2)


def extract_features_110(dataset, fs, window_s, overlap):
    """
    Extrai exatamente as 110 features (87 temporais/espectrais + 23 físicas)
    segundo o paper BodyNets 2011 (Zhang & Sawchuk).
    """
    win_len = int(window_s * fs)
    hop = int(win_len * (1 - overlap))
    n_samples = dataset.shape[0]

    print("Comprimento da janela:", win_len)

    # Colunas do dataset
    acc = dataset[:, 1:4]   # aceleração x,y,z
    gyr = dataset[:, 4:7]   # giroscópio x,y,z
    labels = dataset[:, 11] # atividade
    timestamp = dataset[:, 10] # timestamp

    X, y = [], []
    feature_names = []

    # -- NOMES DAS FEATURES -- #
    base_feats = [
        "mean", "median", "std", "var", "rms", "mean_diff", "skew", "kurtosis",
        "iqr", "zero_cross_rate", "mean_cross_rate", "spec_entropy",
        "dom_freq", "spec_energy"
    ]
    axes = ["x", "y", "z"]
    sensors = ["acc", "gyr"]

    # Criar nomes automáticos das primeiras 84 features (14*3*2)
    for s in sensors:
        for ax in axes:
            for f in base_feats:
                feature_names.append(f"{s}_{ax}_{f}")

    # Correlações 
    corr_pairs = list(combinations(["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"], 2))
    feature_names.extend([f"corr_{a}_{b}" for (a, b) in corr_pairs])

    # Features físicas 
    feature_names.extend([
        "AI_mean", "VI_var", "SMA",
        "EVA1", "EVA2",
        "AVG", "AVH", "ARATG",
        "CAGH", "ARE", "AAE"
    ])

    # Funções auxiliares
    def rms(x): return np.sqrt(np.mean(x**2))
    def iqr(x): return np.percentile(x, 75) - np.percentile(x, 25)
    def zero_crossing_rate(x): return np.sum(np.diff(np.sign(x)) != 0) / len(x)
    def mean_crossing_rate(x): return np.sum(np.diff(np.sign(x - np.mean(x))) != 0) / len(x)
    def spectral_entropy(x, fs):
        f, Pxx = welch(x, fs=fs)
        Pxx /= np.sum(Pxx)
        return entropy(Pxx)
    def dominant_frequency(x, fs):
        f, Pxx = welch(x, fs=fs)
        return f[np.argmax(Pxx)]
    def spectral_energy(x, fs):
        f, Pxx = welch(x, fs=fs)
        return np.sum(Pxx)
    def movement_intensity(acc):  # MI(t)
        return np.sqrt(np.sum(acc**2, axis=1))
    def sma(data):  # Signal Magnitude Area
        return np.sum(np.abs(data)) / len(data)
    def avg_velocity(acc, fs):  # AVG
        vel = np.cumsum(acc, axis=0) / fs
        return np.mean(np.linalg.norm(vel, axis=1))
    def eig_features(data):  # EVA (2 autovalores)
        eigvals = np.linalg.eigvals(np.cov(data.T))
        eigvals = np.real(np.sort(eigvals)[::-1])
        return eigvals[:2]

    # Sliding windows
    for start in range(0, n_samples - win_len + 1, hop):
        end = start + win_len
        lbls = labels[start:end]
        timestamp_acc = timestamp[start:end]

        if len(np.unique(lbls)) != 1:
            continue  # descartar janelas mistas

        if len(lbls)<10:
            continue  # descartar janelas muito pequenas

        acc_w = acc[start:end, :]
        gyr_w = gyr[start:end, :]

        feats = []

        # Armazenar dados de aceleração para atividades 1 a 7
        dados_acc = []
        if lbls[0] <= 7:
            dados_acc = np.hstack((acc_w, timestamp_acc.reshape(-1,1), lbls.reshape(-1,1)))

        # --- Features temporais + espectrais (87) ---
        for sensor in [acc_w, gyr_w]:
            for i in range(3):  # x, y, z
                x = sensor[:, i]
                feats.extend([
                    np.mean(x), np.median(x), np.std(x), np.var(x),
                    rms(x), np.mean(np.diff(x)), skew(x), kurtosis(x),
                    iqr(x), zero_crossing_rate(x), mean_crossing_rate(x),
                    spectral_entropy(x, fs), dominant_frequency(x, fs),
                    spectral_energy(x, fs)
                ])
        # 14 * 3 * 2 = 84 features

        # Correlações (15)
        full = np.hstack((acc_w, gyr_w))
        for (i, j) in combinations(range(6), 2):
            feats.append(np.corrcoef(full[:, i], full[:, j])[0, 1])
        # Total até aqui = 84 + 15 = 99

        # --- Physical features (11) ---
        mi = movement_intensity(acc_w)
        feats.append(np.mean(mi))      # 1. AI
        feats.append(np.var(mi))       # 2. VI
        feats.append(sma(acc_w))       # 3. SMA
        feats.extend(eig_features(acc_w)) # 4–5. EVA (2 autovalores)
        feats.append(avg_velocity(acc_w, fs)) # 6. AVG
        feats.append(np.mean(np.abs(gyr_w)))  # 7. AVH
        feats.append(np.var(gyr_w))           # 8. ARATG

        # CAGH (correlação aceleração-gravidade-heading)
        g_proj = acc_w[:, 2]  # z ~ gravidade
        heading_proj = np.sqrt(acc_w[:, 0]**2 + acc_w[:, 1]**2)
        feats.append(np.corrcoef(g_proj, heading_proj)[0, 1])  # 9. CAGH

        # ARE e AAE (energias médias)
        feats.append(np.mean(acc_w**2))  # 10. ARE
        feats.append(np.mean(gyr_w**2))  # 11. AAE

        # Total: 99 + 11 = 110 

        X.append(feats)
        y.append(lbls[0])

    return np.array(X), np.array(y), feature_names, dados_acc

# Frequência de amostragem
fs = 51.2
X_total, y_total = [], []
X_total_partB, y_total_partB = [], []
dados_acc_total = []

# Extração das 110 features para cada participante
for id_part in np.unique(dados[:, 12]):  # Todos os participantes
    dados_p = dados[dados[:, 12] == id_part]
    X_p, y_p, feature_names, dados_acc = extract_features_110(dados_p, fs, window_s = 5, overlap = 0.5)
    print(f"Matriz de features do participante {id_part}:", X_p.shape)
    print("Labels:", y_p.shape)
    print("features: ", len(feature_names))
    X_total.append(X_p)
    y_total.append(y_p)

    # Apenas atividades de 1 a 7 (Parte B)
    indices_atividade = [1, 2, 3, 4, 5, 6, 7]
    indices = np.where(np.isin(y_p, indices_atividade))[0]
    X_total_partB.append(X_p[indices])
    y_total_partB.append(y_p[indices])

    dados_acc_total.append(dados_acc)

X_total = np.vstack(X_total)[:, :-1]  # Excluir a última coluna (ID participante) para o PCA, relief e fisher
y_total = np.hstack(y_total)
print("Matriz de features Final:", X_total.shape)
print("Labels Finais:", y_total.shape)

# Matrizes para a Parte B (participantes 1 a 7)
X_total_partB = np.vstack(X_total_partB)
y_total_partB = np.hstack(y_total_partB)

# criar um csv com timestamp, acc_x, acc_y, acc_z, timestamp, activity
dados_acc_total = np.vstack(dados_acc_total)
np.savetxt("dados_acc.csv", dados_acc_total, delimiter = ",")

## ------------ EXERCÍCIO 4.3 ------------ ##
## -- PCA PARA REDUZIR A DIMENSIONALIDADE -- ##
def aplicar_pca(feature_set, n_components = None):

    # Normalizar as features (Z-score)
    scaler = StandardScaler()
    feature_set_norm = scaler.fit_transform(feature_set)

    # Aplicar PCA
    pca = PCA(n_components = n_components)
    pca_features = pca.fit_transform(feature_set_norm)

    # Mostrar resultados
    print("\n--- RESULTADOS DO PCA ---")
    print("Variância explicada por cada componente:")
    print(pca.explained_variance_ratio_)

    # Gráfico da variância explicada acumulada
    plt.figure(figsize = (7,4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_)*100, marker = 'o')

    # Linha horizontal nos 75%
    plt.axhline(y = 75, color = 'red', linestyle = '--', linewidth = 2)
    
    # Texto acima da linha
    plt.text(
        x = 37,# Posição x do texto 
        y = 75 + 2, # Um pouco acima da linha
        s = "75 %", 
        color = "red",
        fontsize = 12,
        fontweight = 'bold'
    )

    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('Análise de Componentes Principais (PCA)', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.grid(True)
    plt.xlim (0,40) # x entre [0,40]
    plt.show()

    return pca_features, pca, scaler

## ------------ EXERCÍCIO 4.4 ------------ ##
# Aplicar PCA (mantendo todas as componentes)
pca_features, pca, scaler = aplicar_pca(X_total)

# Ver quantas componentes são necessárias para explicar 75% da variância
variancia_acumulada = np.cumsum(pca.explained_variance_ratio_)
dimensoes_75 = np.argmax(variancia_acumulada >= 0.75) + 1
print(f"Número mínimo de componentes para explicar 75% da variância: {dimensoes_75}")

## ------------ EXERCÍCIO 4.4.1 ------------ ##
def exemplo_pca_instante(feature_set, scaler, pca, idx_exemplo = 0):
    """
    Obtém o vetor PCA completo de um instante específico (todas as componentes).
    Retorna:
        -> Vetor projetado no espaço PCA (todas as componentes)
    """

    # Obter o vetor original da amostra escolhida
    x_original = feature_set[idx_exemplo, :].reshape(1, -1)

    # Normalizar
    x_norm = scaler.transform(x_original)

    # Projetar no espaço PCA completo
    pca_features = x_norm @ pca.components_.T

    # Mostrar resultados
    print(f"\n--- Exemplo PCA Completo (instante {idx_exemplo}) ---")
    print(f"Vetor PCA (todas as componentes, dimensão {pca_features.shape[1]}):")
    print(pca_features)

    return pca_features

exemplo_pca_instante(pca_features, scaler, pca, idx_exemplo = 0)

## ------------ EXERCÍCIO 4.5 ------------ ##
def selecionar_features_fisher_reliefF(X, y, feature_names, k = 10):
    """
    Aplica Fisher Score e ReliefF para selecionar as 10 melhores features.
    """

    # Normalizar os dados
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # --- Fisher Score --- #
    F_values, _ = f_classif(X_norm, y)
    idx_fisher = np.argsort(F_values)[::-1][:k]

    # --- ReliefF --- #
    fs = ReliefF(n_neighbors = 10, n_features_to_select = k, n_jobs = -1) # Paralelizado
    fs.fit(X_norm, y)
    relief_scores = fs.feature_importances_
    idx_relief = np.argsort(relief_scores)[::-1][:k]

    # ---- Mostrar resultados ----
    print("\nTop Features segundo Fisher Score:")
    for i, idx in enumerate(idx_fisher):
        print(f"  {i+1:02d}. {feature_names[idx]} — Score = {F_values[idx]:.4f}")

    print("\nTop Features segundo ReliefF:")
    for i, idx in enumerate(idx_relief):
        print(f"  {i+1:02d}. {feature_names[idx]} — Score = {relief_scores[idx]:.4f}")

    # Retorna as 3 melhores features
    top3_fish = idx_fisher[:3]
    top3_rel = idx_relief[:3]
    # Retorna as 10 melhores features
    top10_fish = idx_fisher[:10]
    top10_rel = idx_relief[:10]

    return top3_fish, top3_rel, top10_fish, top10_rel

## ------------ EXERCÍCIO 4.6 ------------ ##
top3_fish, top3_rel, top10_fish, top10_rel = selecionar_features_fisher_reliefF(X_total, y_total, feature_names, k = 10)

## ------------ EXERCÍCIO 4.6.1 ------------ ##
def obter_features_selecionadas_numInstante(feature_set, indices_melhores, instante):
    # Seleção das colunas correspondentes às 10 melhores features
    vetor_reduzido = feature_set[instante, indices_melhores]
    return vetor_reduzido

vetor_fisher = obter_features_selecionadas_numInstante(X_total, top10_fish, instante = 1)
vetor_relief = obter_features_selecionadas_numInstante(X_total, top10_rel, instante = 1)


def plot_fisher_relief_features_3D(X, y, fisher_idx, relief_idx):
    """
    Cria dois gráficos 3D com as três melhores features
    segundo Fisher Score e ReliefF, cada atividade com uma cor diferente.
    """

    fig = plt.figure(figsize = (16, 7))

    # -- Gráfico Fisher 3D -- #
    ax1 = fig.add_subplot(1, 2, 1, projection = '3d')
    sc1 = ax1.scatter(X[:, fisher_idx[0]], X[:, fisher_idx[1]], X[:, fisher_idx[2]],
                      c = y, cmap = 'tab20', s = 20, alpha = 0.8, edgecolors = 'none')
    ax1.set_title(f'Fisher Score — Features {fisher_idx}', fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    ax1.set_xlabel(f'Feature {fisher_idx[0]}')
    ax1.set_ylabel(f'Feature {fisher_idx[1]}')
    ax1.set_zlabel(f'Feature {fisher_idx[2]}')

    # -- Gráfico ReliefF 3D -- #
    ax2 = fig.add_subplot(1, 2, 2, projection = '3d')
    sc2 = ax2.scatter(X[:, relief_idx[0]], X[:, relief_idx[1]], X[:, relief_idx[2]],
                      c = y, cmap = 'tab20', s = 20, alpha = 0.8, edgecolors = 'none')
    ax2.set_title(f'ReliefF — Features {relief_idx}', fontname = 'Trebuchet MS', fontsize = 16, fontweight = 'bold')
    ax2.set_xlabel(f'Feature {relief_idx[0]}')
    ax2.set_ylabel(f'Feature {relief_idx[1]}')
    ax2.set_zlabel(f'Feature {relief_idx[2]}')

    # Barra de cores
    cbar = fig.colorbar(sc2, ax = [ax1, ax2], fraction = 0.02, pad = 0.04)
    cbar.set_label('Atividade', rotation = 270, labelpad = 15)

    plt.suptitle('Comparação das Features Selecionadas em 3D (Fisher vs ReliefF)', fontsize = 22, fontweight = 'bold', fontname = 'Trebuchet MS', color = '#c00000')
    plt.show()

plot_fisher_relief_features_3D(X_total, y_total, top3_fish, top3_rel)