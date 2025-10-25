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

## EXERCÍCIO ...
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

heatmaps_modules_nonparam(dev_id = 2)