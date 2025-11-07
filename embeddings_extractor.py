import torch
import numpy as np


def load_model():
  ''' Loads the model from the github repo and obtains just the feature encoder. '''

  # Nome do repositório GitHub onde o modelo pré-treinado está guardado
  repo = 'OxWearables/ssl-wearables'

  # class_num não interessa para extrair features; mas o hub pede este arg
  model = torch.hub.load(repo, 'harnet5', class_num=5, pretrained=True)
  model.eval()

  # Passo crucial: ficar só com a parte auto-supervisionada
  # O README diz que há um 'feature_extractor' (pré-treinado) e um 'classifier' (não treinado). :contentReference[oaicite:14]{index=14}
  # Aqui queremos apenas o feature_extractor, que gera os embeddings
  feature_encoder = model.feature_extractor
  feature_encoder.to("cpu")
  feature_encoder.eval()

  # Devolve o feature encoder, que será usado para extrair embeddings dos dados
  return feature_encoder


def resample_to_30hz_5s(acc_xyz, fs_in_hz):
    """
    acc_xyz: matriz (N, 3) -> aceleração (x, y, z)
    fs_in_hz: frequência original (Hz)
    devolve: sinal reamostrado a 30 Hz e a nova frequência
    """
    fs_target = 30.0      # nova frequência desejada (30 Hz)
    win_size = 5          # duração do segmento em segundos

    # tempos originais (com base na frequência inicial)
    t_in = np.arange(acc_xyz.shape[0]) / fs_in_hz

    # tempos novos, uniformes, de 0 a 5s com passo 1/30
    t_out = np.arange(0, win_size, 1.0/fs_target)

    # cria matriz vazia para o sinal reamostrado (150x3)
    acc_resampled = np.zeros((len(t_out), 3), dtype=np.float32)

    # interpola cada eixo (x, y, z) para os novos tempos
    for axis in range(3):
        acc_resampled[:, axis] = np.interp(t_out, t_in, acc_xyz[:, axis])

    # devolve o sinal reamostrado e a nova frequência
    return acc_resampled, fs_target


def acc_segmentation(data):
    '''Extrai segmentos de aceleração (ACC) e as respetivas atividades.'''

    TIMESTAMP_COL = 3          # coluna onde está o timestamp (no teu CSV)
    MIN_SEGMENT_SIZE = 20      # mínimo de amostras para aceitar um segmento
    fs_in_hz = 51.2            # frequência original de amostragem (Hz)
    win_size = 5               # duração da janela em milissegundos (5 segundos)

    # define o primeiro intervalo de tempo (início e fim)
    start_time = data[0, TIMESTAMP_COL]          # tempo inicial
    end_time = start_time + win_size             # tempo final (5 s depois)

    activities = []           # lista para guardar as atividades
    segments = []             # lista para guardar os segmentos de aceleração

    # percorre o ficheiro até chegar ao fim dos dados
    while end_time < data[-1, TIMESTAMP_COL]:

        # máscara que seleciona as linhas dentro do intervalo [start_time, end_time)
        mask = (data[:, TIMESTAMP_COL] >= start_time) & (data[:, TIMESTAMP_COL] < end_time)

        # verifica se há amostras suficientes e se toda a janela é da mesma atividade
        if np.sum(mask) > MIN_SEGMENT_SIZE and np.all(data[mask, -1] == data[mask, -1][0]):

            # extrai as colunas X, Y e Z da aceleração
            acc_xyz = data[mask, 1:4]

            # obtém o valor da atividade (última coluna)
            activity = data[mask, -1][0]

            # guarda o segmento e a sua atividade correspondente
            activities.append(activity)
            segments.append(acc_xyz)

        # move a janela 50% para a frente (overlap de 2.5 s)
        start_time = end_time - win_size/2
        end_time = start_time + win_size

    # devolve as listas de segmentos e atividades
    return segments, activities


# load example file to test embedding
csv_file_path = 'dados_acc.csv'
csv_data = np.loadtxt(csv_file_path, delimiter=',')
print(csv_data[:20, :])

original_segments, activities = acc_segmentation(csv_data)
resampled_segments = [resample_to_30hz_5s(segment, 51.2)[0] for segment in original_segments]

feature_encoder = load_model()

print("n_original_segments:", len(original_segments))
print("n_resampled_segments (antes de filtrar):", len(resampled_segments))
for i, seg in enumerate(resampled_segments[:10]):
    print(i, "shape:", None if seg is None else np.array(seg).shape, "dtype:", type(seg))


embeddings_list = []

# reshape segments to [n_segments, dimensions(xyz), time]
x_all = np.transpose( np.array(resampled_segments), (0, 2, 1) )
print(x_all.shape)

# iterate over the resampled segments and pass them 
#    through the model in batches to get the embeddings
batch_size = 5
with torch.no_grad():
    for i in range(0, x_all.shape[0], batch_size):
        xb = torch.from_numpy(x_all[i:i+batch_size]).float().to("cpu")
        eb = feature_encoder(xb)  # (B, D_embed)
        embeddings_list.append(eb.cpu().numpy())

embeddings = np.concatenate(embeddings_list, axis=0)
print(embeddings.shape)
