from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from diskcache import Cache

from transformer import Transformer
from morph_analyser import Analyser

pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 1000)

cached = Cache('temp')

anlzr = Analyser()


# cached.clear()


def load_standarts_list(cache=False):
    """
    category -> procedure of testing
    {'изделия из пластмасс (ванночка, горшок туалетный и другие изделия для выполнения туалета) для ухода за детьми': ['МУ 1.1.037-95\xa0"Биотестирование продукции из полимерных и других материалов"'...]
    """

    standarts_list = pd.read_csv('data/standarts_list.csv')
    if cache and 'standarts_list' in cached:
        return cached['standarts_list']

    dd = defaultdict(list)
    for record in standarts_list.to_dict('records'):
        k, v = list(record.values())
        k, v = anlzr.to_keywords(k), anlzr.to_keywords(v)
        dd[v].append(k)

    if 'standarts_list' not in cached:
        cached['standarts_list'] = list(dd.items())

    return dd


def load_standarts(cache=False):
    """
    Файл «standarts.csv» содержит связи между продукцией и стандартами.

    """
    standarts = pd.read_csv('data/standarts.csv')

    if cache and 'standarts' in cached:
        return cached['standarts']

    dd = defaultdict(list)
    for record in standarts.to_dict('records'):
        k = anlzr.to_keywords(record.pop('Группа продукции'))
        dd[k].append(record)

    if 'standarts' not in cached:
        cached['standarts'] = list(dd.items())

    return dd


def calc_similarity(embs1, embs2, keys, top_n=1, threshold=0.58):
    similarity_matrix = cosine_similarity(embs1, embs2)

    top_k = top_n
    top_similar_sentences = []

    for i in range(len(embs1)):
        row_similarities = similarity_matrix[i]
        top_indices = np.argsort(row_similarities)[-top_k:][::-1]
        top_sentences = [keys[j] for j in top_indices]
        top_scores = row_similarities[top_indices]
        top_similar_sentences.append(list(zip(top_sentences, top_scores)))

    result = defaultdict(list)
    for i, similar_sentences in enumerate(top_similar_sentences):
        for sentence, score in similar_sentences:
            if score >= threshold:
                result[i + 1].append([sentence, score])
    return result


t = Transformer()


def load_dataset():
    df = pd.read_csv('data/dataset.csv')
    df = df[['id', 'Группа продукции', 'Наименование продукции']]

    df.columns = ['id', 'group', 'name']

    if 'df' not in cached:
        df['group'] = df['group'].apply(anlzr.to_keywords)
        df['name'] = df['name'].apply(anlzr.to_keywords)
        cached['df'] = df

    df: pd.DataFrame = cached['df']

    elements_n = 100
    return df.iloc[:elements_n, :]


def calc_similarity_column(df, name, dataset_keys, dataset_embs, source_embs, source_d):
    df[name] = ''
    result_ = calc_similarity(dataset_embs, source_embs, dataset_keys, threshold=0.70)
    for row, data in dict(result_).items():
        data = source_d[data[0][0]]
        if not data:
            continue
        df.iloc[row - 1, df.columns.get_loc(name)] = ' '.join(data)


def run():
    standarts_lst = load_standarts()
    standarts_lst_keys = list(standarts_lst.keys())
    standarts_lst_embs = t.calc_embeddings(tuple(standarts_lst_keys))

    standarts = load_standarts_list()
    print(standarts)
    standarts_keys = list(standarts.keys())
    standarts_embs = t.calc_embeddings(tuple(standarts_keys))

    df = load_dataset()

    df['group_embs'] = df['group'].apply(t.calc_embeddings)
    group_keys = df['group'].tolist()
    group_embs = df['group_embs'].tolist()

    df['name_embs'] = df['name'].apply(t.calc_embeddings)
    name_keys = df['name'].tolist()
    name_embs = df['name_embs'].tolist()

    calc_similarity_column(df, 'from_standarts', group_keys, group_embs, standarts_embs, standarts)
    calc_similarity_column(df, 'from_standarts_lst', group_keys, group_embs, standarts_lst_embs, standarts_lst)
    calc_similarity_column(df, 'from_name_st', name_keys, name_embs, standarts_embs, standarts)
    calc_similarity_column(df, 'from_name_st_lst', name_keys, name_embs, standarts_lst_embs, standarts_lst)

    print(df[['id', 'from_standarts', 'from_standarts_lst', 'from_name_st', 'from_name_st_lst']])

    # print(*list(load_standarts().keys()), sep='\n')
    # print(*list(load_standarts_list().keys()), sep='\n')


if __name__ == '__main__':
    run()
