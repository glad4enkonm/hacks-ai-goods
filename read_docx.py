import pickle
from docx import Document
from pathlib import Path


def to_pkl(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def de_pkl(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def dump_to_pkl():
    docs = {}

    for p in Path('standards').glob('*.docx'):
        document = Document(p)

        pars = [[p.stem]]
        for n_par, par in enumerate(document.paragraphs):
            pars.extend(par.text.strip().split('. '))
        docs[p.stem] = pars

    to_pkl('data/docks.pkl', docs)


if __name__ == '__main__':
    # dump_to_pkl()

    docs_dict = de_pkl('data/docks.pkl')

    # pprint(docs_dict)

    for k,v in docs_dict.items():
        print(k, v[0:10])