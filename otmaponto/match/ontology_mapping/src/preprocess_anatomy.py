import json
import os
import pprint

from rdflib import Graph
from gensim.models.fasttext import FastText, load_facebook_model

from corpus_build_utils import clean_document_lower
from train_ft import train_ft_custom_model


def load_graph(_name):
    ds_name = _name.split(".")[0]
    _wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dd = os.path.join(_wd, "data", "anatomy-dataset")
    _g = Graph()
    _g.parse(os.path.join(dd, _name))
    _md = os.path.join(_wd, "src", "binaries", "{}_concept_ft.mod".format(ds_name))
    return _g, _md, _wd


def build_concept_graph(_graph):
    def_query = '''SELECT ?a ?b
                   WHERE {?a oboInOwl:hasDefinition ?b}
                   ORDER BY ?a'''
    _q_res = _graph.query(def_query)
    _label_list = []
    _map = {}
    for res in _q_res:
        print(res[0].toPython())
        _map[res[0].toPython()] = res[1].value
        _label_list.append(res[1].value)
    return _map, _label_list


def build_label_graph(_graph):
    def_query = '''SELECT ?a ?b
                   WHERE {?a rdfs:label ?b .
                   FILTER(!isBlank(?a))}
                   ORDER BY ?a'''
    _q_res = _graph.query(def_query)
    _label_list = []
    _map = {}
    for res in _q_res:
        u = res[0].toPython()
        if 'NCI' in u:
            _map[u] = res[1].value
            _label_list.append(res[1].value)
        elif 'MA' in u:
            _map[u] = res[1].value
            _label_list.append(res[1].value)
    return _map, _label_list


def generate_embeddings(_fname, _mt):
    g1, md1, wd = load_graph('{}.owl'.format(_fname))
    _map, _labels = build_label_graph(g1)
    _cleaned = [clean_document_lower(_z) for _z in _labels]
    if _fname == 'mouse':
        _labels.append('http://www.w3.org/2002/07/owl#Thing')
        _cleaned.append('http://www.w3.org/2002/07/owl#Thing')
    print(len(_cleaned))
    if _mt == 'custom':
        _mod, _meta = train_ft_custom_model(_cleaned, md1, '{}_concept_ft'.format(_fname), 3, 6, 3, 1, 100, 5)
        _vectors = {}
        for concept in _labels:
            _vectors[concept] = _mod.wv[concept].tolist()
    elif _mt == 'pretrained':
        fname = os.path.join(wd, 'src', 'binaries', 'crawl-300d-2M-subword.bin')
        _mod = load_facebook_model(fname)
        _vectors = {}
        for concept in _labels:
            _vectors[concept] = _mod.wv[concept].tolist()

    with open(os.path.join(wd, 'src', 'output', '{}_{}_vectors.json'.format(_fname, _mt)), mode='wb') as g:
        g.write(json.dumps(_vectors).encode('utf-8'))
    with open(os.path.join(wd, 'src', 'output', '{}_{}_uri_keys.json'.format(_fname, _mt)), mode='wb') as g:
        g.write(json.dumps(_map).encode('utf-8'))


if __name__ == "__main__":
    generate_embeddings('human', 'pretrained')
    generate_embeddings('mouse', 'pretrained')
    generate_embeddings('human-reasoned', 'pretrained')
    generate_embeddings('mouse-reasoned', 'pretrained')

