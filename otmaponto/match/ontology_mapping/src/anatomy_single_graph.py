import json
import os
import pprint
import pandas as pd
import numpy as np
import random

from xml.dom import minidom
from rdflib import Graph


def load_graph(_name):
    ds_name = _name.split(".")[0]
    _wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dd = os.path.join(_wd, "data", "anatomy-dataset")
    _g = Graph()
    _g.parse(os.path.join(dd, _name))
    _md = os.path.join(_wd, "src", "binaries", "{}_concept_ft.mod".format(ds_name))
    return _g, _md, _wd


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


def build_edge_graph(_graph, _fname):
    subclass_query = '''SELECT ?h ?t
                        WHERE {
                          ?h rdfs:subClassOf ?t .
                          FILTER(!isBlank(?t)) 
                        }
                        ORDER BY ?h
                        '''
    _sub_res = _graph.query(subclass_query)
    _heads = []
    _tails = []
    for res in _sub_res:
        _heads.append(res[0].toPython())
        _tails.append(res[1].toPython())
    _rels = ['subclassOf'] * len(_heads)
    _df = pd.DataFrame({'sub': _heads,
                        'rel': _rels,
                        'obj': _tails})

    partof_query = '''SELECT ?h ?r
                            WHERE {
                              ?h rdfs:subClassOf ?t .
                              ?t owl:someValuesFrom ?r .
                              FILTER(!isBlank(?h)) 
                            }
                            ORDER BY ?h
                            '''
    _part_res = _graph.query(partof_query)
    _p_heads = []
    _p_tails = []
    for res in _part_res:
        _p_heads.append(res[0].toPython())
        _p_tails.append(res[1].toPython())
    _p_rels = ['partOf'] * len(_p_heads)
    _df2 = pd.DataFrame({'sub': _p_heads,
                         'rel': _p_rels,
                         'obj': _p_tails})
    if "reasoned" in _fname:
        reasoner_query = '''SELECT ?h ?t
                                WHERE {
                                  ?h owl:disjointWith ?t .
                                  FILTER(!isBlank(?t)) 
                                }
                                ORDER BY ?h
                                '''
        _reas_res = _graph.query(reasoner_query)
        _p_heads = []
        _p_tails = []
        for res in _reas_res:
            _p_heads.append(res[0].toPython())
            _p_tails.append(res[1].toPython())
        _p_rels = ['disjointWith'] * len(_p_heads)
        print('Added {} disjointWith triples for {}'.format(len(_p_rels), _fname))
        _df3 = pd.DataFrame({'sub': _p_heads,
                             'rel': _p_rels,
                             'obj': _p_tails})
        _triples = pd.concat([_df, _df2, _df3], ignore_index=True)
    else:
        _triples = pd.concat([_df, _df2], ignore_index=True)
    return _triples


def create_triple_indices(_triples):
    _all_subs = list(set(_triples['sub'].tolist()))
    _all_obs = list(set(_triples['obj'].tolist()))
    _entities = list(set(_all_subs + _all_obs))
    _entities.sort()
    _relations = list(set(_triples['rel']))
    _ent_idx = {}
    _rel_idx = {}
    for ent in range(len(_entities)):
        _ent_idx[_entities[ent]] = ent
    for rel in range(len(_relations)):
        _rel_idx[_relations[rel]] = rel
    print(_rel_idx)
    _triples['head_idx'] = _triples['sub'].apply(lambda x: _ent_idx[x])
    _triples['rel_idx'] = _triples['rel'].apply(lambda x: _rel_idx[x])
    _triples['tail_idx'] = _triples['obj'].apply(lambda x: _ent_idx[x])
    _all_trips = _triples[['head_idx', 'rel_idx', 'tail_idx']]
    return _ent_idx, _rel_idx, _all_trips


def load_node_features(_fname, _wd, _ent_idx):
    with open(os.path.join(_wd, 'src', 'output', '{}_pretrained_vectors.json'.format(_fname)), mode='rb') as g:
        vectors = json.load(g)
    with open(os.path.join(_wd, 'src', 'output', '{}_pretrained_uri_keys.json'.format(_fname)), mode='rb') as g:
        map = json.load(g)
    _node_features = {}
    for ent in list(_ent_idx.keys()):
        try:
            id = _ent_idx[ent]
            feat = vectors[map[ent]]
            _node_features[id] = str(feat)[1:-1]
        except KeyError:
            id = _ent_idx[ent]
            print('Error on {}'.format(ent))
            _node_features[id] = str([0]*300)
    with open(os.path.join(_wd, 'src', 'output', 'gcn', 'anatomy_single', _fname, 'raw', '{}_feature_label.txt'.format(_fname)), mode='w') as g:
        for k in _node_features.keys():
            g.write('{}\t{}\t1\n'.format(k, _node_features[k]))


def rand_bin_array(k, n):
    arr = np.zeros(n)
    arr[:k] = -1
    np.random.shuffle(arr)
    return arr


def mask_nodes(_fname, _ent_idx, _sid):
    _wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    _n = len(_ent_idx.keys())
    print('Generating {} masks'.format(_n))
    _train_k = int(.6 * _n)
    _val_k = int(.2 * _n)
    _test_k = int(.2 * _n)
    _train_labs = rand_bin_array(_train_k, _n)
    _val_labs = rand_bin_array(_val_k, _n)
    _test_labs = rand_bin_array(_test_k, _n)
    _all_labs = {}
    _all_labs['train_mask'] = _train_labs
    _all_labs['val_mask'] = _val_labs
    _all_labs['test_mask'] = _test_labs
    _file_out = os.path.join(_wd, 'src', 'output', 'gcn',
                             'anatomy_single', _fname, 'raw', '{}_split_0.6_0.2_{}.npz'.format(_fname, _sid))
    np.savez(_file_out, **_all_labs)


def run(_fname):
    g1, md1, wd = load_graph('{}.owl'.format(_fname))
    mape, labs = build_label_graph(g1)
    _edges = build_edge_graph(g1, _fname)
    ent_idx, rel_idx, all_trips = create_triple_indices(_edges)
    print('{} has {} concepts'.format(_fname, len(ent_idx.keys())))
    with open(os.path.join(wd, 'src', 'output', 'gcn', 'anatomy_single', _fname,
                           'raw', '{}_entity_idx.json'.format(_fname)), mode='wb') as g:
        g.write(json.dumps(ent_idx).encode('utf-8'))
    #with open(os.path.join(wd, 'src', 'output', 'gcn', 'anatomy_single',
    #                       'raw', '{}_relation_idx.json'.format(_fname)), mode='wb') as g:
    #    g.write(json.dumps(rel_idx).encode('utf-8'))
    #with open(os.path.join(wd, 'src', 'output', 'gcn', 'anatomy_single', 'raw', 'ent_ids_{}'.format(_fname)), mode='w') as g:
    #    for k in ent_idx.keys():
    #        g.write('{}\t{}'.format(ent_idx[k], k))
    #with open(os.path.join(wd, 'src', 'output', 'gcn', 'anatomy_single', 'raw', 'rel_ids_{}'.format(_fname)), mode='w') as g:
    #    for k in rel_idx.keys():
    #        g.write('{}\t{}'.format(rel_idx[k], k))
    with open(os.path.join(wd, 'src', 'output', 'gcn', 'anatomy_single', _fname, 'raw', '{}_graph_edges.txt'.format(_fname)), mode='w') as g:
        for idx, vals in all_trips.iterrows():
            g.write("{}\t{}\n".format(vals['head_idx'], vals['tail_idx']))
    load_node_features(_fname, wd, ent_idx)
    return ent_idx


if __name__ == "__main__":
    num_splits = 10
    human_idx = run("human")
    for j in range(num_splits):
        mask_nodes("human", human_idx, j)
    mouse_idx = run("mouse")
    for j in range(num_splits):
        mask_nodes("mouse", mouse_idx, j)

    human_idx = run("human-reasoned")
    for j in range(num_splits):
        mask_nodes("human-reasoned", human_idx, j)
    mouse_idx = run("mouse-reasoned")
    for j in range(num_splits):
        mask_nodes("mouse-reasoned", mouse_idx, j)

