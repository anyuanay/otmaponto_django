import argparse
import json
import os
import numpy as np
import torch
import pprint
import matplotlib.pyplot as plt

from xml.dom import minidom
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

from models.MapOT import GWTransport


def load_gold_labels(_fn, _src_idx, _trg_idx, _src_uri, _trg_uri):
    _wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dd = os.path.join(_wd, "data", "anatomy-dataset")
    xml_data = minidom.parse(os.path.join(dd, _fn))
    mappers = xml_data.getElementsByTagName('map')
    _gold = {}
    for ele in mappers:
        e1 = ele.getElementsByTagName('entity1')[0].attributes['rdf:resource'].value
        e2 = ele.getElementsByTagName('entity2')[0].attributes['rdf:resource'].value
        _gold[e1] = e2
    _gold_idx = {}
    for j in _gold.keys():
        _s = _src_uri[_gold[j]]
        _t = _trg_uri[j]
        # print('{} -- {}'.format(_s, _t))
        _gold_idx[_src_idx[_s]] = [_trg_idx[_t]]
    return _gold, _gold_idx


def load_vectors(_dd, _mt, _ft, _reas):
    _src_idx = {}
    _trg_idx = {}
    if _reas:
        p1 = 'human-reasoned_{}_vectors.json'.format(_mt)
        p2 = 'mouse-reasoned_{}_vectors.json'.format(_mt)
    else:
        p1 = 'human_{}_vectors.json'.format(_mt)
        p2 = 'mouse_{}_vectors.json'.format(_mt)
    with open(os.path.join(_dd, p1), 'rb') as v1:
        _human_vectors = json.load(v1)
    with open(os.path.join(_dd, p2), 'rb') as v2:
        _mouse_vectors = json.load(v2)
    if _ft == 'fasttext':
        _hv = torch.tensor(list(_human_vectors.values()))
        _mv = torch.tensor(list(_mouse_vectors.values()))
    elif _ft == 'gcn':
        if _reas:
            hp = os.path.join(_dd, 'gcn', 'anatomy_single', 'human-reasoned', 'human-reasoned_features.pt')
            mp = os.path.join(_dd, 'gcn', 'anatomy_single', 'mouse-reasoned', 'mouse-reasoned_features.pt')
        else:
            hp = os.path.join(_dd, 'gcn', 'anatomy_single', 'human', 'human_features.pt')
            mp = os.path.join(_dd, 'gcn', 'anatomy_single', 'mouse', 'mouse_features.pt')
        _hv = torch.load(hp)
        _mv = torch.load(mp)
    _src_keys = list(_human_vectors.keys())
    _trg_keys = list(_mouse_vectors.keys())
    for j in range(len(_src_keys)):
        _src_idx[_src_keys[j]] = j
    for j in range(len(_trg_keys)):
        _trg_idx[_trg_keys[j]] = j
    if _reas:
        p1 = 'human-reasoned_{}_uri_keys.json'.format(_mt)
        p2 = 'mouse-reasoned_{}_uri_keys.json'.format(_mt)
    else:
        p1 = 'human_{}_uri_keys.json'.format(_mt)
        p2 = 'mouse_{}_uri_keys.json'.format(_mt)
    with open(os.path.join(_dd, p1), 'rb') as v1:
        _src_uri = json.load(v1)
    with open(os.path.join(_dd, p2), 'rb') as v1:
        _trg_uri = json.load(v1)
    return _hv, _mv, _src_idx, _trg_idx, _src_uri, _trg_uri


def create_test_features(_dd, _gold):
    with open(os.path.join(_dd, 'human_vectors.json'), 'rb') as v1:
        _human_vectors = json.load(v1)
    with open(os.path.join(_dd, 'mouse_vectors.json'), 'rb') as v2:
        _mouse_vectors = json.load(v2)
    with open(os.path.join(_dd, 'human_uri_keys.json'), 'rb') as v1:
        _human_uris = json.load(v1)
    with open(os.path.join(_dd, 'mouse_uri_keys.json'), 'rb') as v2:
        _mouse_uris = json.load(v2)
    _human_test = []
    _mouse_test = []
    print('Testing on {} pairs'.format(len(list(_gold.keys()))))
    for j in list(_gold.keys()):
        _mouse_test.append(_mouse_vectors[_mouse_uris[j]])
        _human_test.append(_human_vectors[_human_uris[_gold[j]]])
    _human_test = torch.tensor(_human_test)
    _mouse_test = torch.tensor(_mouse_test)
    return _human_test, _mouse_test, _human_uris, _mouse_uris


def lookup_pairs(_pairs, _gold, _dd, _scores):
    """
    Gets the human readable labels for all output pairs.

    :param _pairs: List of tuples matched from each ontology.
    :param _gold: Gold standard dictionary.
    :return:
    """
    with open(os.path.join(_dd, 'human_vectors.json'), 'rb') as v1:
        _human_vectors = json.load(v1)
    with open(os.path.join(_dd, 'mouse_vectors.json'), 'rb') as v2:
        _mouse_vectors = json.load(v2)
    _hr_pairs = []
    gold_onto1 = list(_human_vectors.keys())
    gold_onto2 = list(_mouse_vectors.keys())
    print(len(gold_onto1))
    print(len(gold_onto2))
    for j in range(len(_pairs)):
        _hr_pairs.append((gold_onto1[_pairs[j][0]],
                          gold_onto2[_pairs[j][1]],
                          _scores[j]))
    return _hr_pairs


def run(_score_all):
    parser = argparse.ArgumentParser(prog="transport_anatomy",
                                     description="OT for anatomy mappings")
    parser.add_argument('-f', '--features', required=True, type=str,
                        help='The feature type', dest='feat')
    parser.add_argument('-r', '--reasoner', action='store_true',
                        help='Reasoner flag', dest='reason')
    args = parser.parse_args()
    _wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    _dd = os.path.join(_wd, 'src', 'output')
    hv, mv, src_idx, trg_idx, src_uri, trg_uri = load_vectors(_dd, 'pretrained', args.feat, args.reason)
    print('---- Feature shapes ----')
    print(hv.shape)
    print(mv.shape)
    labs, gid = load_gold_labels('reference.rdf', src_idx, trg_idx, src_uri, trg_uri)
    ht, mt, h_uri, m_uri = create_test_features(_dd, labs)
    print('---- Label shapes ----')
    print(ht.shape)
    print(mt.shape)
    mod = GWTransport(hv, mv, 'both', 1e-9, 'uniform', 1e-9, 'projected', src_idx, trg_idx, 'euclidean', .5, gid)
    mod.fit(hv, mv, verbose=True)
    mod.scores = mod.compute_scores(hv, mv, score_type='projected', test=False, adjust='csls', verbose=False)
    mod.score_translations(gid, verbose=True)

    if _score_all:
        acc_dict = {}
        print('Results on test dictionary for fitting vectors: (via coupling)')
        acc_dict['coupling'] = mod.test_accuracy(verbose=True, score_type='coupling')
        print('Results on test dictionary for fitting vectors: (via coupling + csls)')
        acc_dict['coupling_csls'] = mod.test_accuracy(verbose=True, score_type='coupling', adjust='csls')

        print('Results on test dictionary for fitting vectors: (via bary projection)')
        acc_dict['bary'] = mod.test_accuracy(verbose=True, score_type='barycentric')
        print('Results on test dictionary for fitting vectors: (via bary projection + csls)')
        acc_dict['bary_csls'] = mod.test_accuracy(verbose=True, score_type='barycentric', adjust='csls')

        print('Results on test dictionary for fitting vectors: (via orth projection)')
        acc_dict['proj'] = mod.test_accuracy(verbose=True, score_type='projected')
        print('Results on test dictionary for fitting vectors: (via orth projection + csls)')
        acc_dict['proj_csls'] = mod.test_accuracy(verbose=True, score_type='projected', adjust='csls')

        tr = [1.00] * len(sc)
        p, r, th = precision_recall_curve(tr, sc)
        fscore = (2 * p * r) / (p + r)
        ix = np.argmax(fscore)
        print('Estimated F1 score: {}'.format(fscore[ix]))
        average_precision = average_precision_score(tr, sc)
        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
        tr = [1] * len(sc)
        fpr, tpr, thresholds = roc_curve(tr, sc, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    run(False)
