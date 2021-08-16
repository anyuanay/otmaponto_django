# import libraries
import pandas as pd
import numpy as np
import scipy as sp
import json

from rdflib import Graph, URIRef, RDFS
from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import load_facebook_vectors

import corpus_build_utils
from corpus_build_utils import clean_document_lower
from AlignmentFormat import serialize_mapping_to_tmp_file

import sys
import logging
from collections import defaultdict

import jellyfish
import ot

from xml.dom import minidom
from nltk.corpus import wordnet

import importlib

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import OTMapOnto as maponto

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("logging info test")

## Interva Labeling and Retrieving Nodes between Mappings

# Create the hierarchical graph and computer interval labeling numbers
import networkx as nx


# retrieve the DFS preorders and initial maxpre of all leaves
def get_preorders(graph, source=None):
    """
        input: graph
        output: a DataFrame indexed by the graph nodes with columns ['preorder']
                where 'preorder' are nodes' DFS preorder number
    """
    pre_orders = []
    order = 1
    if source == None:
        pre = list(nx.dfs_preorder_nodes(graph))
    else:
        pre = list(nx.dfs_preorder_nodes(graph, source))
    for n in pre:
        pre_orders.append(order)
        order = order + 1
        
    return pd.DataFrame(pre_orders, columns=['preorder'], index=list(pre))


# given the current maxpres and a list of successors, find the largest maxpre
def update_maxpre(maxpre, successors):
    """
        input: mappre: a pandas Series with current maxpre; indexed by nodes
               successors: a list of nodes that are the DFS successors of the calling node
        output: newMaxpre: the largest maxpre among the successors
    """
    # initialize the maxpre with an impossible small value
    max = -1
    for i in range(len(successors)):
        if maxpre[successors[i]] > max:
            max = maxpre[successors[i]]
    
    return max


# get the roots and standalone class nodes in an RDF graph
import rdflib
from rdflib import util
def get_roots_standalones(rdfgraph):
    """
        input: rdfgraph: an RDF graph
        output: a list of roots of rdfs:subClassOf hierarchies and standalone nodes
    """
    roots = []
    for nv in util.find_roots(rdfgraph, rdflib.RDFS.subClassOf):
        if type(nv) != rdflib.term.BNode:
            roots.append(nv.toPython())
    
    if (len(roots) == 1):
        return roots
    else:
        res = rdfgraph.query(
            """
                select ?c 
                where {
                    ?c a owl:Class . 
                    filter(!isBlank(?c)) . 
                    filter not exists{?c rdfs:subClassOf ?o} . 
                    filter not exists{?o rdfs:subClassOf ?c} 
                }
            """
        )
        for item in res:
            roots.append(item[0].toPython())

        return roots



def interval_labeling(rdfgraph, source=None):
    """
        input: rdfgraph: a RDF graph for interval labeling on its rdfs:subClassOf hierarchy
        output: DataFrame indexed by nodes with ['pre', 'maxpre'] for nodes to their (pre_order, maxpre) of interval labeling
    """
    
    logging.info("Computing the intervel labeling numbers for rdfs:subClassOf hierarchy...")
    
    # convert to a networkx graph
    logging.info("Building rdfs:subClassOf hierarchical graph....")
    edges = rdfgraph.query("select ?sub ?obj where {?sub rdfs:subClassOf ?obj . filter(!isBlank(?sub)). filter(!isBlank(?obj))}")
    graph = nx.DiGraph()
    for e in edges:
        graph.add_edge(e[1].toPython(), e[0].toPython())
     

    if source != None:
        pre_maxpre_df = get_preorders(graph, source)
    else:
        # find the roots of the RDF subClassOf hierarchy
        roots = get_roots_standalones(rdfgraph)
        if (len(roots) == 1):
            root = roots[0]
        else:
            # add edges from 'http://www.w3.org/2002/07/owl#Thing' to all roots and standalone nodes
            root = 'http://www.w3.org/2002/07/owl#Thing'
            for anode in roots:
                if anode != root:
                    graph.add_edge(root, anode)
        pre_maxpre_df = get_preorders(graph, root) 
    
    logging.info("Computing the DFS pre-order numbers...")  
    pre_maxpre_df['maxpre'] = pre_maxpre_df.preorder
    
    # update the maxpres
    if source != None:
        suc = nx.dfs_successors(graph, source)
    else:
        suc = nx.dfs_successors(graph, root)
    
    logging.info("Computing the DFS maximum pre-orders numbers of hierarchical descendants....")
    change = True
    while change:
        change = False
        for n in suc:
            newMaxpre = update_maxpre(pre_maxpre_df.maxpre, suc[n])
            if newMaxpre > pre_maxpre_df.loc[n, 'maxpre']:
                pre_maxpre_df.loc[n, 'maxpre'] = newMaxpre
                change = True
    
    return pre_maxpre_df


# given a first node and a list of second nodes, 
# retrieve all nodes between the first and the seconds according to their interval labelings
def get_in_between(first, seconds, interval_labelings):
    """
        input: first: a source node in a directed graph
               seconds: a list of nodes as the targets from the first (source)
               interval_labelings: DataFrame contains the interval labelings of nodes as ('preorder', 'maxpre') indexed by nodes
        output: DataFrame contains a list of nodes between the first and the set of seconds in the hierarchical tree
    """
    #res_index = []
    #res_pre = []
    #res_maxpre = []
    n1_pre = interval_labelings.preorder[first]
    n1_maxpre = interval_labelings.maxpre[first]
    
    '''
    for n in interval_labelings.index:
        pre = interval_labelings.preorder[n]
        maxpre = interval_labelings.maxpre[n]
        # this node is a descendent of n1
        if pre > n1_pre and maxpre <= n1_maxpre:
            # assume to keep this node
            keep = True
            for secn in seconds:
                secn_pre = interval_labelings.preorder[secn]
                secn_maxpre = interval_labelings.maxpre[secn]

                # is the node a descendent of any second node?
                if pre >= secn_pre and maxpre <= secn_maxpre:
                    keep = False
                    break

            if keep:
                res_index.append(n)
                res_pre.append(pre)
                res_maxpre.append(maxpre)
    return pd.DataFrame({'preorder':res_pre, 'maxpre':res_maxpre}, index=res_index)
    '''
    
    # get n1's children
    descendant_pool = interval_labelings[(interval_labelings.preorder > n1_pre) & (interval_labelings.maxpre <= n1_maxpre)]
    
    # for each second, keep those are not second's children
    for secn in seconds:
        secn_pre = interval_labelings.preorder[secn]
        secn_maxpre = interval_labelings.maxpre[secn]
        descendant_pool = descendant_pool[(descendant_pool.preorder < secn_pre) | (descendant_pool.maxpre > secn_maxpre)]
    
    return descendant_pool


# traverse the rdfs:subClassOf hierarchy, only find the potentail
# source to target matchings between two pairs of mapped sources and targets
def get_source_target_pool(mapped_sources, mapped_targets, spre_maxpre, tpre_maxpre, possSources, possTargets):
    """
        input: mapped_sources: a list of mapped sources including the top and bottom ones in the source hierarchy
               mapped_targets: a list of mapped targets including the top and bottom ones in the target hierarchy
               spre_maxpre: interval labeling of the source hierarchy tree
               tpre_maxpre: interval labeling of the target hierarchy tree
               possSources: a list of possible sources
               possTargets: a list of possible targets
        output: a dictionary from {"a source":[list of possible targets],...}
    """
    sseconds_mapped = []
    tseconds_mapped = []
    # from end the list to the beginning->preorders decrease
    source_target_pool = defaultdict(list)
    for nidx in np.arange(len(mapped_sources)-1, 0, -1):
        scurrent = mapped_sources[nidx]
        if scurrent not in sseconds_mapped:
            sseconds_mapped.append(scurrent)

        tcurrent = mapped_targets[nidx]
        if tcurrent not in tseconds_mapped:
            tseconds_mapped.append(tcurrent)

        snext = mapped_sources[nidx - 1]
        # compare next to the end of the seconds list, get the one with the smallest preorder as the first;
        # insert next at the end of a list so that its preorder it the smallest one in the seconds list
        sfirst_mapped, sseconds_mapped = get_first_seconds(snext, sseconds_mapped, spre_maxpre)
        sources_between = get_in_between(sfirst_mapped, sseconds_mapped, spre_maxpre)
        #print(nidx, sources_between)

        tnext = mapped_targets[nidx - 1]
        tfirst_mapped, tseconds_mapped = get_first_seconds(tnext, tseconds_mapped, tpre_maxpre)
        targets_between = get_in_between(tfirst_mapped, tseconds_mapped, tpre_maxpre)
        #print(nidx, targets_between)

        for sn in list(sources_between.index):
            #print("for a sn")
            if sn in list(possSources):
                #print('possSource')
                for tn in list(targets_between.index):
                    if tn in list(possTargets):
                        #print('possTarget')
                        if tn not in source_target_pool[sn]:
                            #print('add a poss mapping')
                            source_target_pool[sn].append(tn)
                            
    return source_target_pool


# get the first and a list of seconds
def get_first_seconds(next, seconds, interval_labelings):
    """
        input: next: a next node which is potentially an upper node in the hierarchy
               seconds: a list of nodes potentially below the next node in the hierarchy
               interval_labeling
        output: first: the true upper node
                seconds: a list of nodes below the first node
    """
    
    next_pre = interval_labelings.preorder[next]
    seconds_end_pre = interval_labelings.preorder[seconds[-1]]
    if next_pre < seconds_end_pre:
        return next, seconds
    
    else:
        seconds_new = []
        first = seconds[-1]
        for _, secn in enumerate(seconds):
            secn_pre = interval_labelings.preorder[secn]
            if secn_pre > next_pre:
                seconds_new.append(secn)
        seconds_new.append(next)
        
        return first, seconds_new



def compute_emb_distance(snode, src_label_clnd_uris, tnode, tgt_label_clnd_uris, embs_model):
    """
        input: snode: a source node, tnode: a target node
               src_label_clnd_uris, tgt_label_clnd_uris: DataFrames with columns 'label', 'uri', 'clndLabel'
               embs_model: pre-trained embedding model
        output: a distance between the embedding of the source clnedLabel and the embedding of the target clndLabel
    """
    
    sclndLabel = src_label_clnd_uris[src_label_clnd_uris.uri == snode].clndLabel.values[0]
    tclndLabel = tgt_label_clnd_uris[tgt_label_clnd_uris.uri == tnode].clndLabel.values[0]
    
    semb = maponto.average_embeddings(sclndLabel, 300, embs_model)
    temb = maponto.average_embeddings(tclndLabel, 300, embs_model)
        
    dist = np.linalg.norm(temb - semb)
    
    return dist





