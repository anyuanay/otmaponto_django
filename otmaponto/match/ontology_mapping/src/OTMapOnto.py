import sys
sys.path.append(".")

# import libraries
import pandas as pd
import numpy as np
import scipy as sp
import json

from rdflib import Graph, URIRef, RDFS
from rdflib.namespace import Namespace, NamespaceManager
from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import load_facebook_vectors

from AlignmentFormat import serialize_mapping_to_tmp_file, serialize_mapping_to_file

import sys
import logging
from collections import defaultdict

import jellyfish
import ot
import camelsplit
import nltk
from nltk.corpus import stopwords
import re

from xml.dom import minidom
from nltk.corpus import wordnet

import warnings, math

from tqdm import tqdm

import OTNeighborhood_TDA as mapneighbor
import OTMapper as otmapper

# Based on the comparison between refs and preds, add an evaluation value to 
# the evaluation DataFrame
def add_evaluation_value(row):
    '''
        input: row: a row from the evaluation DataFrame with columns 
            ['source_refs', 'target_refs', 'source_preds', 'target_preds', 
            'source_label_preds', 'target_label_preds']
        output: a value indicating the type of evaluation results:
                ref=pred, ref!=pred, ref_only, pred_only
    '''
    if row['source_refs'] == row['source_preds']:
        if row['target_refs'] == row['target_preds']:
            return '1_match'
        else:
            return '2_nomatch'
    elif pd.isna(row['source_preds']):
        return '3_ref_only'
    elif pd.isna(row['source_refs']):
        return '4_pred_only'
    else:
        return 'unknown'

    
# compute the average of the embeddings of the words in a label string
def average_embeddings(label, dim, embs_model):
    """
        input: label string, dimension of the embeddings, pre-trained model
        output: ndarray of the averaged embeddings
    """
    
    if label in embs_model.wv.key_to_index:
        
        return embs_model.wv[label]
    
    else:
        embs = []
        
        for z in label.split():
            embs.append(embs_model.wv[z])
    
        return np.array(embs).mean(0)
            
    # return embs_model.wv[label]


# Extract rdf comments
def build_comments_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all comments edges: 'subject', 'predicate'='rdfs:comment',
                'property'=None, 'object'
    '''
    
    logging.info("build_comments_edges()....")
    
    records = []
      
    q = '''select ?s ?c
            where {
                {
                ?s rdfs:comment ?c .
                FILTER(!isBlank(?s)) .
                FILTER(!isBlank(?c))
                }
            }
         '''
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = "rdfs:comment"
        prop = None
        obj = r[1].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract datatype properties
def build_datatype_property_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all datatype property 
                edges: 'subject', 'predicate',
                'property'='dataType', 'object'
    '''
    
    logging.info("build_datatype_property_edges()....")
    records = []
    
    q = "\
            select $s ?b ?c \
            { \
                { \
                ?b rdf:type owl:DatatypeProperty . \
                ?b rdfs:domain ?s . \
                ?b rdfs:range ?c . \
                } \
                UNION \
                {?b rdf:type owl:DatatypeProperty . \
                ?b rdfs:domain ?o . \
                ?o owl:unionOf ?o1 . \
                ?o1 (rdf:rest)*/rdf:first $s . \
                ?b rdfs:range ?c . \
                } \
                filter(!isBlank(?s)) . \
                filter(!isBlank(?b)) . \
                filter(!isBlank(?c)) . \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = r[1].toPython()
        prop = 'owl:DataTypeProperty'
        obj = r[2].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])



# Extract disjoint class
def build_disjoint_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all disjoint 
                edges: 'subject', 'predicate',
                'property'=None, 'object'
    '''
    logging.info("build_disjoint_edges()....")
    records = []
    
    q = " \
            select ?s ?o2 \
            { \
                { \
                ?s owl:disjointWith ?o2 . \
                } \
                UNION \
                {?s owl:disjointWith ?o . \
                ?o owl:unionOf ?o1 . \
                ?o1 (rdf:rest)*/rdf:first ?o2 . \
                } \
                filter(!isBlank(?o2)) . \
                filter(!isBlank(?s)) .\
            } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = 'owl:disjointWith'
        prop = None
        obj = r[1].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract equivalent classes
def build_equivalentClass_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all equivalent 
                edges: 'subject', 'predicate',
                'property'=None, 'object'
    '''
    logging.info("build_equivalentClass_edges()....")
    records = []

    q = "\
            select ?s ?o2 \
            { \
                {?s owl:equivalentClass ?o2 . \
                } \
                UNION \
                {?s owl:equivalentClass ?o . \
                ?o owl:unionOf ?o1 . \
                ?o1 (rdf:rest)*/rdf:first ?o2 . \
                } \
                filter(!isBlank(?o2)) . \
                filter(!isBlank(?s)) .\
            } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = 'owl:equivalentClass'
        prop = None
        obj = r[1].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract relations from all values from
def build_relations_allValuesFrom_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all relations specified allValuesFrom
                edges: 'subject', 'predicate',
                'property'='allValuesFrom', 'object'
    '''
    
    logging.info("build_relations_allValuesFrom_edges()....")
    records = []
    
    q = " \
            select ?s ?p2 ?o2 \
            {?s rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction . \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:allValuesFrom ?o2 \
                filter(!isBlank(?b)) . \
                filter(!isBlank(?c)) .\
                filter(!isBlank(?s)) .\
            }"
    
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = r[1].toPython()
        prop = 'owl:allValuesFrom'
        obj = r[2].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])



# Extract relations as domains and ranges
def build_relations_asDomains_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all relations specified by domains
                edges: 'subject', 'predicate',
                'property'='domain', 'object' (the object will be range.)
    '''
    
    logging.info("build_relations_asDomains_edges()....")
    records = []
    
    
    q = "\
            select ?s ?b ?c \
            { \
                {?s rdfs:subClassOf ?a . \
                ?a rdf:type owl:Restriction . \
                ?a owl:onProperty ?b . \
                ?b rdf:type owl:ObjectProperty . \
                ?b rdfs:range ?c . \
                } \
                UNION \
                { \
                ?b rdf:type owl:ObjectProperty . \
                ?b rdfs:domain ?s . \
                ?b rdfs:range ?c . \
                } \
                filter(!isBlank(?b)) . \
                filter(!isBlank(?c)) .\
                filter(!isBlank(?s)) .\
            } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = r[1].toPython()
        prop = 'rdfs:domain'
        obj = r[2].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract relations from maximum cardinality constraints
def build_relations_maxCardinality_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all relations specified by maxCardinality
                edges: 'subject', 'predicate',
                'property'='owl:maxCardinality', 'object'
    '''
    
    logging.info("build_relations_maxCardinality_edges()....")
    records = []
    
    
    q = "\
            select ?s ?p2 ?o2 \
            {   ?s rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction . \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:maxCardinality ?o2 .\
                filter(!isBlank(?p2)) . \
                filter(!isBlank(?o2)) .\
                filter(!isBlank(?s)) .\
            } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = r[1].toPython()
        prop = 'owl:maxCardinality'
        obj = r[2].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract relations from mininum cardinality constraints
def build_relations_minCardinality_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all relations specified by minCardinality
                edges: 'subject', 'predicate',
                'property'='owl:minCardinality', 'object'
    '''
    
    logging.info("build_relations_minCardinality_edges()....")
    records = []
    
    q = "\
            select ?s ?p2 ?o2\
            {?s rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction . \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:minCardinality ?o2 .\
                filter(!isBlank(?p2)) . \
                filter(!isBlank(?o2)) .\
                filter(!isBlank(?s)) .\
            } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = r[1].toPython()
        prop = 'owl:minCardinality'
        obj = r[2].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract relations from some values from
def build_relations_someValuesFrom_edges(graph):
    '''
        input: graph: an RDF graph
        output: dictionary containing all relations specified by someValuesFrom
                edges: 'subject', 'predicate',
                'property'='owl:someValuesFrom', 'object'
    '''
    
    logging.info("build_relations_someValuesFrom_edges()....")
    records = []
    
    
    q = " \
            select ?s ?p2 ?o2 \
            {?s rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction .  \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:someValuesFrom ?o2 .\
                filter(!isBlank(?p2)) . \
                filter(!isBlank(?o2)) .\
                filter(!isBlank(?s)) .\
             } \
        "
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = r[1].toPython()
        prop = 'owl:someValuesFrom'
        obj = r[2].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])


# Extract all rdfs:subClassOf edges
def build_subClassOf_edges(graph):
    '''
        input: graph: an RDF graph
        output: DataFrame holding rdfs:subClassOf eddges 'subject', 'predicate'=rdfs:subClassOf, 
                'property'=None, 'object'
    '''
    logging.info("build_subClassOf_edges()....")
    
    records = []
      
    q = '''select ?s ?c
            where {
                {
                ?s rdfs:subClassOf ?c .
                FILTER(!isBlank(?s)) .
                FILTER(!isBlank(?c))
                }
            }
         '''
    
    res = graph.query(q)
    for r in res:
        sub = r[0].toPython()
        pred = "rdfs:subClassOf"
        prop = None
        obj = r[1].toPython()
        records.append((sub, pred, prop, obj))
        
    return pd.DataFrame(records, columns=['subject', 'predicate', 'property', 'object'])
    
    
# build relation graph from a RDF graph
def build_relation_graph(rdfGraph):
    """
        input: rdfGraph: a RDF graph parsed by Graph().parse(url)
        output: a DataFrame containing edges of the corresponding relation graph['subject', 'predicate', 'object']
    """
    
    logging.info("build_relation_graph(rdfGraph): Building a relation graph from the given RDF triple graph...")
    q = '''select ?s ?b ?c
            where {
                {?s rdfs:subClassOf ?a .
                ?a rdf:type owl:Restriction . 
                ?a owl:onProperty ?b .
                ?a owl:someValuesFrom ?c .
                filter(!isBlank(?s)) . 
                filter(!isBlank(?b)) . 
                filter(!isBlank(?c))
                }
                union
                {
                ?s rdfs:subClassOf ?a . 
                ?a rdf:type owl:Restriction . 
                ?a owl:onProperty ?b .
                ?b rdf:type owl:ObjectProperty .
                ?b rdfs:range ?c .
                filter(!isBlank(?s)) . 
                filter(!isBlank(?b)) . 
                filter(!isBlank(?c))
                }
                union
                {
                ?s rdfs:subClassOf ?c .
                FILTER(!isBlank(?s)) .
                filter(!isBlank(?c))
                }
            }
         '''
    
    edges = []
    for res in rdfGraph.query(q):
        if res[1] != None:
            edges.append((str(res[0]), str(res[1]), str(res[2])))
        else:
            edges.append((str(res[0]), 'rdfs:subClassOf', str(res[2])))
    
    return pd.DataFrame(edges, columns=['subject', 'predicate', 'object'])
 

# build a DataFrame containing the relation graph edges with properties from a RDF graph
def build_relation_graph_edges_df(rdfGraph):
    """
        input: rdfGraph: a RDF graph parsed by Graph().parse(url)
        output: a DataFrame containing edges along with properties of 
                the corresponding relation graph['subject', 'predicate', 'property', 'object']
                'subject' and 'object' are uris; 'predicate' and 'property' may be strings.
    """
    
    logging.info("build_relation_graph_edges_df(rdfGraph): Building a DataFrame containing the edges of a relation graph from the given RDF triple graph...")
    
    # prepare an empty DataFrame for holding the graph edges
    edges = pd.DataFrame({'subject':[], 'predicate':[], \
                            'property':[], 'object':[]})

    # retrieve rdfs:comment edges
    current_edges = build_comments_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:DataTypeProperty edges
    current_edges = build_datatype_property_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:disjointWith edges
    current_edges = build_disjoint_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:equivalentClass edges
    current_edges = build_equivalentClass_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:allValuesFrom edges
    current_edges = build_relations_allValuesFrom_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve rdfs:domain edges
    current_edges = build_relations_asDomains_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:maxCardinality edges
    current_edges = build_relations_maxCardinality_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:minCardinality edges
    current_edges = build_relations_minCardinality_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve owl:someValuesFrom edges
    current_edges = build_relations_someValuesFrom_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)
    
    # retrieve rdfs:subClassOf edges
    current_edges = build_subClassOf_edges(rdfGraph)
    edges = pd.concat([edges, current_edges]).reset_index(drop=True)    
    
    return edges


def camel_case_split(str):
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', str)).split()
    
    return splitted


def clean_document_keepStops(_in_str):
    """
    A function to prepare a document for word vector training.
    :param _in_str: An input document as a string.
    :return: A list of cleaned tokens to represent the document.
    """   
    # convert camelCase to underlined string
    _in_str = "_".join(camelsplit.camelsplit(_in_str))
    #_in_str = "_".join(camel_case_split(_in_str))
        
    _in_str = str(_in_str)
    _in_str = _in_str.replace('\n', " ")
    _in_str = re.sub(r"<NUMBER>", " ", _in_str)
    _in_str = re.sub(r"<DATE>", " ", _in_str)
    _in_str = re.sub(r"<PHONE>", " ", _in_str)
    _in_str = re.sub(r"<EMAIL>", " ", _in_str)
    _in_str = re.sub(r"<MONEY>", " ", _in_str)
    _in_str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", _in_str)
    _in_str = re.sub(r"\'s", " \'s", _in_str)
    _in_str = re.sub(r"\'ve", " \'ve", _in_str)
    _in_str = re.sub(r"n\'t", " n\'t", _in_str)
    _in_str = re.sub(r"\'re", " \'re", _in_str)
    _in_str = re.sub(r"\'d", " \'d", _in_str)
    _in_str = re.sub(r"\'ll", " \'ll", _in_str)
    _in_str = re.sub(r",", " , ", _in_str)
    _in_str = re.sub(r"!", " ! ", _in_str)
    _in_str = re.sub(r"\(", r" \( ", _in_str)
    _in_str = re.sub(r"\)", r" \) ", _in_str)
    _in_str = re.sub(r"\?", r" \? ", _in_str)
    _in_str = re.sub(r"\s{2,}", " ", _in_str)
    _in_str = re.sub(r"[^\w\s]", " ", _in_str)
    _stops = set(stopwords.words('english'))
    _out_list = list(filter(None, _in_str.lower().split(' ')))
    # _out_list = [w for w in _out_list if w not in _stops]
    return _out_list
    

def clean_document_lower(_in_str):
    """
    A function to prepare a document for word vector training.
    :param _in_str: An input document as a string.
    :return: A list of cleaned tokens to represent the document.
    """
    _in_str = str(_in_str)
    
    # convert camelCase to underlined string
    _in_str = "_".join(camelsplit.camelsplit(_in_str))
    #_in_str = "_".join(camel_case_split(_in_str))
    
    _in_str = _in_str.replace('\n', " ")
    _in_str = re.sub(r"<NUMBER>", " ", _in_str)
    _in_str = re.sub(r"<DATE>", " ", _in_str)
    _in_str = re.sub(r"<PHONE>", " ", _in_str)
    _in_str = re.sub(r"<EMAIL>", " ", _in_str)
    _in_str = re.sub(r"<MONEY>", " ", _in_str)
    _in_str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", _in_str)
    _in_str = re.sub(r"\'s", " \'s", _in_str)
    _in_str = re.sub(r"\'ve", " \'ve", _in_str)
    _in_str = re.sub(r"n\'t", " n\'t", _in_str)
    _in_str = re.sub(r"\'re", " \'re", _in_str)
    _in_str = re.sub(r"\'d", " \'d", _in_str)
    _in_str = re.sub(r"\'ll", " \'ll", _in_str)
    _in_str = re.sub(r",", " , ", _in_str)
    _in_str = re.sub(r"!", " ! ", _in_str)
    _in_str = re.sub(r"\(", r" \( ", _in_str)
    _in_str = re.sub(r"\)", r" \) ", _in_str)
    _in_str = re.sub(r"\?", r" \? ", _in_str)
    _in_str = re.sub(r"\s{2,}", " ", _in_str)
    _in_str = re.sub(r"[^\w\s]", " ", _in_str)
    _stops = set(stopwords.words('english'))
    _out_list = list(filter(None, _in_str.strip().lower().split(' ')))
    _out_list = [w for w in _out_list if w not in _stops]
    return _out_list


def clean_document_norm(_in_str):
    """
    A function to prepare a document for word vector training.
    :param _in_str: An input document as a string.
    :return: A list of cleaned tokens to represent the document.
    """
    _in_str = str(_in_str)
    
    # convert camelCase to underlined string
    #_in_str = "_".join(camelsplit.camelsplit(_in_str))
    _in_str = "_".join(camel_case_split(_in_str))
    
    _in_str = _in_str.replace('\n', " ")
   
    _in_str = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", _in_str)
    
    _in_str = re.sub(r"[^\w\s]", " ", _in_str)
    
    _stops = set(stopwords.words('english'))
    
    _out_list = list(filter(None, _in_str.strip().lower().split()))
    
    _out_list = [w for w in _out_list if w not in _stops]
    
    return _out_list

# Given a uri, extract the label at the end of the uri; clean the label; concatenate
# the individual words in the label with blanks
def clean_extract_label_from_uri(uri):
    '''
        input: uri: a uri in an ontology
        output: cleaned label: a cleaned string at the end of the uri as the label
    '''
    lab = mapneighbor.extract_label_from_uri(uri)
    
    lab_list = clean_document_norm(lab)
    
    return " ".join(lab_list)


# clean and normalize the labels
def clean_labels(label_uri_df, rmStopWords=True):
    """
        input: a DataFrame: ['label', 'uri']
               rmStopWords: if (default) True, remove stopwords; otherwise, keeps them
        output: a new DataFrame: ['label', 'clndLabel', 'uri']
    """
    if rmStopWords:
        clnd_df = pd.DataFrame(label_uri_df.label.apply(lambda x: " ".join(clean_document_lower(x)))).    rename(columns={'label':'clndLabel'})
    else:
        clnd_df = pd.DataFrame(label_uri_df.label.apply(lambda x: " ".join(clean_document_keepStops(x)))).    rename(columns={'label':'clndLabel'})
        
    
    res_df = pd.concat([label_uri_df, clnd_df], axis=1)
    
    res_df = res_df[res_df.clndLabel != '']
    
    return res_df

# compute the wd distance matrix between two list of embedding tuples
def compute_wd_matrix_embedding_tuples(suri_embedding_tuples, turi_embedding_tuples):
    
    wd_costs = []
    for semb_tuple in suri_embedding_tuples:
        wd_costs_row = []
        for temb_tuple in turi_embedding_tuples:
            _, wd = wd_between_embeddings(semb_tuple, temb_tuple)
            wd_costs_row.append(wd)
        wd_costs.append(wd_costs_row)
    wd_costs_arr = np.array(wd_costs)
    
    return wd_costs_arr


# compute the wd distance matrix between two list of embedding tuples
def compute_wd_matrix_embedding_tuples_raw(suri_embedding_tuples, turi_embedding_tuples):
    
    wd_costs = []
    for semb_tuple in suri_embedding_tuples:
        wd_costs_row = []
        for temb_tuple in turi_embedding_tuples:
            _, wd = wd_between_embeddings_raw(semb_tuple, temb_tuple)
            wd_costs_row.append(wd)
        wd_costs.append(wd_costs_row)
    wd_costs_arr = np.array(wd_costs)
    
    return wd_costs_arr


# compute the wd distance matrix between two list of labels
def compute_wd_matrix_label_tuples(suri_triples, turi_triples, embs_model):
    wd_costs = []
    for striple in suri_triples:
        wd_costs_row = []
        for ttriple in turi_triples:
            _, wd = wd_between_labels(striple, ttriple, embs_model)
            wd_costs_row.append(wd)
        wd_costs.append(wd_costs_row)
    wd_costs_arr = np.array(wd_costs)
    
    return wd_costs_arr

# compute the wd distance matrix between two list of labels
def compute_wd_matrix_label_tuples_raw(suri_triples, turi_triples, embs_model):
    wd_costs = []
    for striple in suri_triples:
        wd_costs_row = []
        for ttriple in turi_triples:
            _, wd = wd_between_labels_raw(striple, ttriple, embs_model)
            wd_costs_row.append(wd)
        wd_costs.append(wd_costs_row)
    wd_costs_arr = np.array(wd_costs)
    
    return wd_costs_arr


# convert a list of label tuples to a list of embedding tuples
def convert_label_embedding_tuples(uri_triples, embs_model):
    '''
        input: uri_triples: a list of tuples representing the triples related to a uri
               embs_model: pre-trained word embeddings
        output: a list of tuples each of which is a tuple of embeddings corresponding
                to input labels in a tuple
    '''
    emb_tuples = []
    for label_tuple in uri_triples:
        embs = []
        for label in label_tuple:
            embs.append(average_embeddings(label, 300, embs_model))
        emb_tuples.append(embs)
    
    return emb_tuples

# compute normalized costs between the source and target label embeddings
def costs_embeddings(src_label_list, tgt_label_list, embs_model):
    """
        input: source label list, target label list, a pre-trained embedding model
        output: cost matrix len(source list) x len(target list)
    """
    
    logging.info("costs_embeddings(src_label_list, tgt_label_list, embs_model): Computing the Ground Embedding Costs between the Source and Target Points...")
    sembs = []
    tembs = []
    for slabel in src_label_list:
        sembs.append(average_embeddings(slabel, 300, embs_model))
    for tlabel in tgt_label_list:
        tembs.append(average_embeddings(tlabel, 300, embs_model))
        
    costs = sp.spatial.distance.cdist(sembs, tembs)
    
    logging.info("costs_embeddings(src_label_list, tgt_label_list, embs_model):The shape of the cost matrix is {}".format(costs.shape))
        
    return costs / costs.max()

# compute the raw costs between the source and target label embeddings
def costs_embeddings_raw(src_label_list, tgt_label_list, embs_model):
    """
        input: source label list, target label list, a pre-trained embedding model
        output: cost matrix len(source list) x len(target list)
    """
    
    logging.info("costs_embeddings(src_label_list, tgt_label_list, embs_model): Computing the Ground Embedding Costs between the Source and Target Points...")
    sembs = []
    tembs = []
    for slabel in src_label_list:
        sembs.append(average_embeddings(slabel, 300, embs_model))
    for tlabel in tgt_label_list:
        tembs.append(average_embeddings(tlabel, 300, embs_model))
        
    costs = sp.spatial.distance.cdist(sembs, tembs)
    
    logging.info("costs_embeddings(src_label_list, tgt_label_list, embs_model): The shape of the cost matrix is {}".format(costs.shape))
        
    return costs


def costs_wordCount_embeddings(sword_counts, tword_counts, embs_model, method):
    """
        input: sword_counts: list of dictionaries each of which {word1:count1, word2:count2, ...}
               tword_counts: list of dictionaries each of which {word1:count1, word2:count2, ...} 
               embs_model: pre-trained word embedding model
               method: how to combine a set of word embeddings: 'averaged' and 'weighted'
        output: cost matrix len(source list) x len(target list)
    """
    
    logging.info("costs_wordCount_embeddings(sword_counts, tword_counts, embs_model, method): Computing the Ground Embedding Costs between the Lists of Source and Target Words and Counts... ")
    
    sembs = []
    tembs = []
    
    for sdict in sword_counts:
        sembs.append(weighted_embeddings(sdict, 300, embs_model, method))
    for tdict in tword_counts:
        tembs.append(weighted_embeddings(tdict, 300, embs_model, method))
    
    
    costs = sp.spatial.distance.cdist(sembs, tembs)
        
    logging.info("costs_wordCount_embeddings(sword_counts, tword_counts, embs_model, method): The shape of the cost matrix is {}".format(costs.shape))
    
    return costs / costs.max()



# Evaluate the alignment against references by precision, recall, f-measure
def evaluate(align_df, refs_rdf_path):
    """
        input: align_df: predicted alignments 'source', 'source_label', 'target', 'target_label'
               refs_rdf_path: path to references rdf file
        output: print precision, recall, f1-meaure
    """
    refs_df = load_alignments(refs_rdf_path, 'references')
    
    matched_df = align_df.merge(refs_df, how='inner', left_on=['source', 'target'], \
                                right_on=['source', 'target'])
    
    print("Total number of references is {}".format(refs_df.shape[0]))
    
    print("Total correctly predicted alignments is {}".format(matched_df.shape[0]))
    
    p = matched_df.shape[0] / align_df.shape[0]
    r = matched_df.shape[0] / refs_df.shape[0]
    f = 2 / (1/p + 1/r)
    print("Total number of predicted is {}".format(align_df.shape[0]))
    print("Precision is {}".format(p))
    print("Recall is {}".format(r))
    print("F1-Measure is {}".format(f))
    

# Evaluate the alignment against references by precision, recall, f-measure
def evaluate_noprint(align_df, refs_rdf_path):
    """
        input: align_df: predicted alignments 'source', 'source_label', 'target', 'target_label'
               refs_rdf_path: path to references rdf file
        output: a dictionary with the results such as precision, recall, f1-meaure
    """
    refs_df = load_alignments(refs_rdf_path, 'references')
    
    matched_df = align_df.merge(refs_df, how='inner', left_on=['source', 'target'], \
                                right_on=['source', 'target'])
    
    p = matched_df.shape[0] / align_df.shape[0]
    r = matched_df.shape[0] / refs_df.shape[0]
    f = 0
    if (p != 0) & (r != 0):
        f = 2 / (1/p + 1/r)
    
    return {"numOfRefs":refs_df.shape[0], "numOfCorrectlyPredicted":matched_df.shape[0], 
            "numOfPredicted":align_df.shape[0], "precision":p, "recall":r, "f1":f}
    
    
# check alignments against references;
# find the differences between references and predicted mappings
def evaluate_against_refs(refs_df, preds_df):
    '''
        input: refs_df: DataFrame with reference alignments ['source', 'target', 
               'relation', 'confidence']
               preds_df: DataFrame with predicted alignments ['source', 'source_label', 
               'target', 'target_label']
        output: DataFrame ['source_refs', 'target_refs', 'source_preds',
                'target_preds', 'source_label_preds', 'target_label_preds']], indexed by
                '1_match', '2_nomatch', '3_ref_only', '4_pred_only'
    '''
    refs_refs = refs_df.add_suffix('_refs')
    preds_preds = preds_df.add_suffix('_preds')
    joined_df = pd.merge(refs_refs, preds_preds, how="outer", left_on=["source_refs"], \
         right_on=["source_preds"])[['source_refs', 'target_refs', 'source_preds',\
         'target_preds', 'source_label_preds', 'target_label_preds']]
    
    evals = joined_df.apply(lambda row: add_evaluation_value(row), axis=1)
    
    return joined_df.set_index(evals).sort_index()


# Extract concept labels to URIs;
# A concept is a owl:Class
# return a DataFrame: ['label', 'uri']
def extract_label_uris(graph):
    """
        input: graph: an RDF graph parsed from an owl file by rdflib
        output: a DataFrame of concept labels and uris ['label', 'uri']
    """
    ## Add necessary namespaces
    OWL_NS = Namespace('http://www.w3.org/2002/07/owl#')
    SKOS_NS = Namespace("http://www.w3.org/2004/02/skos/core#")
    graph.namespace_manager.bind("owl", OWL_NS)
    graph.namespace_manager.bind("skos", SKOS_NS)
    
    ##############
    labels = []
    uris = []
    
    q_res = []
    
    try:
        def_query = '''SELECT DISTINCT ?a ?b
                   WHERE {
                   {?a a owl:Class .
                    ?a rdfs:label ?b .
                   }
                   UNION
                   {?a a owl:Class .
                    ?a skos:prefLabel ?b .
                   }
                   FILTER(!isBlank(?a))}
                   ORDER BY ?a'''
    
        q_res = graph.query(def_query)
        
        for res in q_res:
            if len(str(res[1])) > 0:
                labels.append(str(res[1]))
                uris.append(str(res[0]))
    
    except Exception as e:
        print("OTMapOnto.extract_label_uris(): first graph.query(): {}".format(e))
        
        
    if len(q_res) == 0:
        
        try:
            def_query = '''SELECT ?a
                       WHERE {
                       ?a a owl:Class .
                       FILTER(!isBlank(?a))}
                       ORDER BY ?a'''
            q_res = graph.query(def_query)
            
            #prefix = get_prefix_rdfgraph(graph)
            
            for res in q_res:
                uri = str(res[0])
                if uri not in uris:
                    idx_sharp = uri.rfind("#")
                    if idx_sharp > 0:
                        label = uri[idx_sharp + 1:]
                        labels.append(label)
                        uris.append(uri)
                    else:
                        idx_slash = uri.rfind("/")
                        if idx_slash > 0:
                            label = uri[idx_slash + 1:]
                            labels.append(label)
                            uris.append(uri)
        except Exception as e:
            print("OTMapOnto.extract_label_uris(): second graph.query(): {}".format(e))
    
    return pd.DataFrame({'label': labels, 'uri':uris})



# Given a uri, retrieve the individual words with counts in its 
# corresponding cleand label
def get_clndLabel_word_count(label_clnd_uris, uri):
    
    #logging.info("Getting individual words and counts for a cleaned label corresponding to the given uri....")
    word_count = defaultdict(int)
    labelList = label_clnd_uris[label_clnd_uris.uri == uri].clndLabel.tolist()
    if len(labelList) > 0:
        label = labelList[0]
        for w in label.split():
            word_count[w] = word_count[w] + 1
    
    return word_count

def get_context_triples(uri, edges):
    '''
        input: uri: an uri of an concept in an ontology
               edges: a DataFrame containing all relations from an ontology
               ['subject', 'predicate', 'property', 'object']
        output: a list of tuples representing the information of the triples 
                related to the input uri
    '''
    ans = []
    
    # retrieve relations in which the uri is subject
    relations = edges[edges.subject==uri]
    # row[0]: subject, row[1]: predicate, row[2]: property, row[3]:object
    for _, row in relations.iterrows():
        # 1. for rdfs:comment, add the comments string as a single element in a tuple
        if row[1] == 'rdfs:comment':
            t = (clean_extract_label_from_uri(row[3]), )
            ans.append(t)
        
        # 2. for row[1] = owl:disjointWith, add (cleaned_subject, 'disjoint different', cleaned_object)
        if row[1] == 'owl:disjointWith':
            t = (clean_extract_label_from_uri(row[0]), 'disjoint', 
                        clean_extract_label_from_uri(row[3]))
            if t not in ans:
                ans.append(t)
                
        # 3. for row[1] = owl:equivalentClass, 
        # add (cleaned_object, )
        if row[1] == 'owl:equivalentClass':
            t = (clean_extract_label_from_uri(row[3]), )
            if t not in ans:
                ans.append(t)
                
        # 4. for row[2]='owl:DataTypeProperty', 
        # add (cleaned_subject, cleaned_datatypePredicate, cleaned_object)
        if row[2] == 'owl:DataTypeProperty':
            t = (clean_extract_label_from_uri(row[0]), 
                        clean_extract_label_from_uri(row[1]),
                        clean_extract_label_from_uri(row[3]))
            if t not in ans:
                ans.append(t)
                
        # 5. for row[2] = 'owl:allValuesFrom' or 'owl:someValuesFrom'
        # add (cleaned_subject, cleaned_predicate, cleaned_object)
        if (row[2] == 'owl:allValuesFrom') | (row[2] == 'owl:someValuesFrom'):
            t = (clean_extract_label_from_uri(row[0]), 
                        clean_extract_label_from_uri(row[1]),
                        clean_extract_label_from_uri(row[3]))
            if t not in ans:
                ans.append(t)
                
        # 6. for row[2] = 'rdfs:domain'
        # add (cleaned_subject, 'domain', cleaned_predicate, cleaned_object)
        if row[2] == 'rdfs:domain':
            t = (clean_extract_label_from_uri(row[0]),
                        'domain',
                        clean_extract_label_from_uri(row[1]),
                        clean_extract_label_from_uri(row[3]))
            if t not in ans:
                ans.append(t)
                
        # 7. for row[2] = 'owl:maxCardinality'
        # add (cleaned_subject, 'maximum', cleaned_predicate, object)
        if row[2] == 'owl:maxCardinality':
            t = (clean_extract_label_from_uri(row[0]),
                        'maximum',
                        clean_extract_label_from_uri(row[1]),
                        clean_extract_label_from_uri(str(row[3])))
            if t not in ans:
                ans.append(t)
                
        # 8. for row[2] = 'owl:minCardinality'
        # add (cleaned_subject, 'minimum', cleaned_predicate, object)
        if row[2] == 'owl:minCardinality':
            t = (clean_extract_label_from_uri(row[0]),
                        'minimum',
                        clean_extract_label_from_uri(row[1]),
                        clean_extract_label_from_uri(str(row[3])))
            if t not in ans:
                ans.append(t)
                
        # 9. for row[1] = 'rdfs:subClassOf'
        # add (cleaned_subject, 'child', cleaned_object) if cleaned_object != Thing
        # otherwise, add (cleaned_subject, )
        if row[1] == 'rdfs:subClassOf':
            t = (clean_extract_label_from_uri(row[0]),
                        'child',
                        clean_extract_label_from_uri(row[3]))
            if t not in ans:
                ans.append(t)
        
    # retrieve relations in which the uri is object
    relations = edges[edges.object==uri]
    # row[0]: subject, row[1]: predicate, row[2]: property, row[3]:object
    for _, row in relations.iterrows():
       
        # 1. for row[1] = owl:disjointWith, add (cleaned_object, 'disjoint different', cleaned_subject)
        if row[1] == 'owl:disjointWith':
            t = (clean_extract_label_from_uri(row[3]), 'disjoint', 
                        clean_extract_label_from_uri(row[0]))
            if t not in ans:
                ans.append(t)
                
        # 2. for row[1] = owl:equivalentClass, 
        # add (cleaned_subject, )
        if row[1] == 'owl:equivalentClass':
            t = (clean_extract_label_from_uri(row[0]), )
            if t not in ans:
                ans.append(t)
                
        # 3. for row[2] = 'rdfs:domain'
        # add (cleaned_object, 'range', cleaned_predicate, cleaned_subject)
        if row[2] == 'rdfs:domain':
            t = (clean_extract_label_from_uri(row[3]),
                        'range',
                        clean_extract_label_from_uri(row[1]),
                        clean_extract_label_from_uri(row[0]))
            if t not in ans:
                ans.append(t)
                
        # 4. for row[1] = 'rdfs:subClassOf'
        # add (cleaned_object, 'parent', cleaned_subject) 
        if row[1] == 'rdfs:subClassOf':
            t = (clean_extract_label_from_uri(row[3]),
                        'parent',
                        clean_extract_label_from_uri(row[0]))
            if t not in ans:
                ans.append(t)
                
    # if the uri is an isolcated concept, add the concept as a singleton
    if len(ans) == 0:
        ans.append((clean_extract_label_from_uri(uri), ))
        
    return ans


# Given a uri, retrieve the individual words with counts in its 
# immediate neighbors, including the words in the meaningful 
# relation labels (excluding UNDEFINED_part_of, rdfs:subClassOf)
def get_neighbor_words(prefix, relGraph, label_clnd_uris, uri):
    """
        input: 
               prefix: namespace prefix string of the relGraph ontology
               relGraph_df: a relation graph DataFrame
               label_clnd_uris_df: a DataFrame with 'label', 'uri', 'clndLabel'
               uri
        output: a dictionary {word:count, ....}
    """
    word_count = defaultdict(int)
    
    edge_sub = relGraph[relGraph.subject == uri].merge(label_clnd_uris, how="inner", left_on='subject', \
                                                         right_on='uri')
    edge_sub_obj = edge_sub.merge(label_clnd_uris, how='inner', left_on='object', right_on='uri', \
                                  suffixes=['_sub', '_obj'])
    
    
    if edge_sub_obj.shape[0] > 0:
        # this is the cleaned label corresponding to the uri
        #label = edge_sub_obj.clndLabel_sub.tolist()[0]
        #for w in label.split():
        #    word_count[w] = word_count[w] + 1

        # get all the cleaned labels in the immediate neighbors
        neighbor_labels = edge_sub_obj.clndLabel_obj.tolist()
        for label in neighbor_labels:
            for w in label.split():
                word_count[w] = word_count[w] + 1
                
        # get the relations
        rels = edge_sub_obj.predicate.tolist()
        for rel in rels:
            rel_no_prefix = rel.replace(prefix, "")
            if rel_no_prefix != "rdfs:subClassOf" and rel_no_prefix != "UNDEFINED_part_of":
                for w in clean_document_lower(rel_no_prefix):
                    word_count[w] = word_count[w] + 1
        
    else:
        return word_count
        #labelList = label_clnd_uris[label_clnd_uris.uri == uri].clndLabel.tolist()
        #if len(labelList) > 0:
        #    label = labelList[0]
        #    for w in label.split():
        #        word_count[w] = word_count[w] + 1
    
    return word_count


# get the prefix of the rdf graph
def get_prefix_rdfgraph(graph):
    """
        input: graph, a rdf graph parsed by Graph().parse(url)
        output: prefix (namespace) string of the RDF
    """
    
    for pre, n in graph.namespaces():
        if not pre:
            prefix = n
            break
    
    return prefix


# retrieve the set of synonyms and counts for a uri from a DataFrame with 'label', 'uri', 'clndLabel'
def get_synw_counts(label_clnd_uris, uri):
    """
        input: a DataFrame with 'label', 'uri', 'clndLabel'
               uri: given uri
        output: a dictionary {synonym:count, ....} for each word in the synonyms of the clndLabel
    """

    syn_counts = defaultdict(int)
    
    labelList = label_clnd_uris[label_clnd_uris.uri == uri].clndLabel.tolist()
    if len(labelList) > 0:
        label = labelList[0]
        for w in label.split():
            for syn in wordnet.synsets(w, 'n'):
                for synw in syn.lemma_names():
                    # unnecessary because the synonyms corresps to cleaned words already
                    #clnd_synw = " ".join(clean_document_lower(synw))
                    syn_counts[synw] = syn_counts[synw] + 1
            
    return syn_counts



# Load alignments as a DataFrame
def load_alignments(rdf_path, name):
    """
        input: rdf_path: path to a rdf file with alignments
            <map>
                <Cell>
                    <entity1 rdf:resource="http://mouse.owl#MA_0002401"/>
                    <entity2 rdf:resource="http://human.owl#NCI_C52561"/>
                    <measure rdf:datatype="xsd:float">1.0</measure>
                    <relation>=</relation>
                </Cell>
            </map>
              name: 'predicted' or 'reference'
        ouptut: DataFrame with 'source', 'target', 'relation', 'measure'
    """
    
    xml_data = minidom.parse(rdf_path)
    maps = xml_data.getElementsByTagName('map')

    # print("Total number of {} is {}".format(name, len(maps)))
    
    # holds the mappings from uri to uri
    uri_maps = []
    for ele in maps:
        e1 = ele.getElementsByTagName('entity1')[0].attributes['rdf:resource'].value
        e2 = ele.getElementsByTagName('entity2')[0].attributes['rdf:resource'].value
        rel = ele.getElementsByTagName('relation')[0].childNodes[0].data
        confd = ele.getElementsByTagName('measure')[0].childNodes[0].data
        uri_maps.append((e1, e2, rel, confd))
    
    alignment = pd.DataFrame(uri_maps, columns=['source', 'target', 'relation', 'confidence'])
    
    return alignment


# Load a pre-trained word embedding model
def load_embeddings(pretrained_path, model):
    """
        input: path to a pre-trained model
        output: Fasttext Model
    """
    logging.info("load_embeddings(pretrained_path, model): Load pre-trained embeddings for about 10 mins if the model hasn't been load yet.")
    
    if model == None:
        print("Loading the pre-trained Fasttext model...Please be patient...It may take about 10 mins to load...")
        logging.info("load_embeddings(pretrained_path, model): Loading the pre-trained Fasttext model...Please be patient...It may take about 10 mins to load...")
        model = load_facebook_model(pretrained_path, encoding='utf-8')
        print("Completed loading the pre-trained Fasttext model.")
        logging.info("load_embeddings(pretrained_path, model): Completed loading the pre-trained Fasttext model.")
    
    return model


def main(argv):
    if len(argv) == 2:
        print(match(argv[0], argv[1], None))
    elif len(argv) >= 3:
        if len(argv) > 3:
            logging.error("Too many parameters but we will ignore them.")
        print(match(argv[0], argv[1], argv[2], argv[3]))
    else:
        logging.error(
            "Too few parameters. Need at least three (source and target URL of ontologies"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO
    )
    main(sys.argv[1:])
 
    
# MAKE MUTUAL Top-N MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
def make_mappings_mutual_topn(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris: target DataFrame with 'label', 'clndLabel', 'uri'
               couplings
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("make_mappings_mutual_topn(): Making Mutual Top_n Mappings from the Optimal Transport Plan...")
    
    # use mutual top_n NN from the couplings
    
    # order the couplings for each source concept
    # retrieve the top_n (last top_n) columns
    couplings_source_topn_targets = np.argsort(couplings, axis = 1)[:, -top_n:]
    
    # order the couplings for each target concept
    # retrieve the top_n (last top_n) rows
    couplings_target_topn_sources = np.argsort(couplings, axis = 0)[-top_n:, :]
    
    alignments = []
    for src, trgt_list in enumerate(couplings_source_topn_targets):
        suri = slabel_clnd_uris_sub.iloc[src].uri
        slabel = slabel_clnd_uris_sub.iloc[src].label
        for trgt in trgt_list:
            if src in couplings_target_topn_sources[:, trgt]:
                turi = tlabel_clnd_uris_sub.iloc[trgt].uri
                tlabel = tlabel_clnd_uris_sub.iloc[trgt].label
                alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res    
    

# MAKE Mutual Nearest Neighbor MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
def make_mappings_nn(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris: target DataFrame with 'label', 'clndLabel', 'uri'
               couplings
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("make_mappings_nn(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings): Making Mappings from a Pairwise OT Plan Matrix by Mutual NN...")
    
    # get the list of rows whose max > mean_of_all_row_max
    row_max_mean = couplings.max(1).mean()
    row_list = []
    for i in np.arange(couplings.shape[0]):
        if couplings[i].max() > row_max_mean:
            row_list.append(i)
    
    # use all rows
    #row_list = np.arange(couplings.shape[0])
    
    # use mutual NN from the couplings
    best_match_src = couplings.argmax(1)
    best_match_trg = couplings.argmax(0)
    mut_nn = []
    for row, col in enumerate(best_match_src):
        try:
            if row in row_list and best_match_trg[col] == row: 
                mut_nn.append((row, col))
        except IndexError:
            print('Encountered index error')
            pass
    
    alignments = []
    for row, col in mut_nn:
        suri = slabel_clnd_uris_sub.iloc[row].uri
        slabel = slabel_clnd_uris_sub.iloc[row].label
        turi = tlabel_clnd_uris_sub.iloc[col].uri
        tlabel = tlabel_clnd_uris_sub.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res

# MAKE Mutual Nearest Neighbor MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS BASED ON THEIR COSTS
def make_mappings_nn_costs(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, costs):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris: target DataFrame with 'label', 'clndLabel', 'uri'
               costs
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("make_mappings_nn_costs(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, costs): Making Mappings from a Pairwise Costs by Mutual NN...")
     
    # use all rows
    row_list = np.arange(costs.shape[0])
    
    # use mutual NN from the costs
    best_match_src = costs.argmin(1)
    best_match_trg = costs.argmin(0)
    mut_nn = []
    for row, col in enumerate(best_match_src):
        try:
            if row in row_list and best_match_trg[col] == row: 
                mut_nn.append((row, col))
        except IndexError:
            print('Encountered index error')
            pass
    
    alignments = []
    for row, col in mut_nn:
        suri = slabel_clnd_uris_sub.iloc[row].uri
        slabel = slabel_clnd_uris_sub.iloc[row].label
        turi = tlabel_clnd_uris_sub.iloc[col].uri
        tlabel = tlabel_clnd_uris_sub.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res



# MAKE Top-N MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
def make_mappings_topn(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris: target DataFrame with 'label', 'clndLabel', 'uri'
               couplings
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("make_mappings_topn(): Making Top_n Mappings from the Optimal Transport Plan...")
    
    # use top_n NN from the couplings
    
    # order the couplings for each source concept
    couplings_sorted_axis1 = np.argsort(couplings, axis = 1)
    
    # pair the source with top_n targets
    paired = list(zip(list(np.arange(couplings.shape[0])), couplings_sorted_axis1[:, -top_n:].tolist())) 
    
    alignments = []
    for row, col_list in paired:
        suri = slabel_clnd_uris_sub.iloc[row].uri
        slabel = slabel_clnd_uris_sub.iloc[row].label
        for col in col_list:
            turi = tlabel_clnd_uris_sub.iloc[col].uri
            tlabel = tlabel_clnd_uris_sub.iloc[col].label
            alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res


# MAKE Top-N MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS based on their costs
def make_mappings_topn_costs(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, costs, top_n):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris: target DataFrame with 'label', 'clndLabel', 'uri'
               costs
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("make_mappings_topn_costs(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, costs, top_n): Making Top_n Mappings from the Pairwise Costs...")
    
    # use top_n NN from the costs
    
    # order the couplings for each source concept
    costs_sorted_axis1 = np.argsort(costs, axis = 1)
    
    # pair the source with top_n targets
    paired = list(zip(list(np.arange(costs.shape[0])), costs_sorted_axis1[:, :top_n].tolist())) 
    
    alignments = []
    for row, col_list in paired:
        suri = slabel_clnd_uris_sub.iloc[row].uri
        slabel = slabel_clnd_uris_sub.iloc[row].label
        for col in col_list:
            turi = tlabel_clnd_uris_sub.iloc[col].uri
            tlabel = tlabel_clnd_uris_sub.iloc[col].label
            alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res


### Copied from pythonMacher.py in github.com/dwslab/melt/examples/externalPythonMatcherWeb/
def match(source_url, target_url, output_url, input_alignment_url):
    logging.info("match(source_url, target_url, output_url, input_alignment_url): Python matcher info: Match " + source_url + " to " + target_url)

    # in case you want the file object use
    # source_file = get_file_from_url(source_url)
    # target_file = get_file_from_url(target_url)

    source_graph = Graph()
    source_graph.parse(source_url)
    logging.info("match(source_url, target_url, output_url, input_alignment_url): Read source with %s triples.", len(source_graph))

    target_graph = Graph()
    target_graph.parse(target_url)
    logging.info("match(source_url, target_url, output_url, input_alignment_url): Read target with %s triples.", len(target_graph))

    input_alignment = None
    # if input_alignment_url is not None:
    
    
    slabel_uris = extract_label_uris(source_graph)
    tlabel_uris = extract_label_uris(target_graph)
    
    slabel_clnd_uris = clean_labels(slabel_uris)
    tlabel_clnd_uris = clean_labels(tlabel_uris)
    
    # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    match_labels_df = match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, input_alignment)
    
    logging.info("match(source_url, target_url, output_url, input_alignment_url): The number of level_1 predicted matchings is {}".format(match_labels_df.shape[0]))
    
    # extract the concepts that are not matched by labels
    slabel_clnd_uris_notLabMtc = slabel_clnd_uris[slabel_clnd_uris.uri.isin(match_labels_df.source)==False].\
    reset_index(drop=True)
    tlabel_clnd_uris_notLabMtc = tlabel_clnd_uris[tlabel_clnd_uris.uri.isin(match_labels_df.target)==False].\
    reset_index(drop=True)
    
    # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    synset_alignments = match_label_synonyms(slabel_clnd_uris_notLabMtc, tlabel_clnd_uris_notLabMtc, input_alignment)
    
    logging.info("match(source_url, target_url, output_url, input_alignment_url): The number of level_2 predicted matchings is {}".format(synset_alignments.shape[0]))
    
    resulting_alignment = pd.concat([match_labels_df, synset_alignments], axis=0).\
    reset_index(drop=True)
    
    #resulting_alignment = match_embeddings_OT(slabel_clnd_uris_notLabMtc, tlabel_clnd_uris_notLabMtc, \
                                              #embs_model, make_mappings_func, None, input_alignment)
    
    resulting_alignment["relation"] = '='
    resulting_alignment["confidence"] = 1.0

    # in case you have the results in a pandas dataframe, make sure you have the columns
    # source (uri), target (uri), relation (usually the string '='), confidence (floating point number)
    # in case relation or confidence is missing use: df["relation"] = '='  and  df["confidence"] = 1.0
    # then select the columns in the right order (like df[['source', 'target', 'relation', 'confidence']])
    # because serialize_mapping_to_tmp_file expects an iterbale of source, target, relation, confidence
    # and then call .itertuples(index=False)
    # example: alignment_file_url = serialize_mapping_to_tmp_file(df[['source', 'target', 'relation', 'confidence']].itertuples(index=False))

    #alignment_file_url = serialize_mapping_to_tmp_file(resulting_alignment[['source', 'target', 'relation', 'confidence']].itertuples(index=False))

    # alignment_file_url = serialize_mapping_to_tmp_file(resulting_alignment)
    
    #return alignment_file_url

    serialize_mapping_to_file(output_url, resulting_alignment[['source', 'target', 'relation', 'confidence']].itertuples(index=False))

    return output_url


def match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, input_alignment):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris: target DataFrame with 'label', 'clndLabel', 'uri'
               input_alignment
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    # a very simple label matcher:
    
    joined = slabel_clnd_uris.merge(tlabel_clnd_uris, on='clndLabel', how="inner", suffixes=("_source", '_target'))
    
    maps = {'uri_source':'source', 'label_source':'source_label',             'uri_target':'target', 'label_target':'target_label'}
    alignment = joined[['uri_source', 'label_source', 'uri_target', 'label_target']].rename(columns=maps)
    
    return alignment
    # return [('http://one.de', 'http://two.de', '=', 1.0)]
    


# match concept on jaccard similarity
def match_concept_labels_jac(slabel_clnd_uris, tlabel_clnd_uris, input_alignment):
    """
    match the source to target based on the jaccard similarity of the lists of 
    words in their clndLabels
        input: slabel_clnd_uris: source DataFrame with 'label', 'uri', 'clndLabel'
               tlabel_clnd_uris: target DataFrame with 'label', 'uri', 'clndLabel'
               input_alignment
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    source = []
    source_label = []
    target = []
    target_label = []
    for srow in tqdm(slabel_clnd_uris.values):
        slab = srow[2]
        for trow in tlabel_clnd_uris.values:
            tlab = trow[2]
            if mapneighbor.jaccard_similarity(slab.split(), tlab.split()) >= 0.9:
                source.append(srow[1])
                source_label.append(srow[0])
                target.append(trow[1])
                target_label.append(trow[0])
                
    res = pd.DataFrame({'source':source, 'source_label':source_label, \
                        'target':target, 'target_label':target_label})
    
    return res


# collect the labels of concept and related synonyms as a related_label_list;
# match the source and target if their related_label_lists share a common label
def match_concept_relatedSynonym_labels(slabel_clnd_uris_rest, source_graph, 
                                        tlabel_clnd_uris_rest, target_graph, input_alignment):
    
    '''
         input: slabel_clnd_uris_rest: source DataFrame with 'label', 'clndLabel', 'uri'
                source_graph: the source RDF graph
                tlabel_clnd_uris_rest: target DataFrame with 'label', 'clndLabel', 'uri'
                target_graph: the target RDF graph
                input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    '''
    
    # source related synonyms
    source_hasRelatedSynonyms = retrieve_hasRelatedSynonyms(source_graph)
    # target related synonyms
    target_hasRelatedSynonyms = retrieve_hasRelatedSynonyms(target_graph)
    
    if len(source_hasRelatedSynonyms) == 0 and len(target_hasRelatedSynonyms) == 0:
        return pd.DataFrame({'source':[], 'source_label':[], \
                            'target':[], 'target_label':[]})
    
    # make a dictionary from label to index for Hash type matching from targets
    sclndLab_idx = defaultdict(list)
    
    for i, srow in tqdm(enumerate(slabel_clnd_uris_rest.values)):
        #slab = srow[0]
        suri = srow[1]
        sclndLab = srow[2]
        if i not in sclndLab_idx[sclndLab]:
            sclndLab_idx[sclndLab].append(i)
        
        ssyn_labs = source_hasRelatedSynonyms[source_hasRelatedSynonyms.uri == suri].syn_lab.tolist()
        for ssyn_lab in ssyn_labs:
            ssyn_clndLab = " ".join(clean_document_lower(ssyn_lab))
            if i not in sclndLab_idx[ssyn_clndLab]:
                sclndLab_idx[ssyn_clndLab].append(i)
    
    
    # match from the targets
    map = []
    for j, trow in tqdm(enumerate(tlabel_clnd_uris_rest.values)):
        #tlab = trow[0]
        turi = trow[1]
        tclndLab = trow[2]
        if tclndLab in sclndLab_idx:
            sidxs = sclndLab_idx[tclndLab]
            for sidx in sidxs:
                if (sidx, j) not in map:
                    map.append((sidx, j))
        
        tsyn_labs = target_hasRelatedSynonyms[target_hasRelatedSynonyms.uri == turi].syn_lab.tolist()
        for tsyn_lab in tsyn_labs:
            tsyn_clndLab = " ".join(clean_document_lower(tsyn_lab))
            if tsyn_clndLab in sclndLab_idx:
                sidxs = sclndLab_idx[tclndLab]
                for sidx in sidxs:
                    if(sidx, j) not in map:
                        map.append((sidx, j))
    
    alignments = []
    for row, col in map:
        suri = slabel_clnd_uris_rest.iloc[row].uri
        slabel = slabel_clnd_uris_rest.iloc[row].label
        turi = tlabel_clnd_uris_rest.iloc[col].uri
        tlabel = tlabel_clnd_uris_rest.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res


def match_evaluate(source_graph, target_graph, model, reference_url, make_mappings_func, \
                   top_n=1, costBy='label', method='averaged', input_alignment_url=None):
    input_alignment = None
    
    """
        input: source_graph: source RDF graph
               target_graph: target RDF graph
               model: pre-trained embedding model
               reference_url: references for evaluation
               make_mappings_func: the function to make mappings from OT couplings
               top_n: if the make_mappings_func will choose top n from the OT couplings
               costBy: how to compute the costs: 'label', 'labelNeighbor', 'labelSyn', 
                       'labelNeighborSyn'
               method: how to combine a set of word embeddings: 'averaged' and 'weighted'
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
                print precision, recall, f1-measure
    """
    
    slabel_uris = extract_label_uris(source_graph)
    tlabel_uris = extract_label_uris(target_graph)
    
    # 'label', 'clndLabel', 'uri'
    slabel_clnd_uris = clean_labels(slabel_uris)
    tlabel_clnd_uris = clean_labels(tlabel_uris)
    
    # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    match_labels_df = match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, input_alignment)
    
    logging.info("match_evaluate(source_graph, target_graph, model, reference_url, make_mappings_func, \
                   top_n=1, costBy='label', method='averaged', input_alignment_url=None): The number of level_1 predicted matchings is {}".format(match_labels_df.shape[0]))
    
    # extract the concepts that are not matched by labels
    slabel_clnd_uris_notLabMtc = slabel_clnd_uris[slabel_clnd_uris.uri.isin(match_labels_df.source)==False].\
    reset_index(drop=True)
    tlabel_clnd_uris_notLabMtc = tlabel_clnd_uris[tlabel_clnd_uris.uri.isin(match_labels_df.target)==False].\
    reset_index(drop=True)
    
    # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    synset_alignments = match_label_synonyms(slabel_clnd_uris_notLabMtc, tlabel_clnd_uris_notLabMtc, input_alignment)
    
    logging.info("match_evaluate(source_graph, target_graph, model, reference_url, make_mappings_func, \
                   top_n=1, costBy='label', method='averaged', input_alignment_url=None): The number of level_2 predicted matchings is {}".format(synset_alignments.shape[0]))
    
    model = load_embeddings(None, model)
    
    # extract the concepts that are not matched by labels and synsets
    # 'label', 'clndLabel', 'uri'
    slabel_clnd_uris_notLabMtcSyn = slabel_clnd_uris_notLabMtc[slabel_clnd_uris_notLabMtc.\
                                    uri.isin(synset_alignments.source)==False].reset_index(drop=True)
    tlabel_clnd_uris_notLabMtcSyn = tlabel_clnd_uris_notLabMtc[tlabel_clnd_uris_notLabMtc.\
                                    uri.isin(synset_alignments.target)==False].reset_index(drop=True)
    
    
    ot_alignments = pd.DataFrame([])
    if (costBy == 'label'):
        # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
        ot_alignments = match_label_embeddings_OT(slabel_clnd_uris_notLabMtcSyn, tlabel_clnd_uris_notLabMtcSyn, \
                                                      model, make_mappings_func, top_n, input_alignment)
    elif (costBy == 'labelNeighbor'):
        # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
        ot_alignments = match_labelNeighbor_embeddings_OT(slabel_clnd_uris_notLabMtcSyn, tlabel_clnd_uris_notLabMtcSyn, \
                source_graph, target_graph, model, make_mappings_func, top_n, method, input_alignment)
    elif (costBy == 'labelSyn'):
        # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
        ot_alignments = match_labelSyn_embeddings_OT(slabel_clnd_uris_notLabMtcSyn, tlabel_clnd_uris_notLabMtcSyn, \
                                                     model, make_mappings_func, top_n, method, input_alignment)
    elif (costBy == 'labelNeighborSyn'):
        # output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
        ot_alignments = match_labelNeighborSyn_embeddings_OT(slabel_clnd_uris_notLabMtcSyn, tlabel_clnd_uris_notLabMtcSyn, \
                source_graph, target_graph, model, make_mappings_func, top_n, method, input_alignment)
    
    
    logging.info("match_evaluate(source_graph, target_graph, model, reference_url, make_mappings_func, \
                   top_n=1, costBy='label', method='averaged', input_alignment_url=None): The number of level_3 predicted matchings is {}".format(ot_alignments.shape[0]))
    
    resulting_alignment = pd.concat([match_labels_df, synset_alignments, ot_alignments], axis=0).\
    reset_index(drop=True)
    
    evaluate(resulting_alignment, reference_url)
    
    return resulting_alignment



# matching on all label features: start to build matching based
# on label features including labels, synonyms, neighboring labels, WD between neighboring labels
def match_label_features(slabel_clnd_uris, tlabel_clnd_uris, source_graph, target_graph, embs_model,
                        labSyn=False, labNeighbor=False, labNeighborSyn=False, labSyn_ws=False):
    '''
        input: slabel_clnd_uris: source DataFrame with 'label', 'uri', 'clndLabel'
               tlabel_clnd_uris: target DataFrame with 'label', 'uri', 'clndLabel'
               source_graph: source rdf graph
               target_graph: target rdf graph
               embs_mode: pre-trained model
               labSyn=0: matching on label synonyms?
               labNeighbor=0: matching on label neighborhood?
               labNeighborSyn=0: matching on synonyms of lable neighborhood?
               labSyn_ws=0: matching on WD between label synonyms?
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'           
    '''
    
    # prepare an empty DataFrame for holding the predicted matchings
    preds_align = pd.DataFrame({'source':[], 'source_label':[], \
                                'target':[], 'target_label':[]})

    # matching based on label strings
    current_align = match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None )
    preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
    print("match_concepts_labels: {}".format(current_align.shape[0]))

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = otmapper.extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = otmapper.extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')

    # matching based on RelatedSynonyms of Source and Target concepts
    current_align = match_concept_relatedSynonym_labels(slabel_clnd_uris_rest, source_graph,
                                                        tlabel_clnd_uris_rest, target_graph, None)
    preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
    print("match_concept_relatedSynonym_labels: {}".format(current_align.shape[0]))

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = otmapper.extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = otmapper.extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    # matching based on label synonyms
    current_align = match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
    print("match_label_synonyms: {}".format(current_align.shape[0]))

    if labSyn:
        # extract the concepts that are not matched so far
        slabel_clnd_uris_rest = otmapper.extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
        tlabel_clnd_uris_rest = otmapper.extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

        # matching based on label synonym embeddings
        current_align = match_labelSyn_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                              embs_model, make_mappings_nn, None, 'averaged', None)
        preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
        print("match_labelSyn_embeddings_OT: {}".format(current_align.shape[0]))

    if labNeighbor:
        # extract the concepts that are not matched so far
        slabel_clnd_uris_rest = otmapper.extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
        tlabel_clnd_uris_rest = otmapper.extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

        # matching based on label neighbor embeddings
        current_align = match_labelNeighbor_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                        source_graph, target_graph, embs_model, 
                                        make_mappings_nn, None, 'averaged', None)
        preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
        print("match_labelNeighbor_embeddings_OT: {}".format(current_align.shape[0]))

    if labNeighborSyn:
        # extract the concepts that are not matched so far
        slabel_clnd_uris_rest = otmapper.extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
        tlabel_clnd_uris_rest = otmapper.extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

        # matching based on label synonym and neighbor embeddings
        current_align = match_labelNeighborSyn_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                        source_graph, target_graph, embs_model, 
                                        make_mappings_nn, None, 'averaged', None)
        preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
        print("match_labelNeighborSyn_embeddings_OT: {}".format(current_align.shape[0]))

    if labSyn_ws:
        slabel_clnd_uris_rest = otmapper.extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
        tlabel_clnd_uris_rest = otmapper.extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

        # matching based on wasserstein similarity between sets of synonyms in source and target concepts
        current_align = match_labelSyn_wassertein_similarity(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, \
                                    embs_model, make_mappings_nn, None, None)
        preds_align = pd.concat([preds_align, current_align]).reset_index(drop=True)
        print("match_labelSyn_wassertein_similarity: {}".format(current_align.shape[0]))
        
    return preds_align



# match the ontology concepts through the embeddings of the (cleaned) labels
# and the words in the (cleaned) labels of their 1-hop neighbors and the synonyms of 
# the words in the labels by OT
def match_labelNeighborSyn_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                source_graph, target_graph, embs_model, make_mappings_func, top_n, method, input_alignment):
    """
        input: slabel_clnd_uris_sub: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris_sub: target DataFrame with 'label', 'clndLabel', 'uri'
               source_graph: an RDF graph for source ontology
               target_graph: an RDF graph for target ontology 
               embs_model: pre-trained embedding model
               make_mappings_func: the function to make mappings from OT couplings
               top_n: if the make_mappings_func will choose top n from the OT couplings
               method: how to combine a set of word embeddings: 'averaged' and 'weighted'
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("match_labelNeighborSyn_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                source_graph, target_graph, embs_model, make_mappings_func, top_n, method, input_alignment): Matching Label and Neighboring Word and Synonym Embeddings by Optimal Transport...")
    # OT applied to the embeddings of clndLabels
    
    srelGraph = build_relation_graph(source_graph)
    trelGraph = build_relation_graph(target_graph)
    
    sprefix = get_prefix_rdfgraph(source_graph)
    tprefix = get_prefix_rdfgraph(target_graph)
    
    logging.info("match_labelNeighborSyn_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                source_graph, target_graph, embs_model, make_mappings_func, top_n, method, input_alignment): Retrieving the words in the cleaned labels of the 1-hop neighbors and synonyms...  ")
    sword_counts = []
    for uri in tqdm(slabel_clnd_uris_sub.uri.tolist()):
        neigw_count = get_neighbor_words(sprefix, srelGraph, slabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(slabel_clnd_uris_sub, uri)
        labw_neigw_count = merge_dicts(labw_count, neigw_count)
        
        synw_count = get_synw_counts(slabel_clnd_uris_sub, uri)
        labw_neigw_synw_count = merge_dicts(labw_neigw_count, synw_count)
        
        sword_counts.append(labw_neigw_synw_count)
        
    tword_counts = []
    for uri in tqdm(tlabel_clnd_uris_sub.uri.tolist()):
        neigw_count = get_neighbor_words(tprefix, trelGraph, tlabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(tlabel_clnd_uris_sub, uri)
        labw_neigw_count = merge_dicts(labw_count, neigw_count)
        
        synw_count = get_synw_counts(tlabel_clnd_uris_sub, uri)
        labw_neigw_synw_count = merge_dicts(labw_neigw_count, synw_count)
        
        tword_counts.append(labw_neigw_synw_count)
    
    # compute normalized costs between the lists of dictionaries by their embeddings
    # each dictionary contains {word1:count1, word2:count2,... } corresponding to a clndLabel
    costs = costs_wordCount_embeddings(sword_counts, tword_counts, embs_model, method)
    
    # compute OT couplings between source and target labels
    couplings = ot_couplings(slabel_clnd_uris_sub.shape[0], tlabel_clnd_uris_sub.shape[0], costs)
    
    # MAKE MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
    if top_n == None:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings)
    else:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n)
    
    return alignment


# match the ontology concepts through the embeddings of the (cleaned) labels
# and the words in the (cleaned) labels of their 1-hop neighbors by OT
def match_labelNeighbor_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                source_graph, target_graph, embs_model, make_mappings_func, top_n, method, input_alignment):
    """
        input: slabel_clnd_uris_sub: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris_sub: target DataFrame with 'label', 'clndLabel', 'uri'
               source_graph: an RDF graph for source ontology
               target_graph: an RDF graph for target ontology 
               embs_model: pre-trained embedding model
               make_mappings_func: the function to make mappings from OT couplings
               top_n: if the make_mappings_func will choose top n from the OT couplings
               method: how to combine a set of word embeddings: 'averaged' and 'weighted'
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("match_labelNeighbor_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                source_graph, target_graph, embs_model, make_mappings_func, top_n, method, input_alignment): Matching Label and Neighboring Word Embeddings by Optimal Transport...")
    # OT applied to the embeddings of clndLabels
    
    srelGraph = build_relation_graph(source_graph)
    trelGraph = build_relation_graph(target_graph)
    
    sprefix = get_prefix_rdfgraph(source_graph)
    tprefix = get_prefix_rdfgraph(target_graph)
    
    logging.info("match_labelNeighbor_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                source_graph, target_graph, embs_model, make_mappings_func, top_n, method, input_alignment): Retrieving the words in the cleaned labels of the 1-hop neighbors...  ")
    sword_counts = []
    for uri in tqdm(slabel_clnd_uris_sub.uri.tolist()):
        neigw_count = get_neighbor_words(sprefix, srelGraph, slabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(slabel_clnd_uris_sub, uri)
        labw_neigw_count = merge_dicts(labw_count, neigw_count)
        sword_counts.append(labw_neigw_count)
        
    tword_counts = []
    for uri in tqdm(tlabel_clnd_uris_sub.uri.tolist()):
        neigw_count = get_neighbor_words(tprefix, trelGraph, tlabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(tlabel_clnd_uris_sub, uri)
        labw_neigw_count = merge_dicts(labw_count, neigw_count)
        tword_counts.append(labw_neigw_count)
    
    # compute normalized costs between the lists of dictionaries by their embeddings
    # each dictionary contains {word1:count1, word2:count2,... } corresponding to a clndLabel
    costs = costs_wordCount_embeddings(sword_counts, tword_counts, embs_model, method)
    
    # compute OT couplings between source and target labels
    couplings = ot_couplings(slabel_clnd_uris_sub.shape[0], tlabel_clnd_uris_sub.shape[0], costs)
    
    # MAKE MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
    if top_n == None:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings)
    else:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n)
    
    return alignment


# match the ontology concepts through the embeddings of the (cleaned) labels and
# the the synonyms of the words in the labels by OT
def match_labelSyn_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                                 embs_model, make_mappings_func, top_n, method, input_alignment):
    """
        input: slabel_clnd_uris_sub: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris_sub: target DataFrame with 'label', 'clndLabel', 'uri'
               embs_model: pre-trained embedding model
               make_mappings_func: the function to make mappings from OT couplings
               top_n: if the make_mappings_func will choose top n from the OT couplings
               method: how to combine a set of word embeddings: 'averaged' and 'weighted'
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("match_labelSyn_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                                 embs_model, make_mappings_func, top_n, method, input_alignment): Matching Label and Synonym Word Embeddings by Optimal Transport...")
    # OT applied to the embeddings of clndLabels
    
    logging.info("match_labelSyn_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                                 embs_model, make_mappings_func, top_n, method, input_alignment): Retrieving the synonyms of the words in the cleaned labels... ")
    sword_counts = []
    for uri in tqdm(slabel_clnd_uris_sub.uri.tolist()):
        synw_count = get_synw_counts(slabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(slabel_clnd_uris_sub, uri)
        labw_synw_count = merge_dicts(labw_count, synw_count)
        sword_counts.append(labw_synw_count)
        
    tword_counts = []
    for uri in tqdm(tlabel_clnd_uris_sub.uri.tolist()):
        synw_count = get_synw_counts(tlabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(tlabel_clnd_uris_sub, uri)
        labw_synw_count = merge_dicts(labw_count, synw_count)
        tword_counts.append(labw_synw_count)
    
    # compute normalized costs between the lists of dictionaries by their embeddings
    # each dictionary contains {word1:count1, word2:count2,... } corresponding to a clndLabel
    costs = costs_wordCount_embeddings(sword_counts, tword_counts, embs_model, method)
    
    # compute OT couplings between source and target labels
    couplings = ot_couplings(slabel_clnd_uris_sub.shape[0], tlabel_clnd_uris_sub.shape[0], costs)
    
    # MAKE MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
    if top_n == None:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings)
    else:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n)
    
    return alignment


# match the ontology concepts through their (cleaned) label embeddings by OT
def match_label_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, embs_model, \
                        make_mappings_func, top_n, input_alignment):
    """
        input: slabel_clnd_uris_sub: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris_sub: target DataFrame with 'label', 'clndLabel', 'uri'
               embs_model: pre-trained embedding model
               make_mappings_func: the function to make mappings from OT couplings
               top_n: if the make_mappings_func will choose top n from the OT couplings
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("match_label_embeddings_OT(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, embs_model, \
                        make_mappings_func, top_n, input_alignment): Matching Label Embeddings by Optimal Transport...")
    # OT applied to the embeddings of clndLabels
    
    # compute normalized costs between the source and target label lists by their embeddings
    costs = costs_embeddings(slabel_clnd_uris_sub.clndLabel.tolist(), tlabel_clnd_uris_sub.clndLabel.tolist(), embs_model)
    
    # compute OT couplings between source and target labels
    couplings = ot_couplings(slabel_clnd_uris_sub.shape[0], tlabel_clnd_uris_sub.shape[0], costs)
    
    # MAKE MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
    if top_n == None:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings)
    else:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n)
    
    return alignment


# match the ontology concepts through their (cleaned) label embeddings by OT
# using raw costs and non-uniform distribtuion
def match_label_embeddings_OT_raw_nonUniform(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, embs_model, \
                        make_mappings_func, top_n, mu, input_alignment):
    """
        input: slabel_clnd_uris_sub: source DataFrame with 'label', 'clndLabel', 'uri'
               tlabel_clnd_uris_sub: target DataFrame with 'label', 'clndLabel', 'uri'
               embs_model: pre-trained embedding model
               make_mappings_func: the function to make mappings from OT couplings
               top_n: if the make_mappings_func will choose top n from the OT couplings
               mu: smooting factor for non-uniform distribution computing
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("match_label_embeddings_OT_raw_nonUniform(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, embs_model, \
                        make_mappings_func, top_n, mu, input_alignment): Matching Label Embeddings by Optimal Transport...")
    # OT applied to the embeddings of clndLabels
    
    # compute normalized costs between the source and target label lists by their embeddings
    costs = costs_embeddings_raw(slabel_clnd_uris_sub.clndLabel.tolist(), tlabel_clnd_uris_sub.clndLabel.tolist(), embs_model)
    
    # compute OT couplings between source and target labels
    couplings = ot_couplings_non_uniform(costs, mu)
    
    # MAKE MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
    if top_n == None:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings)
    else:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, couplings, top_n)
    
    return alignment


# match labels by their synonyms through WordNet
def match_label_synonyms(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, input_alignment):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'uri', 'clndLabel'
               tlabel_clnd_uris: target DataFrame with 'label', 'uri', 'clndLabel',
               input_alignment
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    # find synsets for each clndLabel in both source and target
    # check if they share a synonym
    
    logging.info("match_label_synonyms(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, input_alignment): Retrieving Synsets by WordNet...")
    
    # make a dictionary from label to index for Hash type matching from targets
    sclndLab_idx = defaultdict(list)
    
    for i, srow in tqdm(enumerate(slabel_clnd_uris_sub.values)):
        sclndLab = srow[2]
        if i not in sclndLab_idx[sclndLab]:
            sclndLab_idx[sclndLab].append(i)
        # retrieve synonyms
        for syn in wordnet.synsets(sclndLab, 'n'):
            for w in syn.lemma_names():
                #w = syn.lemma_names()[0]
                clean_w = " ".join(clean_document_lower(w))
                if i not in sclndLab_idx[clean_w]:
                    sclndLab_idx[clean_w].append(i)
    
    # match from the targets
    map = []
    for j, trow in tqdm(enumerate(tlabel_clnd_uris_sub.values)):
        tclndLab = trow[2]
        if tclndLab in sclndLab_idx:
            sidxs = sclndLab_idx[tclndLab]
            for sidx in sidxs:
                if (sidx, j) not in map:
                    map.append((sidx, j))
        
        # retrieve synonyms
        for tsyn in wordnet.synsets(tclndLab, 'n'):
            for tw in tsyn.lemma_names():
                tclean_w = " ".join(clean_document_lower(tw))
                if tclean_w in sclndLab_idx:
                    sidxs = sclndLab_idx[tclean_w]
                    for sidx in sidxs:
                        if(sidx, j) not in map:
                            map.append((sidx, j))
    
    alignments = []
    for row, col in map:
        suri = slabel_clnd_uris_sub.iloc[row].uri
        slabel = slabel_clnd_uris_sub.iloc[row].label
        turi = tlabel_clnd_uris_sub.iloc[col].uri
        tlabel = tlabel_clnd_uris_sub.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res

# match labels by their synonym phrases through WordNet
def match_label_synonym_phrases(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, input_alignment):
    """
        input: slabel_clnd_uris: source DataFrame with 'label', 'uri', 'clndLabel'
               tlabel_clnd_uris: target DataFrame with 'label', 'uri', 'clndLabel',
               input_alignment
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    # find synsets for each clndLabel in both source and target
    # check if they share a synonym
    
    logging.info("match_label_synonym_phrases(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, input_alignment): Retrieving Synset Phrases by WordNet...")
    
    # make a dictionary from label to index for Hash type matching from targets
    sclndLab_idx = defaultdict(list)
    
    for i, srow in tqdm(enumerate(slabel_clnd_uris_sub.values)):
        sclndLab = srow[2]
        # retrieve all synonym phrases
        ssyn_phrases = mapneighbor.get_syn_phrases(sclndLab)
        for ssyn_phrase in ssyn_phrases:
            if i not in sclndLab_idx[ssyn_phrase]:
                sclndLab_idx[ssyn_phrase].append(i)
    
    # match from the targets
    map = []
    for j, trow in tqdm(enumerate(tlabel_clnd_uris_sub.values)):
        tclndLab = trow[2]
        # retrieve all synonym phrases
        tsyn_phrases = mapneighbor.get_syn_phrases(tclndLab)
        for tsyn_phrase in tsyn_phrases:
            if tsyn_phrase in sclndLab_idx:
                sidxs = sclndLab_idx[tsyn_phrase]
                for sidx in sidxs:
                    if(sidx, j) not in map:
                        map.append((sidx, j))
    
    alignments = []
    for row, col in map:
        suri = slabel_clnd_uris_sub.iloc[row].uri
        slabel = slabel_clnd_uris_sub.iloc[row].label
        turi = tlabel_clnd_uris_sub.iloc[col].uri
        tlabel = tlabel_clnd_uris_sub.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))
    
    res = pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])
    
    return res


# match the ontology concepts through Wasserstein similarity between
# sets of synonym embeddings
def match_labelSyn_wassertein_similarity(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, \
                                     embs_model, make_mappings_func, top_n, input_alignment):
    """
        input: slabel_clnd_uris_sub: source DataFrame with 'label', 'uri', 'clndLabel'
               tlabel_clnd_uris_sub: target DataFrame with 'label', 'uri', 'clndLabel'
               embs_model: pre-trained embedding model
               make_mappings_func: the function to make mappings from couplings
               top_n: if the make_mappings_func will choose top n from the couplings
               input_alignment: no use
        output: alignment DataFrame with: 'source', 'source_label', 'target', 'target_label'
    """
    
    logging.info("match_labelSyn_wassertein_similarity(): Matching Label \
                 and Synonym Embeddings by Wassertein Similarity...")
    
    # OT applied to the embeddings of clndLabels
    
    logging.info("match_labelSyn_wasserstein_similarity(): Retrieving the synonyms of the words in the cleaned labels... ")
    
    sword_counts = []
    for uri in tqdm(slabel_clnd_uris_sub.uri.tolist()):
        synw_count = get_synw_counts(slabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(slabel_clnd_uris_sub, uri)
        labw_synw_count = merge_dicts(labw_count, synw_count)
        # append a dictionary {synonym:count, ....} to the list of all sources
        sword_counts.append(labw_synw_count)
        
    tword_counts = []
    for uri in tqdm(tlabel_clnd_uris_sub.uri.tolist()):
        synw_count = get_synw_counts(tlabel_clnd_uris_sub, uri)
        labw_count = get_clndLabel_word_count(tlabel_clnd_uris_sub, uri)
        labw_synw_count = merge_dicts(labw_count, synw_count)
        tword_counts.append(labw_synw_count)
    
    # compute wasserstein similarities between the lists of dictionaries by their embeddings
    # each dictionary contains {word1:count1, word2:count2,... } corresponding to a clndLabel
    # ws_coupings is numpy array
    ws_couplings = ws_synCount_embeddings(sword_counts, tword_counts, embs_model)
    
    # MAKE MAPPINGS BETWEEN SOURCE LABLES AND TARGET LABELS
    if top_n == None:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, ws_couplings)
    else:
        alignment = make_mappings_func(slabel_clnd_uris_sub, tlabel_clnd_uris_sub, ws_couplings, top_n)
    
    return alignment



# merge two default dictionary
def merge_dicts(firstDefault, secondDefault):
    """
        input: two default dictionaries
        output: a new default dictionary with merged keys and counts (added counts, not replaced)
    """
    res_dict = defaultdict(int)
    for k1 in firstDefault:
        res_dict[k1] =res_dict[k1] + firstDefault[k1]
        
    for k2 in secondDefault:
        res_dict[k2] =res_dict[k2] + secondDefault[k2]
        
    return res_dict


# get all normalized noun synonyms and counts
def normalize_nounsyn_counts(label):
    """
        input: a label string
        output: a dictionary {normalized_synonym:count, ....}
    """
    
    # logging.info("normalize_nounsyn_counts(label): Retrieving Synsets by WordNet...")
    syn_counts = defaultdict(int)
    syn_counts[label] = syn_counts[label] + 1
    
    for syn in wordnet.synsets(label, 'n'):
        for w in syn.lemma_names():
        #w = syn.lemma_names()[0]
            clean_w = " ".join(clean_document_lower(w))
            syn_counts[clean_w] = syn_counts[clean_w] + 1
        
            
    return syn_counts


# computing a non-uniform distribution for a set of points based on 
# the given shortest distances to another set of points
def non_uniform_dist(costs, mu):
    '''
        input:  costs: a nxm matrix (narray) of distances from a set of n points to a set of 
                     m points
                mu: a smoothing value for eliminating zero distances
        output: a probability distribution for the n points 
    '''
    ds = costs.min(axis=1)
    ds_mu_inv = 1 / (ds + mu)
    p = ds_mu_inv / ds_mu_inv.sum()
    
    return p


# compute OT couplings between source and target labels
def ot_couplings(lmv, lnv, costs):
    """
        input: length of mv, length of nv, ground costs
        output: OT couplings between mv and nv
    """
    
    logging.info("ot_couplings(lmv, lnv, costs): Computing Optimal Transport Plan...")
    
    # uniform distributions on source and target
    a = np.ones((lmv,)) / lmv
    b = np.ones((lnv,)) / lnv
    
    with warnings.catch_warnings(record=True) as ws:
        # reg term
        
        #initial value from ot example
        #lambd = 1e-3
        lambd = 6 * 1e-2
        
        logging.info("ot_couplings(lmv, lnv, costs): Computing Wasserstein distance by the Sinkhorn algorithm...")
        couplings = ot.sinkhorn(a, b, costs, lambd)
                        
        if len(ws) > 0:
            # compute earch mover couplings and distances (Wasserstein-2)
            logging.info("ot_couplings(lmv, lnv, costs): The Sinkhorn got warnings. Computing Wasserstein distance by the EMD algorithm...")
            couplings = ot.emd(a, b, costs)
    
    return couplings

# compute OT couplings between source and target labels
# by uniform distributions for both source and target
def ot_couplings_uniform(costs):
    """
        input: ground costs
        output: OT couplings between mv and nv
    """
    
    logging.info("ot_couplings_uniform(costs): Computing Optimal Transport Plan Uniform Distribution...")
    
    # uniform distributions on source and target
    lmv = costs.shape[0]
    lnv = costs.shape[1]
    a = np.ones((lmv,)) / lmv
    b = np.ones((lnv,)) / lnv
    
    with warnings.catch_warnings(record=True) as ws:
        # reg term
        lambd = 1e-3
        logging.info("ot_couplings_uniform(costs): Computing Wasserstein distance by the Sinkhorn algorithm...")
        couplings = ot.sinkhorn(a, b, costs, lambd)
                        
        if len(ws) > 0:
            # compute earch mover couplings and distances (Wasserstein-2)
            logging.info("ot_couplings_uniform(costs): The Sinkhorn got warnings. Computing Wasserstein distance by the EMD algorithm...")
            couplings = ot.emd(a, b, costs)
    
    return couplings


# compute OT couplings between source and target labels
# by non-uniform distributions for source and target
def ot_couplings_non_uniform(costs, mu):
    """
        input: length of mv, length of nv, ground costs
        output: OT couplings between mv and nv
    """
    
    logging.info("ot_couplings_non_uniform(costs, mu): Computing Optimal Transport Plan Non-uniform Distribution...")
    
    # Non-uniform distributions on source and target
    a = non_uniform_dist(costs, mu)
    b = non_uniform_dist(costs.T, mu)
    
    with warnings.catch_warnings(record=True) as ws:
        # reg term
        lambd = 1e-3
        logging.info("ot_couplings_non_uniform(costs, mu): Computing Wasserstein distance by the Sinkhorn algorithm...")
        couplings = ot.sinkhorn(a, b, costs, lambd)
                        
        if len(ws) > 0:
            # compute earch mover couplings and distances (Wasserstein-2)
            logging.info("ot_couplings_non_uniform(costs, mu): The Sinkhorn got warnings. Computing Wasserstein distance by the EMD algorithm...")
            couplings = ot.emd(a, b, costs)
    
    return couplings



# Recursively parse all imported OWL files
def parse_owl_withImports(graph, stack_urls, visited_urls):
    """
        input: graph: an RDF graph to be generated by parsing imported urls
                      stack_urls: a stack of imported urls for parsing
                      visited_urls: urls have beedn parsed
        output: the input graph that have been generated with all imported owls
    """
    # no more owl files to parse
    if len(stack_urls) == 0:
        return
    
    url = stack_urls.pop()
    print(url)
    visited_urls.append(url)
    
    try:
        graph.parse(url, format='xml')
    
        q = """
        SELECT ?o ?s
        WHERE {
            ?o owl:imports  ?s.
        }
        """

        # Apply the query to get all owl:imports sources
        for r in graph.query(q):
            imported_url = r[1].toPython()
            if (imported_url not in stack_urls) and (imported_url not in visited_urls):
                stack_urls.append(imported_url)
    except:
        logging.info("parse_owl_withImports: graph.prase(): an exception occured.")
    
    parse_owl_withImports(graph, stack_urls, visited_urls)
    

# Extract rdf comments
def query_comments(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
        
    q = " \
            select ?o \
            {" + \
                concept + " rdfs:comment ?o . \
                filter(!isBlank(?o)) . \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append("rdfs:comment")
        related.append(r[0].toPython())
        what.append("comment")
    return relation, related, what


# Extract datatype properties
def query_datatype_properties(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = "\
            select ?b ?c \
            { \
                { \
                ?b rdf:type owl:DatatypeProperty . \
                ?b rdfs:domain " + concept + " . \
                ?b rdfs:range ?c . \
                } \
                UNION \
                {?b rdf:type owl:DatatypeProperty . \
                ?b rdfs:domain ?o. \
                ?o owl:unionOf ?o1 . \
                ?o1 (rdf:rest)*/rdf:first " + concept + " . \
                ?b rdfs:range ?c . \
                } \
                filter(!isBlank(?b)) . \
                filter(!isBlank(?c)) . \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("datatype domain")
    return relation, related, what


# Extract disjoint class
def query_disjoint_classes(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = " \
            select ?o2 \
            { \
                {" + \
                concept + " owl:disjointWith ?o2 . \
                } \
                UNION \
                {" + concept + " owl:disjointWith ?o . \
                ?o owl:unionOf ?o1 . \
                ?o1 (rdf:rest)*/rdf:first ?o2 . \
                } \
                filter(!isBlank(?o2)) . \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append("owl:disjointWith")
        related.append(r[0].toPython())
        what.append("disjoint")
    return relation, related, what

# Extract equivalent classes
def query_equivalent_classes(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)

    q = "\
            select ?o2 \
            { \
                {" + \
                 concept + " owl:equivalentClass ?o2 . \
                } \
                UNION \
                {" + concept + " owl:equivalentClass ?o . \
                ?o owl:unionOf ?o1 . \
                ?o1 (rdf:rest)*/rdf:first ?o2 . \
                } \
                filter(!isBlank(?o2)) . \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append("owl:equivalentClass")
        related.append(r[0].toPython())
        what.append("equivalent")
    return relation, related, what


# query all related information for a uri from an RDF graph
def query_related_information(uri, graph):
    '''
        input:uri: a concept uri in the given RDF graph
              graph: an RDF graph
        output: a DataFrame with columns "relation", "related", "what"
    '''
    relation = []
    related = []
    what = []
    
    l1, l2, l3 = query_superClasses(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_subClasses(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_relations_someValuesFrom(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_relations_allValuesFrom(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_relations_minCardinality(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_relations_maxCardinality(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_relations_asDomains(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_datatype_properties(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_relations_asRanges(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_comments(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_equivalent_classes(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    l1, l2, l3 = query_disjoint_classes(uri, graph)
    relation.extend(l1)
    related.extend(l2)
    what.extend(l3)
    
    df = pd.DataFrame({"relation":relation, "related":related, "what":what})
    
    return df.drop_duplicates()


# Extract relations from all values from
def query_relations_allValuesFrom(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = " \
            select ?p2 ?o2 \
            {" + concept + " rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction . \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:allValuesFrom ?o2 \
            }"
    
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("allValuesFrom")
    return relation, related, what


# Extract relations as domains
def query_relations_asDomains(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = "\
            select ?b ?c \
            { \
                {" + concept + " rdfs:subClassOf ?a . \
                ?a rdf:type owl:Restriction . \
                ?a owl:onProperty ?b . \
                ?b rdf:type owl:ObjectProperty . \
                ?b rdfs:range ?c . \
                } \
                UNION \
                { \
                ?b rdf:type owl:ObjectProperty . \
                ?b rdfs:domain " + concept + " . \
                ?b rdfs:range ?c . \
                } \
                filter(!isBlank(?b)) . \
                filter(!isBlank(?c)) \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("domain")
    return relation, related, what




# Extract relations as range
def query_relations_asRanges(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)

    q = "\
            select ?b ?c \
            { \
                {?c rdfs:subClassOf ?a . \
                ?a rdf:type owl:Restriction . \
                ?a owl:onProperty ?b . \
                ?b rdf:type owl:ObjectProperty . \
                ?b rdfs:range " + concept + " . \
                } \
                UNION \
                { \
                ?b rdf:type owl:ObjectProperty . \
                ?b rdfs:range " + concept + " . \
                ?b rdfs:domain ?c . \
                } \
                filter(!isBlank(?b)) . \
                filter(!isBlank(?c)) \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("range")
    return relation, related, what


# Extract relations from maximum cardinality constraints
def query_relations_maxCardinality(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = "\
            select ?p2 ?o2 \
            {" + \
                concept + " rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction . \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:maxCardinality ?o2 \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("owl:maxCardinality")
    return relation, related, what


# Extract relations from mininum cardinality constraints
def query_relations_minCardinality(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = "\
            select ?p2 ?o2\
            {" + \
                concept + " rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction . \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:minCardinality ?o2 \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("owl:minCardinality")
    return relation, related, what


# Extract relations from some values from
def query_relations_someValuesFrom(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    #print(concept)
    
    q = " \
            select ?p2 ?o2 \
            {" + concept + " rdfs:subClassOf ?o1 . \
                ?o1 rdf:type owl:Restriction .  \
                ?o1 owl:onProperty ?p2 . \
                ?o1 owl:someValuesFrom ?o2 \
             } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("someValuesFrom")
    return relation, related, what


# Extract children -> sub classes
def query_subClasses(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
    
    q = " \
            select ?s \
            { \
                ?s rdfs:subClassOf " + concept + ". \
                filter(!isBlank(?s)) . \
            } \
        "
    
    res = graph.query(q)
    for r in res:
        relation.append("superClassOf")
        related.append(r[0].toPython())
        what.append("parent")
    return relation, related, what



# Extract parents -> super classes
def query_superClasses(uri, graph):
    '''
        input: uri: a full uri in the given RDF graph
               graph: an RDF graph
        output: 3 lists: relation: list of relations
                         related: list of related concepts (uris)
                         what: list of types of the relations
    '''
    relation = []
    related = []
    what = []
    
    concept = uri_concept(uri, graph)
      
    q = "select ?o {" + concept + " \
        rdfs:subClassOf ?o . \
        filter(!isBlank(?o)) .} "
    
    res = graph.query(q)
    for r in res:
        relation.append("rdfs:subClassOf")
        related.append(r[0].toPython())
        what.append("child")
    return relation, related, what


# Retrieve oboInOwl:hasRelatedSynonym
def retrieve_hasRelatedSynonyms(graph):
    
    q = " \
            select ?o1 ?o2 ?p\
            {?o1 oboInOwl:hasRelatedSynonym ?o2 . \
                ?o2 rdfs:label ?p . \
            }"

    records = []
    
    try:
        res = graph.query(q)
        for r in res:
            uri = r[0].toPython()
            syn_uri = r[1].toPython()
            syn_lab = r[2].toPython()
            records.append((uri, syn_uri, syn_lab))
    
    except Exception as e:
        prefix = get_prefix_rdfgraph(graph)
        logging.info("retrieve_hasRelatedSynonyms(graph): No oboInOwl:hasRelatedSynonym in this ontology {}: {}".format(prefix, e))
        
    df = pd.DataFrame(records, columns=['uri', 'syn_uri', 'syn_lab'])
    
    return df


# Convert uri to concept for SPARQL query
def uri_concept(uri, graph):
    '''
        input: uri: a full uri in the graph
               graph: a RDF graph
        output: the concept without the base prefix in the format :Concept
    '''
    prefix = get_prefix_rdfgraph(graph)
    concept = ":" + uri.replace(prefix, "")
    
    return concept


# compute the Wasserstein distance between two list of label embeddings
def wd_between_embeddings(semb_list, temb_list):
    '''
        input: semb_list: a list of source label embeddings
               temb_list: a list of target label embeddings
        output: 
                costs: the normalized ground costs between the source and target embeddings
                wd: Wasserstein distance between the source and target embedding lists
    '''
    costs = ot.dist(np.array(semb_list), np.array(temb_list))

    costs = costs / costs.max()

    a = np.ones((len(semb_list),)) / len(semb_list)
    b = np.ones((len(temb_list),)) / len(temb_list)

    wd = ot.emd2(a, b, costs)
    
    return costs, wd

# compute the Wasserstein distance between two list of label embeddings without normalized costs
def wd_between_embeddings_raw(semb_list, temb_list):
    '''
        input: semb_list: a list of source label embeddings
               temb_list: a list of target label embeddings
        output: 
                costs: the normalized ground costs between the source and target embeddings
                wd: Wasserstein distance between the source and target embedding lists
    '''
    costs = ot.dist(np.array(semb_list), np.array(temb_list))

    a = np.ones((len(semb_list),)) / len(semb_list)
    b = np.ones((len(temb_list),)) / len(temb_list)

    wd = ot.emd2(a, b, costs)
    
    return costs, wd

# compute the Wasserstein distance between the concept labels of two ontologies
def wd_between_labels(slabel_list, tlabel_list, embs_model):
    '''
        input: slabel_list: a list of source clnd labels
               tlabel_list: a list of target clnd labels
               embs_model: pre-trained model
        output: 
                costs: the normalized ground costs between the source and target labels
                wd: Wasserstein distance between the source and target label lists
    '''
    sembs = []
    tembs = []
    for slabel in slabel_list:
        sembs.append(average_embeddings(slabel, 300, embs_model))
    for tlabel in tlabel_list:
        tembs.append(average_embeddings(tlabel, 300, embs_model))

    costs = ot.dist(np.array(sembs), np.array(tembs))

    #logging.info("wd_between_labels(slabel_list, tlabel_list, embs_model): The shape of the cost matrix is {}".format(costs.shape))

    costs = costs / costs.max()

    a = np.ones((len(sembs),)) / len(sembs)
    b = np.ones((len(tembs),)) / len(tembs)

    wd = ot.emd2(a, b, costs)
    
    return costs, wd

# compute the raw Wasserstein distance between the concept labels of two ontologies
# using the raw costs, not normalized costs
def wd_between_labels_raw(slabel_list, tlabel_list, embs_model):
    '''
        input: slabel_list: a list of source clnd labels
               tlabel_list: a list of target clnd labels
               embs_model: pre-trained model
        output: 
                costs: the raw ground costs between the source and target labels
                wd: raw Wasserstein distance between the source and target label lists
    '''
    sembs = []
    tembs = []
    for slabel in slabel_list:
        sembs.append(average_embeddings(slabel, 300, embs_model))
    for tlabel in tlabel_list:
        tembs.append(average_embeddings(tlabel, 300, embs_model))

    costs = ot.dist(np.array(sembs), np.array(tembs))

    #logging.info("wd_between_labels_raw(slabel_list, tlabel_list, embs_model): The shape of the cost matrix is {}".format(costs.shape))

    a = np.ones((len(sembs),)) / len(sembs)
    b = np.ones((len(tembs),)) / len(tembs)

    wd = ot.emd2(a, b, costs)
    
    return costs, wd


# compute wasserstein similarities between the lists of dictionaries by their embeddings
# each dictionary contains {word1:count1, word2:count2,... } corresponding to a clndLabel
# ws_coupings is numpy array
def ws_synCount_embeddings(sword_counts, tword_counts, embs_model):
    """
        input: sword_counts: source list of dictionaries each of which ontains syn words 
                             to their counts for a source clnd label
               tword_counts: target list of dictionaries each of which ontains syn words 
                             to their counts for a target clnd label
               embs_model: pre-trained embedding model
        output: an ndarray of Wasserstein similarities between the sclndLabels and tclndLabels
    """
    
    logging.info("ws_synCount_embeddings(): Computing pairwise Wasserstein \
    Similarities between Synonym Sets of Source and Target Labels...")
    
    # convert source words to embeddings
    sembs_distrs = []
    for sword_counts_dict in tqdm(sword_counts):
        src_embs = []
        src_counts = []
        for sw in sword_counts_dict:
            src_embs.append(average_embeddings(sw, 300, embs_model))
            src_counts.append(sword_counts_dict[sw])
        src_distr = np.array(src_counts) / np.array(src_counts).sum()
        sembs_distrs.append((src_embs, src_distr))
    
    tembs_distrs = []
    for tword_counts_dict in tqdm(tword_counts):  
        tgrt_embs = []
        tgrt_counts = []
        for tw in tword_counts_dict:
            tgrt_embs.append(average_embeddings(tw, 300, embs_model))
            tgrt_counts.append(tword_counts_dict[tw])
        tgrt_distr = np.array(tgrt_counts) / np.array(tgrt_counts).sum()
        tembs_distrs.append((tgrt_embs, tgrt_distr))
        
    ws_lists = []  
    for sembs_distr in tqdm(sembs_distrs):
        ws = []
        for tembs_distr in tembs_distrs:
            costs = sp.spatial.distance.cdist(sembs_distr[0], tembs_distr[0])
            wd = ot.emd2(sembs_distr[1], tembs_distr[1], costs)
            ws.append(math.exp(-wd))
        
        ws_lists.append(ws)
    
    return np.array(ws_lists)


# compute the weighted embedding of a list of words with counts in a dictionary
def weighted_embeddings(aDict, dim, embs_model, method): 
    """
        input: aDict: dictionary with {word1:count1, word2:count2,...}
               dim: dimension of the embedding
               embs_model: pre-trained embedding model
               method: how to combine a set of word embeddings: 'averaged' and 'weighted'
        output: weighted embedding of the word1, word2,... by their counts...
    """
    embs = []
    counts = []
    
    for z in aDict:
        embs.append(embs_model.wv[z])
        counts.append(aDict[z])
    
    embs_arr = np.array(embs)
    weights_arr = np.array(counts) / np.array(counts).sum()

    if method == 'weighted':
        weighted_emb = (embs_arr * weights_arr.reshape(-1, 1)).sum(0)
        
    if method == 'averaged':
        weighted_emb = embs_arr.mean(0)
        
    return weighted_emb















