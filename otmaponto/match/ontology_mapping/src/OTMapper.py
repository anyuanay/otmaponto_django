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

import corpus_build_utils
from corpus_build_utils import clean_document_lower
from AlignmentFormat import serialize_mapping_to_tmp_file, serialize_mapping_to_file

import sys
import logging
from collections import defaultdict

import jellyfish
import ot

from xml.dom import minidom
from nltk.corpus import wordnet

import importlib

import OTMapOnto as maponto
import OTNeighborhood_TDA as mapneighbor

from gtda.homology import VietorisRipsPersistence

from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# extract the concepts that are not matched so far
def extract_rest_concepts(label_clnd_uris, current_align, where):
    """
        input: label_clnd_uris: DataFrame with the concepts for mapping, {'label', 'uri', 'clndLabel'}
               current_align: DataFrame with the current mapping created by previous step
               where: a string indicating whether this is for source or target concept
        output: a DataFrame with rest of the concepts for mapping, {'label', 'uri', 'clndLabel'}
    """
    if where == 'source':
        return label_clnd_uris[~label_clnd_uris.uri.isin(current_align.source)].reset_index(drop=True)
    elif where == 'target':
        return label_clnd_uris[~label_clnd_uris.uri.isin(current_align.target)].reset_index(drop=True)
    else:
        return None



# create ensemble mappings from all methods
def mapping_label_syn_jac_was_ph_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               LEVEL_4. WASSERSTEIN DISTANCE, LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_jac_was_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               LEVEL_4. WASSERSTEIN DISTANCE, LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    logging.info("Skip level_5 mapping.")

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_jac_was(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               LEVEL_4. WASSERSTEIN DISTANCE, LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    logging.info("Skip Level_5 and Level_6 mapping.")
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_was_ph_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3) 
               LEVEL_4. WASSERSTEIN DISTANCE, LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    
    logging.info("Skip the level_3 mapping")
        

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_was_ph(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3) 
               LEVEL_4. WASSERSTEIN DISTANCE, LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    
    logging.info("Skip the level_3 mapping")
        

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    logging.info("Skip level_6 mapping.")
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_was_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3) 
               LEVEL_4. WASSERSTEIN DISTANCE, (SKIP LEVEL_5), AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    
    logging.info("Skip the level_3 mapping")
        

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    
    logging.info("Skip level_5 mapping.")

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_was(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3) 
               LEVEL_4. WASSERSTEIN DISTANCE, (SKIP LEVEL_5), AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    
    logging.info("Skip the level_3 mapping")
        

    # mapping using wasserstein distance between the embeddings of sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    wds = mapneighbor.compute_pairwise_Wasserstein_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords, embs_model)
    if len(wds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds, None) 
        logging.info("The number of level_4 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    
    logging.info("Skip level_5 mapping.")
    logging.info("Skip level_6 mapping.")
    
    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_jac_ph_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               (SKIP LEVEL_4), LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    logging.info("Skip the level_4 mapping.")

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_jac_ph(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               (SKIP LEVEL_4), LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    logging.info("Skip the level_4 mapping.")

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    logging.info("Skip level_6 mapping.")

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_ph_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               (SKIP LEVEL_4), LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    
    logging.info("Skip the level_3 mapping.")

    logging.info("Skip the level_4 mapping.")

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_jac_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               (SKIP LEVEL_4), (SKIP LEVEL_5), AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    
    logging.info("Skip the level_4 mapping.")
        
    logging.info("Skip the level_5 mapping.")
        
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_jac(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, LEVEL_3. JACCARD, 
               AND (SKIP LEVEL_4, LEVEL_5, LEVEL_6)
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    # mapping using Jaccard distance between sets of related words
    suris = list(slabel_clnd_uris_rest.uri)
    turis = list(tlabel_clnd_uris_rest.uri)
    jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, src_uris_relatedWords, turis, tgt_uris_relatedWords)
    if len(jds) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, jds, None) 
        logging.info("The number of level_3 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    

    
    logging.info("Skip the level_4 mapping.")
        
    logging.info("Skip the level_5 mapping.")
        
    logging.info("Skip the level_6 mappping.")
    
    return ensemble_align.reset_index(drop=True)



# create ensemble mappings from all methods
def mapping_label_syn_ph_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3), 
               (SKIP LEVEL_4), LEVEL_5. PH, AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    logging.info("Skip the level_3 mapping.")
    logging.info("Skip the level_4 mapping.")

    
    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_ph(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3), 
               (SKIP LEVEL_4), LEVEL_5. PH, AND (SKIP LEVEL_6)
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')
   

    src_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each source concept....")
    for row in tqdm(slabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, src_graph, slabel_clnd_uris)
        src_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of source concepts computed with relatd words is {}".format(len(src_uris_relatedWords)))

    tgt_uris_relatedWords = defaultdict(dict)
    logging.info("Compute the related words for each target concept....")
    for row in tqdm(tlabel_clnd_uris_rest.itertuples()):
        word_count = mapneighbor.get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, tgt_graph, tlabel_clnd_uris)
        tgt_uris_relatedWords[row.uri] = word_count
    logging.info("Total number of target concepts computed with related words is {}".format(len(tgt_uris_relatedWords)))

    logging.info("Skip the level_3 mapping.")
    logging.info("Skip the level_4 mapping.")

    
    logging.info("Compute Persistent Homology Diagrams for embeddings of sets of related words for source and target concepts...")
    src_diagrams = mapneighbor.compute_phDiagrams(slabel_clnd_uris_rest, src_uris_relatedWords, embs_model)
    tgt_diagrams = mapneighbor.compute_phDiagrams(tlabel_clnd_uris_rest, tgt_uris_relatedWords, embs_model)

    # mapping using WD between PH diagrams of source and target concepts
    wds_phDiagrams_arr = mapneighbor.compute_pairwise_wd_phDiagrams(src_diagrams, tgt_diagrams)
    if len(wds_phDiagrams_arr) > 0:
        current_align = mapneighbor.make_mappings_distance_nn(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, wds_phDiagrams_arr, None)
        logging.info("The number of level_5 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    logging.info("Skip the level_6 mapping.")

    return ensemble_align.reset_index(drop=True)


# create ensemble mappings from all methods
def mapping_label_syn_rest(slabel_clnd_uris, tlabel_clnd_uris, src_graph, tgt_graph, embs_model):
    """
    MAPPING BY LEVEL_1. LABEL STRING, LEVEL_2. LABLE SYNONYMS, (SKIP LEVEL_3, LEVEL_4, LEVEL_5), 
            AND LEVEL_6. OT ON THE REST
        input: slabel_clnd_uris: DataFrame containing source concepts and uris with columns {'label', 'uri', 'clndLabel'}
               tlabel_clnd_uris: DataFrame containing target concepts and uris with columns {'label', 'uri', 'clndLabel'}
               src_graph: source RDF graph
               tgt_graph: target RDF graph
               embs_model: pre-trained embedding model
        output: DataFrame containing mappings with columns {'source', 'source_label', 'target', 'target_label'}
    """

    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    # start with mapping label strings
    current_align = maponto.match_concept_labels(slabel_clnd_uris, tlabel_clnd_uris, None)
    if len(current_align) > 0:
        logging.info("The number of level_1 predicted mapping is {}.".format(current_align.shape[0]))
        # concatenate found mappings
        ensemble_align = pd.concat([ensemble_align, current_align], 0)

    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris, current_align, 'target')
    # mapping label synonyms
    current_align = maponto.match_label_synonyms(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, None)
    if len(current_align) > 0:
        logging.info("The number of level_2 predicted mapping is {}".format(current_align.shape[0]))
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    
    logging.info("Skip the level_3 mapping.")
    logging.info("Skip the level_4 mapping.")
    logging.info("Skip the level_5 mapping.")

    
    # extract the concepts that are not matched so far
    slabel_clnd_uris_rest = extract_rest_concepts(slabel_clnd_uris_rest, current_align, 'source')
    tlabel_clnd_uris_rest = extract_rest_concepts(tlabel_clnd_uris_rest, current_align, 'target')

    if len(slabel_clnd_uris_rest) >0 and len(tlabel_clnd_uris_rest) > 0:
        current_align = maponto.match_label_embeddings_OT(slabel_clnd_uris_rest, tlabel_clnd_uris_rest, 
                                                      embs_model, maponto.make_mappings_nn, None, None)
        logging.info("The number of level_6 predicted mapping is {}".format(current_align.shape[0]))
    
        ensemble_align = pd.concat([ensemble_align, current_align], 0)
    
    #maponto.evaluate(ensemble_align, refs_url)

    return ensemble_align.reset_index(drop=True)


# combine mappings for concepts, object properties, and datatype properties
def ensemble_map(source_url, target_url, embs_model):
    
    logging.info("Python ensemble mapper info: map " + source_url + " to " + target_url)
    
    OWL_NS = Namespace('http://www.w3.org/2002/07/owl#')
    SKOS_NS = Namespace("http://www.w3.org/2004/02/skos/core#")
    
    source_graph = Graph()
    source_graph.namespace_manager.bind("owl", OWL_NS)
    source_graph.namespace_manager.bind("skos", SKOS_NS)

    #source_graph.parse(source_url)
    
    stack_urls = []
    stack_urls.append(source_url)
    visited_urls = []
    maponto.parse_owl_withImports(source_graph, stack_urls, visited_urls)
    
    logging.info("Read source with %s triples.", len(source_graph))

    target_graph = Graph()
    target_graph.namespace_manager.bind("owl", OWL_NS)
    target_graph.namespace_manager.bind("skos", SKOS_NS)
    
    #target_graph.parse(target_url)
    
    stack_urls = []
    stack_urls.append(target_url)
    visited_urls = []
    maponto.parse_owl_withImports(target_graph, stack_urls, visited_urls)
    
    logging.info("Read target with %s triples.", len(target_graph))
    
    
    # initialize the final ensemble mappings
    column_names = ["source", "source_label", "target", "target_label"]
    ensemble_align = pd.DataFrame(columns = column_names)
    
    
    # map concepts
    logging.info("MAP CONCEPTS")
    
    slabel_uris = maponto.extract_label_uris(source_graph)
    tlabel_uris = maponto.extract_label_uris(target_graph)

    
    slabel_clnd_uris = maponto.clean_labels(slabel_uris)
    tlabel_clnd_uris = maponto.clean_labels(tlabel_uris)
        
    max_len = len(slabel_clnd_uris)
    if len(tlabel_clnd_uris) > max_len:
        max_len = len(tlabel_clnd_uris)
    if max_len < 1000:
        concept_align = mapping_label_syn_jac_was_rest(slabel_clnd_uris, tlabel_clnd_uris, source_graph, target_graph, embs_model)
        logging.info("TOTAL NUMBER OF MAPPINGS BETWEEN CONCPETS IS {}".format(len(concept_align)))
        ensemble_align = pd.concat([ensemble_align, concept_align], 0)
    else:
        concept_align = mapping_label_syn_rest(slabel_clnd_uris, tlabel_clnd_uris, source_graph, target_graph, embs_model)
        ensemble_align = pd.concat([ensemble_align, concept_align], 0)
        logging.info("TOTAL NUMBER OF MAPPINGS BETWEEN CONCPETS IS {}".format(len(concept_align)))
    
    logging.info("=================================================")
    
    # map object properties
    logging.info("MAP OBJECT PROPERTIES")
    sobject_uris = mapneighbor.extract_objectProperty_uris(source_graph)
    sobject_clnd_uris = maponto.clean_labels(sobject_uris)
    tobject_uris = mapneighbor.extract_objectProperty_uris(target_graph)
    tobject_clnd_uris = maponto.clean_labels(tobject_uris)
    
    max_len = len(sobject_clnd_uris)
    if len(tobject_clnd_uris) > max_len:
        max_len = len(tobject_clnd_uris)
    if max_len < 500:
        object_align = mapping_label_syn_jac(sobject_clnd_uris, tobject_clnd_uris, source_graph, 
                                          target_graph, embs_model)
        logging.info("TOTAL NUMBER OF MAPPINGS BETWEEN OBJECT PROPERTIES IS {}".format(len(object_align)))
        ensemble_align = pd.concat([ensemble_align, object_align], 0)
    else:
        object_align = mapping_label_syn_rest(sobject_clnd_uris, tobject_clnd_uris, source_graph, 
                                          target_graph, embs_model)
        ensemble_align = pd.concat([ensemble_align, object_align], 0)
        logging.info("TOTAL NUMBER OF MAPPINGS BETWEEN OBJECT PROPERTIES IS {}".format(len(object_align)))
    
    logging.info("========================================================")
    
    # map datatype properties
    logging.info("MAP DATATYPE PROPERTIES")
    sdatatype_uris = mapneighbor.extract_datatypeProperty_uris(source_graph)
    sdatatype_clnd_uris = maponto.clean_labels(sdatatype_uris)
    tdatatype_uris = mapneighbor.extract_datatypeProperty_uris(target_graph)
    tdatatype_clnd_uris = maponto.clean_labels(tdatatype_uris)
    
    max_len = len(sdatatype_clnd_uris)
    if len(tdatatype_clnd_uris) > max_len:
        max_len = len(tdatatype_clnd_uris)
    if max_len < 500:
        datatype_align = mapping_label_syn_jac(sdatatype_clnd_uris, tdatatype_clnd_uris, source_graph, 
                                          target_graph, embs_model)
        logging.info("TOTAL NUMBER OF MAPPINGS BETWEEN DATATYPE PROPERTIES IS {}".format(len(datatype_align)))
        ensemble_align = pd.concat([ensemble_align, datatype_align], 0)
    else:
        datatype_align = mapping_label_syn_rest(sdatatype_clnd_uris, tdatatype_clnd_uris, source_graph, 
                                          target_graph, embs_model)
        ensemble_align = pd.concat([ensemble_align, datatype_align], 0)
        logging.info("TOTAL NUMBER OF MAPPINGS BETWEEN DATATYPE PROPERTIES IS {}".format(len(datatype_align)))
    
    logging.info("===========================================================")
    
   
    return ensemble_align.reset_index(drop=True)


### Copied from pythonMacher.py in github.com/dwslab/melt/examples/externalPythonMatcherWeb/
def match(source_url, target_url, embs_model, output_url, input_alignment_url):
    
    
    resulting_alignment = ensemble_map(source_url, target_url, embs_model)
    
    
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


class Mapper:
    
    def __init__(self, embs_model):
        self.embs_model = embs_model
    
    def align(self, source_url, target_url, output_url):
        match(source_url, target_url, self.embs_model, output_url, None)


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




