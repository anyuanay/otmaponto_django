#!/usr/bin/env python
# coding: utf-8

# # Ontology Mapping by ensembling several mapping strategies
# 1. concept label mapping
# 2. label synonyms mapping
# 3. Jaccard distances between related words of concepts
# 4. Wasserstein distances between embeddings of related words of concepts
# 5. Wasserstein distances between persistent homology graphs of embeddings of related words of concepts
# 6. Wasserstein distances between the average embeddings of the rest concepts

# In[1]:


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
import OTNeighborhood_TDA as mapneighbor

from gtda.homology import VietorisRipsPersistence

from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("logging info test")


# # Load pre-trained Fasttext embeddings

# In[2]:


importlib.reload(maponto)


# In[3]:


importlib.reload(mapneighbor)




cmt_url = "../data/conference/cmt.owl"
conference_url = "../data/conference/Conference.owl"
cmt_graph = Graph().parse(cmt_url)
conference_graph = Graph().parse(conference_url)


# In[1140]:


cmt_conference_url = "../data/conference/reference-alignment/cmt-conference.rdf"


# In[1141]:


cmtlabel_uris = maponto.extract_label_uris(cmt_graph)
conferencelabel_uris = maponto.extract_label_uris(conference_graph)


# In[1142]:


cmtlabel_clnd_uris = maponto.clean_labels(cmtlabel_uris, rmStopWords=True)
conferencelabel_clnd_uris = maponto.clean_labels(conferencelabel_uris, rmStopWords=True)
cmtlabel_clnd_uris.columns, conferencelabel_clnd_uris.columns


# In[1143]:


conf_label_align = maponto.match_concept_labels(cmtlabel_clnd_uris, conferencelabel_clnd_uris, None)
conf_label_align.shape


# In[1144]:


conf_label_align


# In[1145]:


maponto.evaluate(conf_label_align, cmt_conference_url)


# In[1146]:


# get concepts that are not matched by labels
# extract the concepts that are not matched by labels
cmtlabel_clnd_uris_rest = cmtlabel_clnd_uris[~cmtlabel_clnd_uris.uri.isin(conf_label_align.source)].    reset_index(drop=True)
conferencelabel_clnd_uris_rest = conferencelabel_clnd_uris[~conferencelabel_clnd_uris.uri.isin(conf_label_align.target)].    reset_index(drop=True)


# In[1147]:


cmtlabel_clnd_uris_rest.shape, conferencelabel_clnd_uris_rest.shape


# In[1148]:


conf_label_syn_align = maponto.match_label_synonyms(cmtlabel_clnd_uris_rest, conferencelabel_clnd_uris_rest, None)
conf_label_syn_align.shape


# In[1149]:


conf_label_syn_align


# In[1150]:


maponto.evaluate(conf_label_syn_align, cmt_conference_url)


# In[1151]:


# extract the concepts that are not matched so far
cmtlabel_clnd_uris_rest = cmtlabel_clnd_uris_rest[~cmtlabel_clnd_uris_rest.uri.isin(conf_label_syn_align.source)].    reset_index(drop=True)
conferencelabel_clnd_uris_rest = conferencelabel_clnd_uris_rest[~conferencelabel_clnd_uris_rest.uri.isin(conf_label_syn_align.target)].    reset_index(drop=True)
cmtlabel_clnd_uris_rest.shape, conferencelabel_clnd_uris_rest.shape


# In[1152]:


cmt_uris_relatedWords = defaultdict(dict)
for row in tqdm(cmtlabel_clnd_uris_rest.itertuples()):
    word_count = get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, cmt_graph, cmtlabel_clnd_uris)
    cmt_uris_relatedWords[row.uri] = word_count
len(cmt_uris_relatedWords)


# In[1153]:


conference_uris_relatedWords = defaultdict(dict)
for row in tqdm(conferencelabel_clnd_uris_rest.itertuples()):
    word_count = get_relatedWords_counts(None, row.uri, row.label, row.clndLabel, conference_graph, conferencelabel_clnd_uris)
    conference_uris_relatedWords[row.uri] = word_count
len(conference_uris_relatedWords)


# In[1104]:


suris = list(cmtlabel_clnd_uris_rest.uri)
turis = list(conferencelabel_clnd_uris_rest.uri)
jds = mapneighbor.compute_pairwise_Jaccard_distances(suris, cmt_uris_relatedWords, turis, conference_uris_relatedWords)


# In[1105]:


jds_align_relatedWords = make_mappings_distance_nn(cmtlabel_clnd_uris_rest, conferencelabel_clnd_uris_rest, jds, None) 
jds_align_relatedWords


# In[1106]:


maponto.evaluate(jds_align_relatedWords, cmt_conference_url)


# In[1107]:


# extract the concepts that are not matched so far
cmtlabel_clnd_uris_rest = cmtlabel_clnd_uris_rest[~cmtlabel_clnd_uris_rest.uri.isin(jds_align_relatedWords.source)].    reset_index(drop=True)
conferencelabel_clnd_uris_rest = conferencelabel_clnd_uris_rest[~conferencelabel_clnd_uris_rest.uri.isin(jds_align_relatedWords.target)].    reset_index(drop=True)
cmtlabel_clnd_uris_rest.shape, conferencelabel_clnd_uris_rest.shape


# In[1154]:


suris = list(cmtlabel_clnd_uris_rest.uri)
turis = list(conferencelabel_clnd_uris_rest.uri)
wds = compute_pairwise_Wasserstein_distances(suris, cmt_uris_relatedWords, turis, conference_uris_relatedWords, embs_model)


# In[1155]:


wds_align_relatedWords = make_mappings_distance_nn(cmtlabel_clnd_uris_rest, conferencelabel_clnd_uris_rest, wds, None) 
wds_align_relatedWords


# In[1156]:


maponto.evaluate(wds_align_relatedWords, cmt_conference_url)


# In[1157]:


# extract the concepts that are not matched so far
cmtlabel_clnd_uris_rest = cmtlabel_clnd_uris_rest[~cmtlabel_clnd_uris_rest.uri.isin(wds_align_relatedWords.source)].    reset_index(drop=True)
conferencelabel_clnd_uris_rest = conferencelabel_clnd_uris_rest[~conferencelabel_clnd_uris_rest.uri.isin(wds_align_relatedWords.target)].    reset_index(drop=True)
cmtlabel_clnd_uris_rest.shape, conferencelabel_clnd_uris_rest.shape


# In[1113]:


cmt_diagrams = compute_phDiagrams(cmtlabel_clnd_uris_rest, cmt_uris_relatedWords)
len(cmt_diagrams)


# In[1114]:


conference_diagrams = compute_phDiagrams(conferencelabel_clnd_uris_rest, conference_uris_relatedWords)
len(conference_diagrams)


# In[1115]:


wds_phDiagrams_arr = compute_pairwise_wd_phDiagrams(cmt_diagrams, conference_diagrams)


# In[1116]:


wds_align_ph = make_mappings_distance_nn(cmtlabel_clnd_uris_rest, conferencelabel_clnd_uris_rest, wds_phDiagrams_arr, None)


# In[1117]:


wds_align_ph


# In[1118]:


maponto.evaluate(wds_align_ph, cmt_conference_url)


# In[1119]:


# extract the concepts that are not matched so far
cmtlabel_clnd_uris_rest = cmtlabel_clnd_uris_rest[~cmtlabel_clnd_uris_rest.uri.isin(wds_align_ph.source)].    reset_index(drop=True)
conferencelabel_clnd_uris_rest = conferencelabel_clnd_uris_rest[~conferencelabel_clnd_uris_rest.uri.isin(wds_align_ph.target)].    reset_index(drop=True)
cmtlabel_clnd_uris_rest.shape, conferencelabel_clnd_uris_rest.shape


# In[1158]:


rest_emb_align = maponto.match_label_embeddings_OT(cmtlabel_clnd_uris_rest, conferencelabel_clnd_uris_rest,                                                       embs_model, maponto.make_mappings_nn, None, None)
rest_emb_align.shape


# In[1160]:


maponto.evaluate(rest_emb_align, cmt_conference_url)


# # Combine conf_label_align, conf_label_syn_align, wds_align_relatedWords, rest_emb_align

# In[1162]:


ensemble_align = pd.concat([conf_label_align, conf_label_syn_align, wds_align_relatedWords, rest_emb_align], 0)


# In[ ]:


maponto.evaluate(ensemble_align, cmt_conference_url)


# # Mapping Functions with Various Mapping Strategies

# In[ ]:





# # Load human.owl and mouse.owl

# In[825]:


mouse_url = "../data/mouse.owl"
human_url = "../data/human.owl"
mouse_graph = Graph().parse(mouse_url)
human_graph = Graph().parse(human_url)


# # Mapping on Object and Datatype Properties

# In[847]:


cmtobject_uris = extract_objectProperty_uris(cmt_graph)
cmtobject_clnd_uris = maponto.clean_labels(cmtobject_uris)
conferenceobject_uris = extract_objectProperty_uris(conference_graph)
conferenceobject_clnd_uris = maponto.clean_labels(conferenceobject_uris)


# In[924]:


cmt_object_uris_relatedWords = defaultdict(dict)
for row in tqdm(cmtobject_clnd_uris.itertuples()):
    word_count = get_relatedWords_counts("Conference", row.uri, row.label, row.clndLabel, cmt_graph, cmtobject_clnd_uris)
    cmt_object_uris_relatedWords[row.uri] = word_count
len(cmt_object_uris_relatedWords)


# In[925]:


conference_object_uris_relatedWords = defaultdict(dict)
for row in tqdm(conferenceobject_clnd_uris.itertuples()):
    word_count = get_relatedWords_counts("Conference", row.uri, row.label, row.clndLabel, conference_graph, conferenceobject_clnd_uris)
    conference_object_uris_relatedWords[row.uri] = word_count
len(conference_object_uris_relatedWords)


# In[931]:


get_ipython().run_cell_magic('time', '', 'sobject_uris = list(cmtobject_clnd_uris.uri)\ntobject_uris = list(conferenceobject_clnd_uris.uri)\nobject_wds = compute_pairwise_Wasserstein_distances(sobject_uris, cmt_object_uris_relatedWords, \\\n                                                    tobject_uris, conference_object_uris_relatedWords, embs_model)')


# In[932]:


object_wds_align_relatedWords = make_mappings_distance_nn(cmtobject_clnd_uris, conferenceobject_clnd_uris, object_wds, 999) 
object_wds_align_relatedWords


# In[933]:


cmtdatatype_uris = extract_datatypeProperty_uris(cmt_graph)
cmtdatatype_clnd_uris = maponto.clean_labels(cmtdatatype_uris)
conferencedatatype_uris = extract_datatypeProperty_uris(conference_graph)
conferencedatatype_clnd_uris = maponto.clean_labels(conferencedatatype_uris)


# In[934]:


cmt_datatype_uris_relatedWords = defaultdict(dict)
for row in tqdm(cmtdatatype_clnd_uris.itertuples()):
    word_count = get_relatedWords_counts("Conference", row.uri, row.label, row.clndLabel, cmt_graph, cmtdatatype_clnd_uris)
    cmt_datatype_uris_relatedWords[row.uri] = word_count
len(cmt_datatype_uris_relatedWords)


# In[935]:


conference_datatype_uris_relatedWords = defaultdict(dict)
for row in tqdm(conferencedatatype_clnd_uris.itertuples()):
    word_count = get_relatedWords_counts("Conference", row.uri, row.label, row.clndLabel, conference_graph, conferencedatatype_clnd_uris)
    conference_datatype_uris_relatedWords[row.uri] = word_count
len(conference_datatype_uris_relatedWords)


# In[938]:


get_ipython().run_cell_magic('time', '', 'sdatatype_uris = list(cmtdatatype_clnd_uris.uri)\ntdatatype_uris = list(conferencedatatype_clnd_uris.uri)\ndatatype_wds = compute_pairwise_Wasserstein_distances(sdatatype_uris, cmt_datatype_uris_relatedWords, \\\n                                                    tdatatype_uris, conference_datatype_uris_relatedWords, embs_model)')


# In[940]:


datatype_wds_align_relatedWords = make_mappings_distance_nn(cmtdatatype_clnd_uris, conferencedatatype_clnd_uris, datatype_wds, 999) 
datatype_wds_align_relatedWords


# In[941]:


cmt_object_diagrams = compute_phDiagrams(cmtobject_clnd_uris, cmt_object_uris_relatedWords)
len(cmt_object_diagrams)


# In[943]:


conference_object_diagrams = compute_phDiagrams(conferenceobject_clnd_uris, conference_object_uris_relatedWords)
len(conference_object_diagrams)


# In[944]:


object_wds_arr = compute_pairwise_wd_phDiagrams(cmt_object_diagrams, conference_object_diagrams)


# In[945]:


object_wds_arr[9]


# In[952]:


object_wds_align_ph = make_mappings_distance_nn(cmtobject_clnd_uris, conferenceobject_clnd_uris, object_wds_arr, None)


# In[953]:


object_wds_align_ph


# In[ ]:





# # Combine object_align_relatedWords and datatype_align_relatedWords

# In[998]:


ensemble_align = pd.concat([ensemble_align, object_wds_align_relatedWords, datatype_wds_align_relatedWords])


# In[1000]:


maponto.evaluate(ensemble_align, cmt_conference_url)


# In[1001]:


ensemble_align

