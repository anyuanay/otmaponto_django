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

from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from IPython.display import display, HTML

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model

import OTMapOnto as maponto

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("logging info test")

# # Make StellarGraph from RDF Graph
# 1. extract concept labels from RDF graph
# 2. assign ids to each concept node
# 3. extract a relation graph from RDF graph, i.e., relation graph has edges corresponding to subClassOf and various relations
# 4. extract label embeddings as node features. When extracting label embeddings, first check whether the label is in the pre-trained model, if it is not, break the label into individual words then take the average of the individual word embeddings.
# 5. build a StellarGraph from the nodes and edges

# embed a clndLabel
def emb_concept_clndLabel(clndLabel, embs_model):
    """
        input: clndLabel: a cleaned concept label string by " ".join(clean_document_lower(x))
               embs_mode: pre-trained word embedding model
        output: a d-dimensional embedding of the clndLabel
    """
  
    if clndLabel in embs_model.wv.key_to_index:
        
        return embs_model.wv[clndLabel]
    
    else:
        embs = []
        for z in clndLabel.split():
            embs.append(embs_model.wv[z])
    
        return np.array(embs).mean(0)
    

# GNN embedding on an ontology graph
def gnn_embeddings_infomax(stellargraph, emb_dims, activations, epochs):
    """
        input: stellargraph, a stellarGraph created from an ontology graph
               emb_dims: the list of dimensions of the base GCN layers
               activations: the list of activtions corresponding to the GCN layers
               epoch: number of training epochs
        output: a DataFrame with the index corresponding to stellargraph.nodes and other columns corresponding to embeddings
    """
    fullbatch_generator = FullBatchNodeGenerator(stellargraph, sparse=False)
    
    # base GCN model with one layer in emb_dim
    gcn_model = GCN(layer_sizes=emb_dims, activations=activations, generator=fullbatch_generator)
    
    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(stellargraph.nodes())
    
    infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()
    
    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(learning_rate=1e-3))
    
    logging.info("Fitting the GNN on the ontology graph...")
    es=EarlyStopping(monitor='loss', min_delta=0, patience=20)
    history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
    
    x_emb_in, x_emb_out = gcn_model.in_out_tensors()
    x_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = Model(inputs=x_emb_in, outputs=x_out)
    
    all_embeddings = emb_model.predict(fullbatch_generator.flow(stellargraph.nodes()))
    
    all_embeddings_df = pd.DataFrame(all_embeddings, index=stellargraph.nodes())
    
    return all_embeddings_df    
    

# make a DataFrame of concept nodes with embeddings as features; each node identified by its uri
def node_features_from_clndLabel_embeddings(label_clnd_uris, embs_model, dim):
    """
        input: label_clnd_uri: DataFrame 'label', 'uri', 'clndLabel'
               embs_model: pre-trained embedding model
               dim: the dimension of the word embeddings
        output: DataFrame with index=uri and features 'd1', 'd2', ...'d300'
    """
    
    # for each clndLabel, retrieve a dim dimensional embedding.
    # the apply() gets a Series of embedding arrays
    embs_array = label_clnd_uris.clndLabel.apply(lambda x: emb_concept_clndLabel(x, embs_model))
    
    # make the column names for the dimensions of an embedding
    embs_cols = ['d' + str(i) for i in np.arange(dim)]

    # conver the Series of embedding arrays to a DataFrame;
    # assign the column names to the embedding dimensions
    embs_df = pd.DataFrame(dict(zip(embs_array.index, embs_array.values)), index = embs_cols).T
    
    # concanate the embeddings to ['label', 'uri', 'clndLabel']
    label_clnd_uri_embs = pd.concat([label_clnd_uris, embs_df], axis = 1)
    
    # drop the ['label', 'clndLabel'] columns and set the uri as index
    return label_clnd_uri_embs.drop(['label', 'clndLabel'], axis=1).set_index('uri')



# build a stellargraph with node label embeddings from an RDF graph and pre-trained embedding model
def stellar_from_rdf(rdfGraph, embs_model, dim):
    """
        input: rdfGraph: an RDF graph parsed from an owl file by rdflib
               embs_model: pre-trained word embedding model
               dim: embedding dimension
        output: a StellarGraph with node features correpsonding to the label embeddings in the RDF graph
    """
    
    # build relation graph from the RDF graph
    relGraph = maponto.build_relation_graph(rdfGraph)
    
    # extract embeddings for concept nodes
    label_uris = maponto.extract_label_uris(rdfGraph)
    label_clnd_uris = maponto.clean_labels(label_uris)
    concept_node_embeddings = node_features_from_clndLabel_embeddings(label_clnd_uris, embs_model, dim)
    
    #get the prefix of the RDF
    prefix = maponto.get_prefix_rdfgraph(rdfGraph).toPython()
    
    if(prefix == 'http://mouse.owl#'):
        # remove the edges to http://www.w3.org/2002/07/owl#Thing
        relGraph = relGraph[relGraph.object.str.contains('http://www.w3.org/2002/07/owl#Thing') == False]
    if(prefix == 'http://human.owl#'):
        # remove the edges to Anatomic_Structure_System_or_Substance http://human.owl#NCI_C12219
        relGraph = relGraph[relGraph.object.str.contains('http://human.owl#NCI_C12219')]
    
    stellarG = StellarGraph(concept_node_embeddings, relGraph, source_column='subject', target_column='object', 
                            node_type_default='concept', edge_type_column='predicate')

    return stellarG


# build a stellargraph without node label embeddings from an RDF graph
def stellar_from_rdf_edges(rdfGraph):
    """
        input: rdfGraph: an RDF graph parsed from an owl file by rdflib
        output: a StellarGraph corresponding to the relation edges in the RDF graph
    """
    
    # build relation graph from the RDF graph
    relGraph = maponto.build_relation_graph(rdfGraph)
    
    stellarG = StellarGraph(edges=relGraph, source_column='subject', target_column='object', 
                            node_type_default='concept', edge_type_column='predicate')

    return stellarG



# # Apply GNN to the StellarGraphs of the Ontologies to Get Concept Node Embeddings



# # Interva Labeling and Retrieving Nodes between Mappings
