#Ontology Mapping by Looking at Neighborhood of a Concept and Computing Wasserstein Distance or Using Topological Analysis 

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

import OTMapOnto as maponto

from gtda.homology import VietorisRipsPersistence

from tqdm import tqdm
from scipy.stats.morestats import ansari

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("logging info test")


def get_relatedWords_counts(topic, uri, label, clndLabel, rdfgraph, label_clnd_uris):
    """
        input: topic: a string representing the topic of the rdfgraph (i.e., the ontology)
               uri: an ontology uri
               label: the label corresponding to the uri
               clndLabel: cleaned label corresponding to the label
               rdfgraph: the RDF graph of the ontology graph
               label_clnd_uris: a DataFrame containing uri and label information 'uri', 'clndLabel', 'label'
        output: dictionary of related words and counts for the clndLabel corresponding to the uri,
                related words including all super classes and synonyms,
                {relatedWord1:count1, relatedWord2:count2, .....}
    
    """
    
    # always add the topic of the ontology to the set of related words
    words = defaultdict(int)
    if topic != None:
        words[topic.lower()] = words[topic.lower()] + 1

    # add the label to the set of related words
    #words[label.lower()] = words[label.lower()] + 1
    
    
    # get all superclasses in rdfs:subClassOf hierarchy
    uri_id = uri
    idx_sharp = uri.rfind("#")
    if idx_sharp > 0:
        uri_id = uri[idx_sharp + 1:]
    else:
        idx_slash = uri.rfind("/")
        if idx_slash > 0:
            uri_id = uri[idx_slash + 1:]
    q = "select ?super where { :" + uri_id + "(rdfs:subClassOf)+ ?super . filter(!isBlank(?super)) . \
        filter(?super != owl:Thing) \
    }"
    
    try:
        res = rdfgraph.query(q)
        for item in res:
            super_uri = item[0].toPython()
            super_label = label_clnd_uris[label_clnd_uris.uri == super_uri].clndLabel.tolist()[0]
            words[super_label] = words[super_label] + 1
    except:
        print("OTNeighborhood_TDA.get_relatedWords_count(): res=rdfgraph.query(q)")
    
    for syn in wordnet.synsets(label):
        for synw in syn.lemma_names():
            words[synw.lower()] = words[synw.lower()] + 1
    
    
    
    for tok in clndLabel.split():
        words[tok] = words[tok] + 1
        for syn in wordnet.synsets(tok):
            for synw in syn.lemma_names():
                words[synw.lower()] = words[synw.lower()] + 1
    
    
    return words


# obtain phrases as combined synonyms from a phrase with multiple worlds
def get_syn_phrases(phrase):
    """
        input: phrase: a string with multiple words
        output: a list of phrases consisting of synonyms
    """
    ans = []
    temp = []
    
    words = phrase.split()
    
    for word in words:
        synonyms = []
        for syn in wordnet.synsets(word):
            for synw in syn.lemma_names():
                if synw not in synonyms:
                    synonyms.append(synw)
        if len(ans) == 0:
            ans.extend(synonyms)
        else:
            temp.clear()
            for curr in ans:
                for w in synonyms:
                    temp.append(curr + " " + w)    
    
        ans.clear()
        ans.extend(temp)
        
    return ans

# compute the Wasserstein distance between two uris based on their relatedWords and counts
def compute_Wasserstein_distance(srelatedWords_counts, trelatedWords_counts, embs_model):
    """
        input: srelatedWords_counts: a dictionary for source words to their counts
               trelatedWords_counts: a dictionary for target words to their counts
               embs_model: a pre-trained embedding model
        output: a Wasserstein distance between the set of source embeddings to the set of target embeddings
    """
    src_embs = []
    src_counts = []
    for sw in srelatedWords_counts:
        src_embs.append(embs_model.wv[sw])
        src_counts.append(srelatedWords_counts[sw])
        
    tgrt_embs = []
    tgrt_counts = []
    for tw in trelatedWords_counts:
        tgrt_embs.append(embs_model.wv[tw])
        tgrt_counts.append(trelatedWords_counts[tw])
        
        
    src_distr = np.array(src_counts) / np.array(src_counts).sum()
    tgrt_distr = np.array(tgrt_counts) / np.array(tgrt_counts).sum()

    costs = sp.spatial.distance.cdist(src_embs, tgrt_embs)
    costs_norm = costs/costs.max()

    wsd = ot.emd2(src_distr, tgrt_distr, costs_norm)
            
    return wsd



# compute the pairwise Wasserstein distances between the list of source uris and the list of 
# target uris
def compute_pairwise_Wasserstein_distances(suris, suris_relatedWords_counts, turis, turis_relatedWords_counts, embs_model):
    """
        input: suris: a list of source uris
               suri_relatedWords_counts: a dictionary from source uris to source words and their counts
               turis: a list of target uris
               turi_relatedWords_counts: a dictionary from target uris to target words and their counts
               embs_model: a pre-trained embedding model
        output: an ndarray of Wasserstein distances between the suris and turis
    """
    logging.info("Compute Pairwise Wasserstein Distances...")
    wds_lists = []
    
    if len(suris_relatedWords_counts) == 0 or len(turis_relatedWords_counts) == 0:
        return np.array(wds_lists)
    
    for suri in tqdm(suris):
        wds = []
        for turi in turis:
            wds.append(compute_Wasserstein_distance(suris_relatedWords_counts[suri],                                                     turis_relatedWords_counts[turi], embs_model))
        
        wds_lists.append(wds)
    
    return np.array(wds_lists)



# compute the jaccard distance (1-jaccard similarity) between two uris based on their relatedWords and counts
def compute_Jaccard_distance(srelatedWords_counts, trelatedWords_counts):
    """
        input: srelatedWords_counts: a dictionary from the related words for a source concept to their counts
               trelatedWords_counts: a dictionary from the related words for a target concept to their counts
        output: a Jaccard distance between the set of source related words to the set of target related words.
    """
    
    source_words = srelatedWords_counts.keys()
    target_words = trelatedWords_counts.keys()
            
    return 1 - jaccard_similarity(source_words, target_words)


# compute the pairwise Jaccard distances between the list of source uris and the list of 
# target uris
def compute_pairwise_Jaccard_distances(suris, suris_relatedWords_counts, turis, turis_relatedWords_counts):
    """
        input: suris: a list of source uris
               suri_relatedWords_counts: a dictionary from source uris to source words and their counts
               turis: a list of target uris
               turi_relatedWords_counts: a dictionary from target uris to target words and their counts
        output: an ndarray of Jaccard distances between the suris and turis
    """
    
    jds_lists = []
    
    if len(suris_relatedWords_counts) == 0 or len(turis_relatedWords_counts) == 0:
        return np.array(jds_lists)
    
    for suri in tqdm(suris):
        jds = []
        for turi in turis:
            jds.append(compute_Jaccard_distance(suris_relatedWords_counts[suri], \
                                                    turis_relatedWords_counts[turi]))
        
        jds_lists.append(jds)
    
    return np.array(jds_lists)


# use mutual minimum WD
def make_mappings_distance_nn(slabel_clnd_uris, tlabel_clnd_uris, distance_arr, row_min_mean=None):
    """
        input: slabel_clnd_uris: source DataFrame with columns 'label', 'uri', 'clndLabel'
               tlabel_clnd_uris: target DataFrame with columns 'label', 'uri', 'clndLabel'
               distance_arr: a distance matrix (ndarray)
               row_min_mean: the mean of mininums of all rows
        output: Mapping DataFrame 'source', 'source_label', 'target', 'targt_label'
    """
    if row_min_mean == None:
        row_min_mean = distance_arr.min(1).mean()
        
    row_list = []
    for i in np.arange(distance_arr.shape[0]):
        if distance_arr[i].min() < row_min_mean:
            row_list.append(i)


    best_match_src = distance_arr.argmin(1)
    best_match_trg = distance_arr.argmin(0)
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
        suri = slabel_clnd_uris.iloc[row].uri
        slabel = slabel_clnd_uris.iloc[row].label
        turi = tlabel_clnd_uris.iloc[col].uri
        tlabel = tlabel_clnd_uris.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))

    return pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])



# use minimum WD
def make_mappings_distance_min(slabel_clnd_uris, tlabel_clnd_uris, distance_arr):
    best_match_src = distance_arr.argmin(1)
    mut_nn = []
    for row, col in enumerate(best_match_src):
        mut_nn.append((row, col))

    alignments = []
    for row, col in mut_nn:
        suri = slabel_clnd_uris.iloc[row].uri
        slabel = slabel_clnd_uris.iloc[row].label
        turi = tlabel_clnd_uris.iloc[col].uri
        tlabel = tlabel_clnd_uris.iloc[col].label
        alignments.append((suri,slabel, turi, tlabel))

    return pd.DataFrame(alignments, columns=['source', 'source_label', 'target', 'target_label'])


# compute PH diagrams from the embeddings of related words for a list or uris
def compute_phDiagrams(label_clnd_uris, uris_relatedWords, embs_model):
    """
    
    """
    diagrams_list = []

    uris = list(label_clnd_uris.uri)
    
    for uri in uris:
        related_embeddings = []
        acloud = []
        for aw in uris_relatedWords[uri]:
            related_embeddings.append(embs_model.wv[aw])
        embs_arr = np.asarray(related_embeddings)
        acloud.append(embs_arr)

        VR = VietorisRipsPersistence(homology_dimensions=[0])  # Parameter explained in the text
        diagrams = VR.fit_transform(acloud)
        diagrams_list.append(diagrams[0])
    
    return diagrams_list


# Compute the Pairwise Wasserstein Distances between PH Diagrams of Source and Target Concepts
def compute_pairwise_wd_phDiagrams(sdiagrams, tdiagrams):
    """
        input: sdiagrams: PH diagrams of source concepts
               tdiagrams: PH diagrams of target concepts
        output: ndarray of WD between the source and traget diagrams
    """
    wds = []
    for sdiagram in tqdm(sdiagrams):
        swds = []
        for tdiagram in tdiagrams:
            costs = sp.spatial.distance.cdist(sdiagram[:, [0,1]], tdiagram[:, [0, 1]])
            costs_norm = costs/costs.max()
            lmv = sdiagram.shape[0]
            lnv = tdiagram.shape[0]
            a = np.ones((lmv,)) / lmv
            b = np.ones((lnv,)) / lnv
            wd = ot.emd2(a, b, costs_norm)
            swds.append(wd)
        wds.append(swds)
    return np.asarray(wds)


# extract the final label from a uri in an ontology
def extract_label_from_uri(uri):
    '''
        input: uri: a uri in an ontology
        output: label: a string at the end of the uri as the label
    '''
    idx_sharp = uri.rfind("#")
    if idx_sharp > 0:
        return  uri[idx_sharp + 1:]
    else:
        idx_slash = uri.rfind("/")
        if idx_slash > 0:
            return uri[idx_slash + 1:]
    
    return uri   


# Extract ObjectProperty labels to URIs;
def extract_objectProperty_uris(graph):
    """
        input: graph: an RDF graph parsed from an owl file by rdflib
        output: a DataFrame of ObjectProperty labels and uris ['label', 'uri']
    """
    def_query = '''SELECT ?p
                   WHERE {
                   ?p a owl:ObjectProperty .
                   }
                   ORDER BY ?p'''
    q_res = graph.query(def_query)

    labels = []
    uris = []
    
    for res in q_res:
        uri = res[0].toPython()
        
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
    
    return pd.DataFrame({'label': labels, 'uri':uris})


# Extract ObjectProperty labels including domain and range labels to URIs;
def extract_objectProperty_domain_range_uris(graph):
    """
        input: graph: an RDF graph parsed from an owl file by rdflib
        output: a DataFrame of ObjectProperty+domain+range labels and uris ['label', 'uri']
    """
    
    def_query = '''SELECT ?d ?p ?r
                   {
                   ?p a owl:ObjectProperty .
                   ?p rdfs:domain ?d .
                   ?p rdfs:range ?r .
                   filter(!isBlank(?d)) .
                   filter(!isBlank(?p)) .
                   filter(!isBlank(?r))
                   }
                   ORDER BY ?p
            '''
    res = graph.query(def_query)

    labels = []
    uris = []

    for res in res:
        stat = ""
        domain = res[0].toPython()
        stat += extract_label_from_uri(domain) + " "

        objproperty = res[1].toPython()
        stat += extract_label_from_uri(objproperty) + " "

        range = res[2].toPython()
        stat += extract_label_from_uri(range)

        labels.append(stat.strip())
        uris.append(objproperty)
    
    return pd.DataFrame({'label': labels, 'uri':uris})


# Extract DatatypeProperty labels to URIs;
def extract_datatypeProperty_uris(graph):
    """
        input: graph: an RDF graph parsed from an owl file by rdflib
        output: a DataFrame of DatatpyeProperty labels and uris ['label', 'uri']
    """
    def_query = '''SELECT ?p
                   WHERE {
                   ?p a owl:DatatypeProperty .
                   }
                   ORDER BY ?p'''
    q_res = graph.query(def_query)

    labels = []
    uris = []
    
    for res in q_res:
        uri = res[0].toPython()
        
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
        
    return pd.DataFrame({'label': labels, 'uri':uris})


# Extract DatatypeProperty labels and domain labels to URIs;
def extract_datatypeProperty_domain_uris(graph):
    """
        input: graph: an RDF graph parsed from an owl file by rdflib
        output: a DataFrame of DatatpyeProperty+domain labels and uris ['label', 'uri']
    """
    
    def_query = '''SELECT ?d ?p
                   {
                   ?p a owl:DatatypeProperty .
                   ?p rdfs:domain ?d .
                   ?p rdfs:range ?r .
                   filter(!isBlank(?d)) .
                   filter(!isBlank(?p)) .
                   filter(!isBlank(?r))
                   }
                   ORDER BY ?p
            '''
    res = graph.query(def_query)

    labels = []
    uris = []

    for res in res:
        stat = ""
        domain = res[0].toPython()
        stat += extract_label_from_uri(domain) + " "

        datatypeproperty = res[1].toPython()
        stat += extract_label_from_uri(datatypeproperty)

        labels.append(stat)
        uris.append(datatypeproperty)
        
    return pd.DataFrame({'label': labels, 'uri':uris})


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

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







