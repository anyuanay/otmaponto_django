import scipy as sp
import json

from rdflib import Graph, URIRef, RDFS
from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import load_facebook_vectors


from corpus_build_utils import clean_document_lower
from AlignmentFormat import serialize_mapping_to_tmp_file

import sys
import logging
from collections import defaultdict

import jellyfish
import ot

from xml.dom import minidom
from nltk.corpus import wordnet

import warnings

import OTMapOnto as maponto

