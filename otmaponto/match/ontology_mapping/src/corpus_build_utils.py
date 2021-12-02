import numpy as np
import os
import re
import ssl
import urllib.parse
import urllib.request

from gensim.models import phrases

import nltk
from nltk.corpus import stopwords

import camelsplit

def clean_document(_in_str):
    """
    A function to prepare a document for word vector training.
    :param _in_str: An input document as a string.
    :return: A list of cleaned tokens to represent the document.
    """
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


def corpus_combination(_u, _p, _e, _q, _g, _reverse_definitions):
    _corpus = []
    _reverse_definitions = False
    if _q == "concept":
        corp0 = create_corpus(_u, _p, "ORIGINAL", _reverse_definitions, _e)
        corp1 = create_corpus(_u, _p, "CONCEPT", _reverse_definitions, _e)
        corp2 = create_corpus(_u, _p, "CONCEPT_INDIVIDUAL", _reverse_definitions, _e)
        _corpus = _corpus + corp0 + corp1 + corp2
    elif _q == "synonym":
        corp0 = create_corpus(_u, _p, "ORIGINAL", _reverse_definitions, _e)
        corp1 = create_corpus(_u, _p, "CONCEPT", _reverse_definitions, _e)
        corp2 = create_corpus(_u, _p, "SYNONYM", _reverse_definitions, _e)
        _corpus = _corpus + corp0 + corp1 + corp2
    elif _q == "abbreviation":
        corp0 = create_corpus(_u, _p, "ORIGINAL", _reverse_definitions, _e)
        corp1 = create_corpus(_u, _p, "CONCEPT", _reverse_definitions, _e)
        corp2 = create_corpus(_u, _p, "ABBREVIATION", _reverse_definitions, _e)
        _corpus = _corpus + corp0 + corp1 + corp2
    elif _q == "all":
        corp0 = create_corpus(_u, _p, "ORIGINAL", _reverse_definitions, _e)
        corp1 = create_corpus(_u, _p, "CONCEPT", _reverse_definitions, _e)
        corp2 = create_corpus(_u, _p, "ABBREVIATION", _reverse_definitions, _e)
        corp3 = create_corpus(_u, _p, "SYNONYM", _reverse_definitions, _e)
        print('Synonym query: {f}'.format(f=len(corp3)))
        corp5 = create_corpus(_u, _p, "CONCEPT_INDIVIDUAL", _reverse_definitions, _e)
        _corpus = _corpus + corp0 + corp1 + corp2 + corp3 + corp5
    else:
        print("Invalid query parameters, please retry.")
    if _g == 'bigram':
        phrase = phrases.Phrases(_corpus, min_count=10)
        bigram = phrases.Phraser(phrase)
        _corpus = list(bigram[_corpus])
    elif _g == 'trigram':
        phrase = phrases.Phrases(_corpus, min_count=10)
        bigram = phrases.Phraser(phrase)
        trigram = phrases.Phrases(bigram[_corpus], min_count=2)
        _corpus = list(trigram[bigram[_corpus]]) + _corpus
    elif _g == 'quadgram':
        print('Running quad detection')
        phrase = phrases.Phrases(_corpus, min_count=10)
        bigram = phrases.Phraser(phrase)
        c1 = list(bigram[_corpus])
        trigram = phrases.Phrases(bigram[_corpus], min_count=10)
        c2 = list(trigram[bigram[_corpus]])
        quadgram = phrases.Phrases(trigram[bigram[_corpus]], min_count=10)
        c3 = list(quadgram[trigram[bigram[_corpus]]])
        _corpus = c3 + c2 + c1 + _corpus
    return _corpus


def create_concepts(_user, _pw, _endp, _q):
    """
    :param _user: The database user name.
    :param _pw: The database password.
    :param _endp: The SPARQL endpoint.
    :return: List of concepts in the ontology.
    """
    data = query_ontology(_user, _pw, _q, _endp)
    data['concept'] = data['word'].apply(un_camel)
    data['concept'] = data['concept'].apply(create_term)
    return data['concept'].tolist()


def create_corpus(_user, _pw, _query, _doub, _endp, _logging):
    """"
    A function to query the ontology and convert to clean corpus.
    :param _user: The database user name.
    :param _pw: The database password.
    :param _query: The flag for a particular query, defined in script.
    :param _doub: Binary flag to swap subject-object definitions.
    :param _endp: The SPARQL endpoint.
    :param _logging: To log a document to stdout.
    :return: A corpus that can be fed into an embedding model.
    """
    if _query == "ORIGINAL":
        data = query_ontology(_user, _pw, ORIGINAL_QUERY, _endp)
        data['edge'] = data['term'].apply(find_edges)
        fibo_def = data[data['edge'] == 0]
        fibo_def['term'] = fibo_def['term'].apply(un_camel)
        fibo_def['term'] = fibo_def['term'].apply(create_term)
        fibo_def['document'] = fibo_def['term'] + ' is defined by ' + fibo_def['def']
        fibo_def['doc_out'] = fibo_def['document'].apply(clean_document_lower)
        fibo_def['doc_full'] = fibo_def['document'].apply(clean_document)
        _corpus = fibo_def['doc_out'].tolist() + fibo_def['doc_full'].tolist()
        if _doub:
            fibo_def['document2'] = fibo_def['def'] + ' has definition ' + \
            fibo_def['term']
        fibo_def['doc_list2'] = fibo_def['document2'].apply(clean_document)
        fibo_def['doc_list3'] = fibo_def['document2'].apply(clean_document_lower)
        _corp2 = fibo_def['doc_list2'].tolist() + fibo_def['doc_list3'].tolist()
        _corpus = _corpus + _corp2
    elif _query == "CONCEPT":
        data = query_ontology(_user, _pw, CONCEPT_STATEMENT_QUERY, _endp)
        data['word'] = data['label'].apply(create_term)
        data = data.fillna(value='')
        data['document'] = data['word'] + ' ' + data['text']
        data['doc_out'] = data['document'].apply(clean_document)
        data['doc_out2'] = data['document'].apply(clean_document_lower)
        _corpus = data['doc_out'].tolist() + data['doc_out2'].tolist()
        if _doub:
            data['document2'] = data['text'] + ' defines ' + data['word']
            data['doc_list2'] = data['document2'].apply(clean_document)
            data['doc_list3'] = data['document3'].apply(clean_document)
            _corp2 = data['doc_list2'].tolist() + data['doc_list3'].tolist()
            _corpus = _corpus + _corp2
    elif _query == "CONCEPT2":
        data = query_ontology(_user, _pw, CONCEPT_TEXT_QUERY, _endp)
        data['concept'] = data['word'].apply(un_camel)
        data['concept'] = data['concept'].apply(create_term)
        data['document'] = data['concept'] + ' ' + data['definition']
        data['doc_out'] = data['document'].apply(clean_document)
        data['doc_out2'] = data['document'].apply(clean_document_lower)
        _corpus = data['doc_out'].tolist() + data['doc_out2'].tolist()
        if _doub:
            data['document2'] = data['definition'] + ' defines ' + data['concept']
            data['doc_list2'] = data['document2'].apply(clean_document)
            data['doc_list3'] = data['document2'].apply(clean_document_lower)
            _corp2 = data['doc_list2'].tolist() + data['doc_list3'].tolist()
            _corpus = _corpus + _corp2
    elif _query == "SYNONYM":
        data = query_ontology(_user, _pw, SYNONYM_STATEMENT_QUERY, _endp)
        data['word'] = data['label'].apply(create_term)
        data = data.fillna(value='')
        data['document'] = data['word'] + ' ' + data['text']
        data['doc_out'] = data['document'].apply(clean_document)
        data['doc_out2'] = data['document'].apply(clean_document_lower)
        _corpus = data['doc_out2'].tolist()
        if _doub:
            data['document2'] = data['text'] + ' defines ' + data['word']
            data['doc_list2'] = data['document2'].apply(clean_document)
            data['doc_list3'] = data['document2'].apply(clean_document_lower)
            _corp2 = data['doc_list2'].tolist() + data['doc_list3'].tolist()
            _corpus = _corpus + _corp2
    elif _query == "ABBREVIATION":
        data = query_ontology(_user, _pw, ABBREVIATION_STATEMENT_QUERY, _endp)
        data['word'] = data['label'].apply(create_term)
        data = data.fillna(value='')
        data['document'] = data['word'] + ' ' + data['text']
        data['doc_out'] = data['document'].apply(clean_document)
        data['doc_out2'] = data['document'].apply(clean_document_lower)
        _corpus = data['doc_out'].tolist() + data['doc_out2'].tolist()
        if _doub:
            data['document2'] = data['text'] + ' defines ' + data['word']
            data['doc_list2'] = data['document2'].apply(clean_document)
            data['doc_list3'] = data['document2'].apply(clean_document_lower)
            _corp2 = data['doc_list2'].tolist() + data['doc_list3'].tolist()
            _corpus = _corpus + _corp2
    elif _query == "CONCEPT_INDIVIDUAL":
        data = query_ontology(_user, _pw, CONCEPT_INDIVIDUAL_QUERY, _endp)
        data['concept'] = data['word'].apply(un_camel)
        data['concept'] = data['concept'].apply(create_term)
        data['concept_label'] = data['label'].apply(create_term)
        data['document'] = data['concept'] + ' ' + data['concept_label'] + ' ' + data['definition']
        data['doc_out'] = data['document'].apply(clean_document)
        data['doc_out2'] = data['document'].apply(clean_document_lower)
        _corpus = data['doc_out'].tolist() + data['doc_out2'].tolist()
        if _doub:
            data['document2'] = data['definition'] + ' defines ' + data['concept']
            data['doc_list2'] = data['document2'].apply(clean_document)
            data['doc_list3'] = data['document2'].apply(clean_document_lower)
            _corp2 = data['doc_list2'].tolist( )+ data['doc_list3'].tolist()
            _corpus = _corpus + _corp2
    else:
        print('Invalid query parameter, please try again.')
    if _logging:
        print(_corpus[-1])
    return _corpus


def create_label_concepts(_user, _pw, _endp, _q):
    """
    :param _user: The database user name.
    :param _pw: The database password.
    :param _endp: The SPARQL endpoint.
    :return: List of concepts in the ontology.
    """
    data = query_ontology(_user, _pw, _q, _endp)
    data['concept'] = data['label'].apply(un_camel)
    data['concept'] = data['concept'].apply(create_term)
    return data['concept'].tolist()


def create_term(_str):
    """
    A function to make concept terms a unique token.
    CHANGE LOG 2/18/2020: Adjusted the concept creation to keep abbreviations
    at the head of concepts.
    :param _str: Input string with spaces.
    :return: String with spaces replaced by underscores.
    """
    _str = _str.replace("-", "_")
    _str = _str.replace(" ", "_")
    pttrn = '([A-Z][a-z]+){1}(_[A-Z][a-z]+[0-9]*)*$'
    if re.match(pttrn, _str) and 'California' not in _str:
        _str = _str.lower()
    else:
        str_lst = _str.split("_")
        head = str_lst.pop(0)
        str_lst = [j.lower() for j in str_lst]
        str_lst.insert(0, head)
        _str = "_".join(str_lst)
    _str = _str.lower()
    return _str



def finalize_corpus(_wd, _user, _password, _query, _endp, _web, _gram, _logging):
    corpus = []
    reverse_definitions = False
    if _query == "concept":
        corp0 = create_corpus(_user, _password, "ORIGINAL", reverse_definitions, _endp, _logging)
        corp1 = create_corpus(_user, _password, "CONCEPT", reverse_definitions, _endp, _logging)
        corpus = corp0 + corp1
    elif _query == "synonym":
        corp0 = create_corpus(_user, _password, "ORIGINAL", reverse_definitions, _endp, _logging)
        corp1 = create_corpus(_user, _password, "CONCEPT", reverse_definitions, _endp, _logging)
        corp2 = create_corpus(_user, _password, "SYNONYM", reverse_definitions, _endp, _logging)
        corpus = corp0 + corp1 + corp2
    elif _query == "abbreviation":
        corp0 = create_corpus(_user, _password, "ORIGINAL", reverse_definitions, _endp, _logging)
        corp1 = create_corpus(_user, _password, "CONCEPT", reverse_definitions, _endp, _logging)
        corp2 = create_corpus(_user, _password, "ABBREVIATION",
                              reverse_definitions, _endp)
        corpus = corp0 + corp1 + corp2
    elif _query == "ind":
        corp5 = create_corpus(_user, _password, "CONCEPT_INDIVIDUAL",
                              reverse_definitions, _endp, _logging)
        corpus = corp5
    elif _query == "all":
        corp0 = create_corpus(_user, _password, "ORIGINAL", reverse_definitions, _endp, _logging)
        print('Original corpus: {f}'.format(f=len(corp0)))
        corp1 = create_corpus(_user, _password, "CONCEPT", reverse_definitions, _endp, _logging)
        print('Concept corpus: {f}'.format(f=len(corp1)))
        corp2 = create_corpus(_user, _password, "ABBREVIATION",
                              reverse_definitions, _endp, _logging)
        print('Abbreviation corpus: {f}'.format(f=len(corp2)))
        corp3 = create_corpus(_user, _password, "SYNONYM", reverse_definitions, _endp, _logging)
        print('Synonym corpus: {f}'.format(f=len(corp3)))
        corp5 = create_corpus(_user, _password, "CONCEPT_INDIVIDUAL",
                              reverse_definitions, _endp, _logging)
        print('Concept individual: {f}'.format(f=len(corp5)))
        corpus = corp0 + corp1 + corp2 + corp3 + corp5
        print('Final corpus length: {f}'.format(f=len(corpus)))
    else:
        print("Invalid query parameters, please retry.")

    concepts = create_concepts(_user, _password, _endp, CONCEPT_TEXT_QUERY)
    concepts_named_individuals = create_concepts(_user, _password,
                                                 _endp, CONCEPT_INDIVIDUAL_QUERY)
    concepts = concepts + concepts_named_individuals
    concept_labels = create_label_concepts(_user, _password,
                                           _endp, CONCEPT_LABEL_TEXT_QUERY)
    concepts = concepts + concept_labels
    if _web:
        with open(os.path.join(_wd, "data", "web_corpus.txt"), 'r') as f:
            web_corpus = f.readlines()
        corpus = corpus + web_corpus
        print('Corpus with web data has {d} documents'.format(d=len(corpus)))
    if _gram == 'bigram':
        phrase = phrases.Phrases(corpus, min_count=10)
        bigram = phrases.Phraser(phrase)
        corpus = list(bigram[corpus]) + corpus
    elif _gram == 'trigram':
        phrase = phrases.Phrases(corpus, min_count=10)
        bigram = phrases.Phraser(phrase)
        tg_phrases = phrases.Phrases(bigram[corpus], min_count=2)
        trigram = phrases.Phraser(tg_phrases)
        corpus = list(trigram[bigram[corpus]]) + corpus
    elif _gram == 'quadgram':
        print('Running quad detection')
        phrase = phrases.Phrases(corpus, min_count=5)
        bigram = phrases.Phraser(phrase)
        c1 = list(bigram[corpus])
        trigram = phrases.Phrases(bigram[corpus], min_count=10)
        c2 = list(trigram[bigram[corpus]])
        quadgram = phrases.Phrases(trigram[bigram[corpus]], min_count=5)
        c3 = list(quadgram[trigram[bigram[corpus]]])
        corpus = c3 + c2 + c1 + corpus
    print('Total documents in training set: {d}'.format(d=len(corpus)))
    print('Average document length is {a}'.format(a=np.mean([len(x) for x in corpus])))
    print('Total concepts are {c}'.format(c=len(concepts)))
    with open('concept_keys.txt', 'w') as f:
        for item in concepts:
            f.write("%s\n" % item)
    return corpus, concepts


def find_edges(_x):
    """
    A function to find edges between entities.
    :param _x: Input string to check for edge relationship.
    :return: Binary flag if string contains an edge identifier.
    """
    _y = 0
    if _x.find('has') is not -1:
        _y += 1
    return _y



def query_ontology(_un, _pw, _qs, _endp):
    """
    A function to query the ontology and return financial definitions.
    :param _un: The database username.
    :param _pw: The database password.
    :param _qs: The query type passed from the command line.
    :param _endp: The SPARQL endpoint.
    :return: A data frame of terms and definitions.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    proxy_support = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_support,
                                         urllib.request.HTTPSHandler(context=context))
    urllib.request.install_opener(opener)
    endpoint = RemoteEndpoint(
        _endp,
        user=_un,
        passwd=_pw)
    results = endpoint.select(_qs)
    return results


def un_camel(_x):
    """
    A function to parse camel-case term names.
    :param _x: An input camel-cased string.
    :return: The re-formatted input string.
    """
    return '_'.join(re.sub('(?!^)([A-Z][a-z]+)', r' \1', _x).split())



