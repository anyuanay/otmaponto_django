#!/usr/bin/env python
# coding: utf-8

# # OTMapper test

# In[295]:


import OTMapOnto as maponto
import OTMapper as otmapper
import OTNeighborhood_TDA as mapneighbor
from rdflib import Graph

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


get_ipython().run_cell_magic('time', '', 'model_path="../model/crawl-300d-2M-subword.bin"\nembs_model = maponto.load_embeddings(model_path, None)')


# In[4]:


mapper = otmapper.Mapper(embs_model)


# # MSE Benchmark

# In[36]:


import importlib
importlib.reload(maponto)
importlib.reload(mapneighbor)
importlib.reload(otmapper)


# In[66]:


source_url = "../data/MSE-Benchmark/testCases/firstTestCase/MaterialInformation-Reduced.owl"
target_url = "../data/MSE-Benchmark/testCases/firstTestCase/MatOnto.owl"
output_url = '../data/results/MSE_firstTest_case_alignments.rdf'
refs_url = "../data/MSE-Benchmark/testCases/firstTestCase/RefAlign1.rdf"


# In[67]:


mapper.align(source_url, target_url, output_url)


# In[68]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# In[69]:


source_url = "../data/MSE-Benchmark/testCases/secondTestCase/MaterialInformation.owl"
target_url = "../data/MSE-Benchmark/testCases/secondTestCase/MatOnto.owl"
output_url = '../data/results/MSE_secondTest_case_alignments.rdf'
refs_url = "../data/MSE-Benchmark/testCases/secondTestCase/RefAlign2.rdf"


# In[70]:


mapper.align(source_url, target_url, output_url)


# In[46]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# In[43]:


source_url = "../data/MSE-Benchmark/testCases/thirdTestCase/MaterialInformation.owl"
target_url = "../data/MSE-Benchmark/testCases/thirdTestCase/EMMO.owl"
output_url = '../data/results/MSE_thirdTest_case_alignments.rdf'
refs_url = "../data/MSE-Benchmark/testCases/thirdTestCase/RefAlign3.rdf"


# In[44]:


mapper.align(source_url, target_url, output_url)


# In[45]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# In[63]:


aligns = maponto.load_alignments(output_url, None)


# In[64]:


aligns.iloc[1]


# In[65]:


refs = maponto.load_alignments(refs_url, None)
refs


# In[58]:


target_graph = Graph()
stack_urls = []
visited_urls = []
stack_urls.append(target_url)
maptonto.parse_owl_withImports(target_graph, stack_urls, visited_urls)
len(target_graph)


# # Retrieve skos:prefLabel

# In[59]:


maponto.extract_label_uris(target_graph)


# In[45]:


#target_url = "../data/anatomy_track/anatomy_track-default/mouse-human-suite/target.rdf"
target_url = "../data/MSE-Benchmark/testCases/thirdTestCase/EMMO.owl"


# In[46]:


target_graph = Graph()
stack_urls = []
stack_urls.append(target_url)
visited_urls = []
maponto.parse_owl_withImports(target_graph, stack_urls, visited_urls)


# In[47]:


len(target_graph)


# In[48]:


from rdflib.namespace import Namespace, NamespaceManager
SKOS_NS = Namespace("http://www.w3.org/2004/02/skos/core#")
target_graph.namespace_manager.bind("skos", SKOS_NS)
    
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
                   
                   ORDER BY ?a
            '''
    
q_res = target_graph.query(def_query)

labels = []
uris = []
for res in q_res:
    if len(str(res[1])) > 0:
        labels.append(str(res[1]))
        uris.append(str(res[0]))


# In[52]:


labels[:5]


# # Anatomy

# In[47]:


source_url = "../data/anatomy_track/anatomy_track-default/mouse-human-suite/source.rdf"
target_url = "../data/anatomy_track/anatomy_track-default/mouse-human-suite/target.rdf"
output_url = '../data/results/mouse_human_alignments.rdf'
refs_url = "../data/anatomy_track/anatomy_track-default/mouse-human-suite/reference.rdf"


# In[48]:


mapper.align(source_url, target_url, output_url)


# In[49]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# # Conference

# In[12]:


source_url = "../data/conference/conference-v1/cmt-conference/source.rdf"
target_url = "../data/conference/conference-v1/cmt-conference/target.rdf"
output_url = '../data/results/cmt_conference_alignments.rdf'
refs_url = "../data/conference/conference-v1/cmt-conference/reference.rdf"
mapper.align(source_url, target_url, output_url)
preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# In[14]:


source_graph = Graph()
stack_urls = []
visited_urls = []
stack_urls.append(source_url)
maponto.parse_owl_withImports(source_graph, stack_urls, visited_urls)
#len(source_graph)
target_graph = Graph()
stack_urls = []
visited_urls = []
stack_urls.append(target_url)
maponto.parse_owl_withImports(target_graph, stack_urls, visited_urls)
len(source_graph), len(target_graph)


# In[16]:


source_clndLabel_uris = maponto.clean_labels(maponto.extract_label_uris(source_graph))
source_clndLabel_uris


# In[17]:


target_clndLabel_uris = maponto.clean_labels(maponto.extract_label_uris(target_graph))
target_clndLabel_uris


# In[19]:


refs_df = maponto.load_alignments(refs_url, None)
refs_df


# In[20]:


preds_df = maponto.load_alignments(output_url, None)
preds_df


# In[22]:


relation_graph = maponto.build_relation_graph(source_graph)
relation_graph


# # For a URI in an RDF graph, extract all related classes and properties

# In[182]:


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


# In[185]:


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
    
    q = "select ?o {" + concept + "         rdfs:subClassOf ?o .         filter(!isBlank(?o)) .} "
        
    res = graph.query(q)
    for r in res:
        relation.append("rdfs:subClassOf")
        related.append(r[0].toPython())
        what.append("child")
    return relation, related, what


# In[186]:


query_superClasses("http://cmt#Review", source_graph)


# In[189]:


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
    
    q = "             select ?s             {                 ?s rdfs:subClassOf " + concept + ".                 filter(!isBlank(?s)) .             }         "
    res = graph.query(q)
    for r in res:
        relation.append("superClassOf")
        related.append(r[0].toPython())
        what.append("parent")
    return relation, related, what


# In[196]:


query_subClasses("http://cmt#User", source_graph)


# In[233]:


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
    
    q = "             select ?p2 ?o2             {" + concept + " rdfs:subClassOf ?o1 .                 ?o1 rdf:type owl:Restriction .                  ?o1 owl:onProperty ?p2 .                 ?o1 owl:someValuesFrom ?o2              }         "
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("someValuesFrom")
    return relation, related, what


# In[234]:


query_relations_someValuesFrom("http://conference#Review_expertise", target_graph)


# In[235]:


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
    
    q = "             select ?p2 ?o2             {" + concept + " rdfs:subClassOf ?o1 .                 ?o1 rdf:type owl:Restriction .                 ?o1 owl:onProperty ?p2 .                 ?o1 owl:allValuesFrom ?o2             }"
    
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("allValuesFrom")
    return relation, related, what


# In[236]:


query_relations_allValuesFrom("http://conference#Review", target_graph)


# In[237]:


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
    
    q = "            select ?p2 ?o2            {" +                 concept + " rdfs:subClassOf ?o1 .                 ?o1 rdf:type owl:Restriction .                 ?o1 owl:onProperty ?p2 .                 ?o1 owl:minCardinality ?o2             }         "
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("owl:minCardinality")
    return relation, related, what


# In[239]:


query_relations_minCardinality("http://cmt#Paper", source_graph)


# In[241]:


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
    
    q = "            select ?p2 ?o2             {" +                 concept + " rdfs:subClassOf ?o1 .                 ?o1 rdf:type owl:Restriction .                 ?o1 owl:onProperty ?p2 .                 ?o1 owl:maxCardinality ?o2             }         "
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("owl:maxCardinality")
    return relation, related, what


# In[242]:


query_relations_maxCardinality("http://cmt#Paper", source_graph)


# In[251]:


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
    
    q = "            select ?b ?c             {                 {" + concept + " rdfs:subClassOf ?a .                 ?a rdf:type owl:Restriction .                 ?a owl:onProperty ?b .                 ?b rdf:type owl:ObjectProperty .                 ?b rdfs:range ?c .                 }                 UNION                 {                 ?b rdf:type owl:ObjectProperty .                 ?b rdfs:domain " + concept + " .                 ?b rdfs:range ?c .                 }                 filter(!isBlank(?b)) .                 filter(!isBlank(?c))             }         "
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("domain")
    return relation, related, what


# In[252]:


query_relations_asDomains("http://cmt#Paper", source_graph)


# In[256]:


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
    
    q = "            select ?b ?c             {                 {                 ?b rdf:type owl:DatatypeProperty .                 ?b rdfs:domain " + concept + " .                 ?b rdfs:range ?c .                 }                 UNION                 {?b rdf:type owl:DatatypeProperty .                 ?b rdfs:domain ?o.                 ?o owl:unionOf ?o1 .                 ?o1 (rdf:rest)*/rdf:first " + concept + " .                 ?b rdfs:range ?c .                 }                 filter(!isBlank(?b)) .                 filter(!isBlank(?c)) .             }         "
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("datatype domain")
    return relation, related, what


# In[260]:


query_datatype_properties("http://conference#Person", target_graph)


# In[268]:


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

    q = "            select ?b ?c             {                 {?c rdfs:subClassOf ?a .                 ?a rdf:type owl:Restriction .                 ?a owl:onProperty ?b .                 ?b rdf:type owl:ObjectProperty .                 ?b rdfs:range " + concept + " .                 }                 UNION                 {                 ?b rdf:type owl:ObjectProperty .                 ?b rdfs:range " + concept + " .                 ?b rdfs:domain ?c .                 }                 filter(!isBlank(?b)) .                 filter(!isBlank(?c))             }         "
    res = graph.query(q)
    for r in res:
        relation.append(r[0].toPython())
        related.append(r[1].toPython())
        what.append("range")
    return relation, related, what


# In[269]:


query_relations_asRanges("http://cmt#Paper", source_graph)


# In[272]:


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

    q = "             select ?o             {" +                 concept + " rdfs:comment ?o .                 filter(!isBlank(?o)) .             }         "
        
    res = graph.query(q)
    for r in res:
        relation.append("rdfs:comment")
        related.append(r[0].toPython())
        what.append("comment")
    return relation, related, what


# In[274]:


query_comments("http://cmt#Meta-Reviewer", source_graph)


# In[277]:


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

    q = "            select ?o2             {                 {" +                  concept + " owl:equivalentClass ?o2 .                 }                 UNION                 {" + concept + " owl:equivalentClass ?o .                 ?o owl:unionOf ?o1 .                 ?o1 (rdf:rest)*/rdf:first ?o2 .                 }                 filter(!isBlank(?o2)) .             }         "
    res = graph.query(q)
    for r in res:
        relation.append("owl:equivalentClass")
        related.append(r[0].toPython())
        what.append("equivalent")
    return relation, related, what


# In[278]:


query_equivalent_classes("http://cmt#Chairman", source_graph)


# In[286]:


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

    q = "             select ?o2             {                 {" +                 concept + " owl:disjointWith ?o2 .                 }                 UNION                 {" + concept + " owl:disjointWith ?o .                 ?o owl:unionOf ?o1 .                 ?o1 (rdf:rest)*/rdf:first ?o2 .                 }                 filter(!isBlank(?o2)) .             }         "
    res = graph.query(q)
    for r in res:
        relation.append("owl:disjointWith")
        related.append(r[0].toPython())
        what.append("disjoint")
    return relation, related, what


# In[310]:


query_disjoint_classes("http://cmt#Preference", source_graph)


# In[311]:


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


# In[320]:


query_related_information("http://cmt#SubjectArea", source_graph)


# In[319]:


query_related_information("http://conference#Topic", target_graph)


# # List Variables and Sizes

# In[10]:


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


# 

# # =================================
# ## Other Tracks
# # =================================

# # IIMB Benchmark

# In[10]:


source_url = "../data/iimb/v1/072/source.rdf"
target_url = "../data/iimb/v1/072/target.rdf"
output_url = '../data/results/iimb_source_target_072_alignment.rdf'
refs_url = "../data/iimb/v1/072/reference.rdf"


# In[11]:


mapper.align(source_url, target_url, output_url)


# In[12]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# # Knowledge Graph

# In[51]:


source_url = "../data/knowledgegraph/commonkg/nell-dbpedia/source.rdf"
target_url = "../data/knowledgegraph/commonkg/nell-dbpedia/target.rdf"
output_url = '../data/results/knowledgegraph_commonkg_alignment.rdf'
refs_url = "../data/knowledgegraph/commonkg/nell-dbpedia/reference.rdf"
#source_url = "../data/knowledgegraph/small-test/test-small/source.rdf"
#target_url = "../data/knowledgegraph/small-test/test-small/target.rdf"
#output_url = '../data/results/knowledgegraph_small_test_alignment.rdf'
#refs_url = "../data/knowledgegraph/small-test/test-small/reference.rdf"


# In[52]:


mapper.align(source_url, target_url, output_url)


# In[53]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# In[4]:


source_url = "../data/knowledgegraph/v3/ontologies/starwars.rdf"
target_url = "../data/knowledgegraph/v3/ontologies/swg.rdf"
output_url = '../data/results/knowledgegraph_starwars_swg_alignment.rdf'
refs_url = "../data/knowledgegraph/v3/references/starwars-swg.rdf"


# In[5]:


mapper.align(source_url, target_url, output_url)


# In[6]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# # Biodiv

# In[54]:


source_url = "../data/biodiv/2018/envo-sweet/source.rdf"
target_url = "../data/biodiv/2018/envo-sweet/target.rdf"
output_url = '../data/results/biodiv_2018_envo_sweet_alignment.rdf'
refs_url = "../data/biodiv/2018/envo-sweet/reference.rdf"


# In[55]:


mapper.align(source_url, target_url, output_url)


# In[56]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# In[57]:


source_url = "../data/biodiv/2018/flopo-pto/source.rdf"
target_url = "../data/biodiv/2018/flopo-pto/target.rdf"
output_url = '../data/results/biodiv_2018_flopo_pto_alignment.rdf'
refs_url = "../data/biodiv/2018/flopo-pto/reference.rdf"


# In[58]:


mapper.align(source_url, target_url, output_url)


# In[59]:


preds = maponto.load_alignments(output_url, None)
maponto.evaluate(preds, refs_url)


# # largebio

# In[60]:


source_url = "../data/largebio/largebio-all_tasks_2016/small-fma-nci/source.rdf"
target_url = "../data/largebio/largebio-all_tasks_2016/small-fma-nci/target.rdf"
output_url = '../data/results/largebio_small_fma_nci_alignment.rdf'
refs_url = "../data/largebio/largebio-all_tasks_2016/small-fma-nci/reference.rdf"


# In[378]:


source_graph = Graph().parse(source_url)
target_graph = Graph().parse(target_url)


# In[379]:


slabel_clnd_uris = maponto.clean_labels(maponto.extract_label_uris(source_graph))
tlabel_clnd_uris = maponto.clean_labels(maponto.extract_label_uris(target_graph))
slabel_clnd_uris.shape, tlabel_clnd_uris.shape


# In[404]:


tlabel_clnd_uris = tlabel_clnd_uris[tlabel_clnd_uris.clndLabel != '']


# In[405]:


sembs = []
tembs = []
for slabel in slabel_clnd_uris.clndLabel:
    sembs.append(maponto.average_embeddings(slabel, 300, embs_model))
for tlabel in tlabel_clnd_uris.clndLabel:
    tembs.append(maponto.average_embeddings(tlabel, 300, embs_model))


# In[406]:


len(sembs), len(tembs)


# In[407]:


for i, emb in enumerate(tembs):
    try:
        if len(emb) != 300:
            print(emb)
    except:
        print(i, emb)


# In[409]:


maponto.match_label_embeddings_OT(slabel_clnd_uris, tlabel_clnd_uris, embs_model,                         maponto.make_mappings_nn, None, None)


# In[415]:


importlib.reload(maponto)


# In[61]:


mapper.align(source_url, target_url, output_url)


# In[62]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# In[63]:


source_url = "../data/largebio/largebio-all_tasks_2016/small-fma-snomed/source.rdf"
target_url = "../data/largebio/largebio-all_tasks_2016/small-fma-snomed/target.rdf"
output_url = '../data/results/largebio_small_fma_snomed_alignment.rdf'
refs_url = "../data/largebio/largebio-all_tasks_2016/small-fma-snomed/reference.rdf"


# In[64]:


mapper.align(source_url, target_url, output_url)


# In[65]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# In[421]:


source_url = "../data/largebio/largebio-all_tasks_2016/small-snomed-nci/source.rdf"
target_url = "../data/largebio/largebio-all_tasks_2016/small-snomed-nci/target.rdf"
output_url = '../data/results/largebio_small_snomed_nci_alignment.rdf'
refs_url = "../data/largebio/largebio-all_tasks_2016/small-snomed-nci/reference.rdf"


# In[ ]:


# Kernel died on running this
#mapper.align(source_url, target_url, output_url)


# In[ ]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# # Phenotype

# In[66]:


source_url = "../data/phenotype/phenotype-doid-ordo-2017-bioportal/doid-ordo/source.rdf"
target_url = "../data/phenotype/phenotype-doid-ordo-2017-bioportal/doid-ordo/target.rdf"
output_url = '../data/results/phenotye_doid_ordo_2017_alignment.rdf'
refs_url = "../data/phenotype/phenotype-doid-ordo-2017-bioportal/doid-ordo/reference.rdf"


# In[67]:


mapper.align(source_url, target_url, output_url)


# In[68]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# In[9]:


source_graph = Graph().parse(source_url)
target_graph = Graph().parse(target_url)
slabel_clnd_uris = maponto.clean_labels(maponto.extract_label_uris(source_graph))
tlabel_clnd_uris = maponto.clean_labels(maponto.extract_label_uris(target_graph))


# In[10]:


maponto.evaluate(maponto.match_label_synonyms(slabel_clnd_uris, tlabel_clnd_uris, None), 
                 refs_url)


# In[40]:


source_url = "../data/phenotype/phenotype-hp-mp-2017-bioportal/hp-mp/source.rdf"
target_url = "../data/phenotype/phenotype-hp-mp-2017-bioportal/hp-mp/target.rdf"
output_url = '../data/results/phenotye_hp_mp_2017_alignment.rdf'
refs_url = "../data/phenotype/phenotype-hp-mp-2017-bioportal/hp-mp/reference.rdf"


# In[41]:


mapper.align(source_url, target_url, output_url)


# In[42]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# # Geolink

# In[14]:


source_url = "../data/geolink/geolink-v1/gbo-gmo/source.rdf"
target_url = "../data/geolink/geolink-v1/gbo-gmo/target.rdf"
output_url = '../data/results/geolink_gbo_gmo_alignment.rdf'
refs_url = "../data/geolink/geolink-v1/gbo-gmo/reference.rdf"


# In[15]:


mapper.align(source_url, target_url, output_url)


# In[ ]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# # Hydrography

# In[16]:


source_url = "../data/hydrography/hydrography-v1/cree-swo/source.rdf"
target_url = "../data/hydrography/hydrography-v1/cree-swo/target.rdf"
output_url = '../data/results/hydrograph_cree_swo_alignment.rdf'
refs_url = "../data/hydrography/hydrography-v1/cree-swo/reference.rdf"


# In[17]:


mapper.align(source_url, target_url, output_url)


# In[ ]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# In[ ]:





# In[18]:


source_url = "../data/hydrography/hydrography-v1/hydro3-swo/source.rdf"
target_url = "../data/hydrography/hydrography-v1/hydro3-swo/target.rdf"
output_url = '../data/results/hydrograph_hydro3_swo_alignment.rdf'
refs_url = "../data/hydrography/hydrography-v1/hydro3-swo/reference.rdf"


# In[19]:


mapper.align(source_url, target_url, output_url)


# In[ ]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# # Popenslaved

# In[35]:


source_url = "../data/popenslaved/popenslaved-v1/enslaved-wikidata/source.rdf"
target_url = "../data/popenslaved/popenslaved-v1/enslaved-wikidata/target.rdf"
output_url = '../data/results/popenslaved_enslaved_wikidata_alignment.rdf'
refs_url = "../data/popenslaved/popenslaved-v1/enslaved-wikidata/reference.rdf"


# In[36]:


mapper.align(source_url, target_url, output_url)


# In[37]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# # Popgeolink

# In[23]:


source_url = "../data/popgeolink/popgeolink-v1/popgbo-popgmo/source.rdf"
target_url = "../data/popgeolink/popgeolink-v1/popgbo-popgmo/target.rdf"
output_url = '../data/results/popgeolink_popgbo_popgmo_alignment.rdf'
refs_url = "../data/popgeolink/popgeolink-v1/popgbo-popgmo/reference.rdf"


# In[24]:


mapper.align(source_url, target_url, output_url)


# In[ ]:


preds = maponto.load_alignments(output_url, 'predicted')
maponto.evaluate(preds, refs_url)


# In[ ]:




