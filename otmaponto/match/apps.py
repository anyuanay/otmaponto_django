from django.apps import AppConfig

from .ontology_mapping.src import OTMapOnto as maponto
from .ontology_mapping.src import OTMapper as mapper

from django.conf import settings

import os

class MatchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'match'

    model_path=os.path.abspath("./match/ontology_mapping/model/crawl-300d-2M-subword.bin")
    embs_model = maponto.load_embeddings(model_path, None)

    #mapper = mapper.Mapper(settings.EMBS_MODEL)
    mapper = mapper.Mapper(embs_model)
    
