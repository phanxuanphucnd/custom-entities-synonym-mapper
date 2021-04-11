import os
import warnings
import logging
import re
import operator

import rasa.utils.io

from typing import Any, Dict, Optional, Text

from rasa.shared.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import TrainingData, Message
from rasa.nlu.utils import write_json_to_file
from rasa.shared.utils.io import read_json_file, raise_warning

from azir.contants import *
from azir.utils import get_max_fuzzy_score, fuzz_ratio, fuzz_partial_ratio
from pprint import pformat, pprint

logger = logging.getLogger(__name__)



class AzirEntitySynonymMapper(EntityExtractor):
    """A Custom Entity Synonyms Mapper component. 
    
    Azir was a mortal emperor of Shurima in a far distant age, a proud man who stood at the cusp of immortality. 
    His hubris saw him betrayed and murdered at the moment of his greatest triumph, but now, millennia later, he 
    has been reborn as an Ascended being of immense power. With his buried city risen from the sand, Azir seeks 
    to restore Shurima to its former glory.
    """

    name = 'AzirEntitySynonymMapper'

    defaults = {
        LIST_DEFINED_ENTITIES: [
            'inform#object_type', 'ask_confirm#object_type', 'ask_availability#object_type', 'deny#object_type', 
            'inform#location'
        ], 
        GENERAL_OBJECT_TYPES: ['ao', 'bo', 'set', 'combo', 'vay', 'quan'],
        IGNORE_VALUES: [],
        THRESHOLD: 0.80,
        NUM_CANDIDATES: 5, 
    }

    provides = ["entities"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> None:
        super(AzirEntitySynonymMapper, self).__init__(component_config)

        self.synonyms = synonyms if synonyms else {}
        
        synonyms_reversed = {}
        for k, v in self.synonyms.items():
            if not v in self.component_config[IGNORE_VALUES]:
                if v in synonyms_reversed.keys():
                    synonyms_reversed[v].append(k)
                else:
                    synonyms_reversed[v] = [k]

        for i in synonyms_reversed.keys():
            synonyms_reversed[i].append(re.sub("_", " ", i))


        self.synonyms_reversed = synonyms_reversed

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:


        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.data.get("entities", []):
                entity_val = example.data.get('text')[entity["start"]: entity["end"]]
                self.add_entities_if_synonyms(
                    entity_val, str(entity.get("value")))

        synonyms_reversed = {}
        for k, v in self.synonyms.items():
            if not v in self.component_config[IGNORE_VALUES]:
                if v in synonyms_reversed.keys():
                    synonyms_reversed[v].append(k)
                else:
                    synonyms_reversed[v] = [k]

        for i in synonyms_reversed.keys():
            synonyms_reversed[i].append(re.sub("_", " ", i))

        self.synonyms_reversed = synonyms_reversed

    def process(self, message: Message, **kwargs: Any) -> None:

        logger.debug(f"On process...")
        if self.synonyms:
            updated_entities = message.get("entities", [])[:]
            self.replace_synonyms(updated_entities)
            message.set("entities", updated_entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:

        if self.synonyms:
            file_name = file_name + ".json"
            entity_synonyms_file = os.path.join(model_dir, file_name)
            write_json_to_file(
                entity_synonyms_file, self.synonyms, separators=(",", ": ")
            )
            return {"file": file_name}
        else:
            return {"file": None}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["AzirEntitySynonymMapper"] = None,
        **kwargs: Any,
    ) -> "AzirEntitySynonymMapper":

        file_name = meta.get("file")
        if not file_name:
            synonyms = None
            return cls(meta, synonyms)

        entity_synonyms_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entity_synonyms_file):
            synonyms = read_json_file(entity_synonyms_file)
        else:
            synonyms = None
            raise_warning(
                f"Failed to load synonyms file from '{entity_synonyms_file}'.",
                docs=DOCS_URL_TRAINING_DATA_NLU + "#entity-synonyms",
            )
        return cls(meta, synonyms)

    def get_candidates(self, entity_value):
        scores = []
        syns = self.synonyms_reversed.keys()

        for syn in syns:
            score = get_max_fuzzy_score(entity_value.lower(), self.synonyms_reversed[syn])
            scores.append([syn, score])

        results = sorted(scores, key=lambda x: x[1], reverse=True)

        return results

    def replace_synonyms(self, entities) -> None:
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])

            if entity_value.lower() in self.synonyms:
                if entity["entity"] in self.component_config[LIST_DEFINED_ENTITIES]:
                    logger.debug(
                        f"Replace {entity['entity']}:{entity['value']} "
                        f"into {entity['entity']}:{self.synonyms[entity_value.lower()]}")

                    entity["value"] = self.synonyms[entity_value.lower()]
                    entity["confidence"] = 1.0

                    candidates = self.get_candidates(entity_value.lower())

                    if entity["value"] in self.component_config[GENERAL_OBJECT_TYPES]:
                        entity["candidates"] = candidates[1:self.component_config[NUM_CANDIDATES]+1]
                    else:
                        entity["candidates"] = candidates[0:self.component_config[NUM_CANDIDATES]]
                    
                    self.add_processor_name(entity)
                
            elif entity["entity"] in self.component_config[LIST_DEFINED_ENTITIES]: 
                candidates = self.get_candidates(entity_value.lower())

                if candidates[0][1] >= self.component_config[THRESHOLD]:
                    logger.debug(
                        f"Replace {entity['entity']}:{entity['value']} "
                        f"into {entity['entity']}:{candidates[0][0]}")

                    entity["value"] = candidates[0][0]
                    entity["confidence"] = candidates[0][1]

                else:
                    logger.debug(
                        f"Replace {entity['entity']}:{entity['value']} "
                        f"into {entity['entity']}:{None}") 

                    entity["value"] = None
                    entity["confidence"] = None


                if entity["value"] in self.component_config[GENERAL_OBJECT_TYPES]:
                    entity["candidates"] = candidates[1:self.component_config[NUM_CANDIDATES]+1]
                else:
                    entity["candidates"] = candidates[0:self.component_config[NUM_CANDIDATES]]
                    
                self.add_processor_name(entity)
                

    def add_entities_if_synonyms(self, entity_a, entity_b) -> None:
        if entity_b is not None:
            original = str(entity_a)
            replacement = str(entity_b)

            if original != replacement:
                original = original.lower()
                if original in self.synonyms and self.synonyms[original] != replacement:
                    raise_warning(
                        f"Found conflicting synonym definitions "
                        f"for {repr(original)}. Overwriting target "
                        f"{repr(self.synonyms[original])} with "
                        f"{repr(replacement)}. "
                        f"Check your training data and remove "
                        f"conflicting synonym definitions to "
                        f"prevent this from happening.",
                        docs=DOCS_URL_TRAINING_DATA_NLU + "#entity-synonyms",
                    )

                self.synonyms[original] = replacement
