"""
This code was written by:
Tomaz Bratanic
https://towardsdatascience.com/extract-knowledge-from-text-end-to-end-information-extraction-pipeline-with-spacy-and-neo4j-502b2b1e0754
https://github.com/tomasonjo/blogs/blob/master/ie_pipeline/SpaCy_informationextraction.ipynb
Accessed: August 2022

referencing model from:
https://github.com/Babelscape/rebel
"""

import pandas as pd
import spacy
import crosslingual_coreference
import requests
import re
import hashlib
from spacy import Language
from typing import List
from spacy.tokens import Doc, Span
from transformers import pipeline


def call_wiki_api(item):
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return 'id-less'

def extract_triplets(text):
    """
    Function to parse the generated text and extract the triplets
    """
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})

    return triplets

@Language.factory(
    "rebel",
    requires=["doc.sents"],
    assigns=["doc._.rel"],
    default_config={
        "model_name": "Babelscape/rebel-large",
        "device": 0,
    },
)
class RebelComponent:
    def __init__(
        self,
        nlp,
        name,
        model_name: str,
        device: int,
    ):
        assert model_name is not None, ""
        self.triplet_extractor = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
        self.entity_mapping = {}
        # Register custom extension on the Doc
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default={})

    def get_wiki_id(self, item: str):
        mapping = self.entity_mapping.get(item)
        if mapping:
            return mapping
        else:
            res = call_wiki_api(item)
            self.entity_mapping[item] = res
            return res
    
    def _generate_triplets(self, sent: Span) -> List[dict]:
        output_ids = self.triplet_extractor(sent.text, return_tensors=True, return_text=False)[0]["generated_token_ids"]["output_ids"]
        extracted_text = self.triplet_extractor.tokenizer.batch_decode(output_ids[0])
        extracted_triplets = extract_triplets(extracted_text[0])
        return extracted_triplets
    
    def get_ner_tags(self, doc: Doc):
        ner_tags = {}
        for ent in doc.ents:
            ner_tags[ent.text] = ent.label_
        return ner_tags
    
    def set_annotations(self, doc: Doc, triplets: List[dict], sent):
        for triplet in triplets:
            
            # Remove self-loops (relationships that start and end at the entity)
            if triplet['head'] == triplet['tail']:
                continue

            # Use regex to search for entities
            head_span = re.search(triplet["head"], doc.text)
            tail_span = re.search(triplet["tail"], doc.text)

            # Skip the relation if both head and tail entities are not present in the text
            # Sometimes the Rebel model hallucinates some entities
            if not head_span or not tail_span:
                continue

            index = hashlib.sha1("".join([triplet['head'], triplet['tail'], triplet['type']]).encode('utf-8')).hexdigest()
            if index not in doc._.rel:
                # Get wiki ids and store results
                doc._.rel[index] = {"sentence": sent.text, "relation": triplet["type"], 
                                    "head_span": {'text': triplet['head'], 'id': self.get_wiki_id(triplet['head'])}, 
                                    "tail_span": {'text': triplet['tail'], 'id': self.get_wiki_id(triplet['tail'])},
                                    "NER_tags": self.get_ner_tags(sent)}

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets, sent)
        return doc
