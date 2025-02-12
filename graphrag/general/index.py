#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
from functools import reduce, partial
import networkx as nx
import time
from collections import Counter
import os

from api import settings
from graphrag.general.community_reports_extractor import CommunityReportsExtractor
from graphrag.entity_resolution import EntityResolution
from graphrag.general.extractor import Extractor
from graphrag.general.graph_extractor import DEFAULT_ENTITY_TYPES
from graphrag.utils import graph_merge, set_entity, get_relation, set_relation, get_entity, get_graph, set_graph, \
    chunk_id, update_nodes_pagerank_nhop_neighbour
from rag.nlp import rag_tokenizer, search
from rag.utils.redis_conn import RedisDistributedLock


class Dealer:
    def __init__(self,
                 extractor: Extractor,
                 tenant_id: str,
                 kb_id: str,
                 llm_bdl,
                 chunks: list[tuple[str, str]],
                 language,
                 entity_types=DEFAULT_ENTITY_TYPES,
                 embed_bdl=None,
                 callback=None
                 ):
        logging.info(f"Initializing Dealer for kb_id: {kb_id}, language: {language}")
        docids = list(set([docid for docid,_ in chunks]))
        logging.info(f"Processing {len(chunks)} chunks from {len(docids)} unique documents")
        
        self.llm_bdl = llm_bdl
        self.embed_bdl = embed_bdl
        ext = extractor(self.llm_bdl, language=language,
                        entity_types=entity_types,
                        get_entity=partial(get_entity, tenant_id, kb_id),
                        set_entity=partial(set_entity, tenant_id, kb_id, self.embed_bdl),
                        get_relation=partial(get_relation, tenant_id, kb_id),
                        set_relation=partial(set_relation, tenant_id, kb_id, self.embed_bdl)
                        )
        logging.info(f"Starting entity and relation extraction with {extractor.__name__}")
        ents, rels = ext(chunks, callback)
        logging.info(f"Extraction completed. Found {len(ents)} entities and {len(rels)} relations")
        
        self.graph = nx.Graph()
        for en in ents:
            self.graph.add_node(en["entity_name"], entity_type=en["entity_type"])#, description=en["description"])
        logging.info(f"Added {self.graph.number_of_nodes()} nodes to graph")


        for rel in rels:
            self.graph.add_edge(
                rel["src_id"],
                rel["tgt_id"],
                weight=rel["weight"],
            )
        logging.info(f"Added {self.graph.number_of_edges()} edges to graph")

        with RedisDistributedLock(kb_id, 60*30):

            start_time = time.time()
            
            logging.info("Retrieving existing graph from storage...")
            old_graph, old_doc_ids = get_graph(tenant_id, kb_id)
            logging.info(f"Graph retrieval took {time.time() - start_time:.2f} seconds")
            
            if old_graph is not None:
                logging.info(f"Found existing graph - nodes: {old_graph.number_of_nodes()}, edges: {old_graph.number_of_edges()}")
                logging.info(f"Current graph - nodes: {self.graph.number_of_nodes()}, edges: {self.graph.number_of_edges()}")
                
                merge_start = time.time()
                self.graph = reduce(graph_merge, [old_graph, self.graph])
                logging.info(f"Graph merge completed in {time.time() - merge_start:.2f} seconds")
                logging.info(f"After merge - nodes: {self.graph.number_of_nodes()}, edges: {self.graph.number_of_edges()}")
                
                # Debug info for merged graph
                node_types = Counter([data.get('entity_type') for _, data in self.graph.nodes(data=True)])
                logging.info(f"Node types distribution: {dict(node_types)}")
            
            logging.info("Starting pagerank and n-hop neighbor calculation...")
            pagerank_start = time.time()
            update_nodes_pagerank_nhop_neighbour(tenant_id, kb_id, self.graph, 2)
            logging.info(f"Pagerank calculation took {time.time() - pagerank_start:.2f} seconds")
            
            if old_doc_ids:
                docids.extend(old_doc_ids)
                docids = list(set(docids))
                logging.info(f"Updated document IDs, total unique documents: {len(docids)}")
            
            
            save_start = time.time()
            set_graph(tenant_id, kb_id, self.graph, docids)
            logging.info(f"Graph storage took {time.time() - save_start:.2f} seconds")
            
            total_time = time.time() - start_time
            logging.info(f"Total graph processing time: {total_time:.2f} seconds")
            logging.info("Graph processing and storage completed successfully")


class WithResolution(Dealer):
    def __init__(self,
                 tenant_id: str,
                 kb_id: str,
                 llm_bdl,
                 embed_bdl=None,
                 callback=None
                 ):
        self.llm_bdl = llm_bdl
        self.embed_bdl = embed_bdl

        with RedisDistributedLock(kb_id, 60*30):

            self.graph, doc_ids = get_graph(tenant_id, kb_id)
            if not self.graph:
                logging.error(f"Faild to fetch the graph. tenant_id:{kb_id}, kb_id:{kb_id}")
                if callback:
                    callback(-1, msg="Faild to fetch the graph.")
                return

            if callback:
                callback(msg="Fetch the existing graph.")
            er = EntityResolution(self.llm_bdl,
                                    get_entity=partial(get_entity, tenant_id, kb_id),
                                    set_entity=partial(set_entity, tenant_id, kb_id, self.embed_bdl),
                                    get_relation=partial(get_relation, tenant_id, kb_id),
                                    set_relation=partial(set_relation, tenant_id, kb_id, self.embed_bdl))
            reso = er(self.graph)
            self.graph = reso.graph
            logging.info("Graph resolution is done. Remove {} nodes.".format(len(reso.removed_entities)))
            if callback:
                callback(msg="Graph resolution is done. Remove {} nodes.".format(len(reso.removed_entities)))
            update_nodes_pagerank_nhop_neighbour(tenant_id, kb_id, self.graph, 2)
            set_graph(tenant_id, kb_id, self.graph, doc_ids)

        settings.docStoreConn.delete({
            "knowledge_graph_kwd": "relation",
            "kb_id": kb_id,
            "from_entity_kwd": reso.removed_entities
        }, search.index_name(tenant_id), kb_id)
        settings.docStoreConn.delete({
            "knowledge_graph_kwd": "relation",
            "kb_id": kb_id,
            "to_entity_kwd": reso.removed_entities
        }, search.index_name(tenant_id), kb_id)
        settings.docStoreConn.delete({
            "knowledge_graph_kwd": "entity",
            "kb_id": kb_id,
            "entity_kwd": reso.removed_entities
        }, search.index_name(tenant_id), kb_id)


class WithCommunity(Dealer):
    def __init__(self,
                 tenant_id: str,
                 kb_id: str,
                 llm_bdl,
                 embed_bdl=None,
                 callback=None
                 ):

        self.community_structure = None
        self.community_reports = None
        self.llm_bdl = llm_bdl
        self.embed_bdl = embed_bdl

        logging.info(f"Start to extract community reports for {kb_id}")

        with RedisDistributedLock(kb_id, 60*30):

            self.graph, doc_ids = get_graph(tenant_id, kb_id)
            if not self.graph:
                logging.error(f"Faild to fetch the graph. tenant_id:{kb_id}, kb_id:{kb_id}")
                if callback:
                    callback(-1, msg="Faild to fetch the graph.")
                return
            if callback:
                callback(msg="Fetch the existing graph.")

            cr = CommunityReportsExtractor(self.llm_bdl,
                                get_entity=partial(get_entity, tenant_id, kb_id),
                                set_entity=partial(set_entity, tenant_id, kb_id, self.embed_bdl),
                                get_relation=partial(get_relation, tenant_id, kb_id),
                                set_relation=partial(set_relation, tenant_id, kb_id, self.embed_bdl))
            cr = cr(self.graph, callback=callback)
            self.community_structure = cr.structured_output
            self.community_reports = cr.output
            set_graph(tenant_id, kb_id, self.graph, doc_ids)

        if callback:
            callback(msg="Graph community extraction is done. Indexing {} reports.".format(len(cr.structured_output)))

        settings.docStoreConn.delete({
            "knowledge_graph_kwd": "community_report",
            "kb_id": kb_id
        }, search.index_name(tenant_id), kb_id)

        for stru, rep in zip(self.community_structure, self.community_reports):
            obj = {
                "report": rep,
                "evidences": "\n".join([f["explanation"] for f in stru["findings"]])
            }
            chunk = {
                "docnm_kwd": stru["title"],
                "title_tks": rag_tokenizer.tokenize(stru["title"]),
                "content_with_weight": json.dumps(obj, ensure_ascii=False),
                "content_ltks": rag_tokenizer.tokenize(obj["report"] +" "+ obj["evidences"]),
                "knowledge_graph_kwd": "community_report",
                "weight_flt": stru["weight"],
                "entities_kwd": stru["entities"],
                "important_kwd": stru["entities"],
                "kb_id": kb_id,
                "source_id": doc_ids,
                "available_int": 0
            }
            chunk["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk["content_ltks"])
            #try:
            #    ebd, _ = self.embed_bdl.encode([", ".join(community["entities"])])
            #    chunk["q_%d_vec" % len(ebd[0])] = ebd[0]
            #except Exception as e:
            #    logging.exception(f"Fail to embed entity relation: {e}")
            settings.docStoreConn.insert([{"id": chunk_id(chunk), **chunk}], search.index_name(tenant_id))

