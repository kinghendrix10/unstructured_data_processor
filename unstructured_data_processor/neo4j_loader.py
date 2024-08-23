# unstructured_data_processor/neo4j_loader.py
from neo4j import GraphDatabase
import json
import logging
import re
from typing import Dict, Any, List

class Neo4jLoader:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)
        logging.info("Database cleared")

    def create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Legislation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GovernmentBody) REQUIRE n.id IS UNIQUE"
        ]
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
        logging.info("Constraints created")

    @staticmethod
    def sanitize_label(label):
        return ''.join(word.capitalize() for word in re.findall(r'\w+', label))

    def load_entities(self, entities: List[Dict[str, Any]]):
        with self.driver.session() as session:
            for entity in entities:
                session.execute_write(self._create_entity, entity)
        logging.info(f"Loaded {len(entities)} entities")

    @staticmethod
    def _create_entity(tx, entity):
        sanitized_type = Neo4jLoader.sanitize_label(entity['type'])
        query = (
            f"MERGE (e:{sanitized_type} {{id: $id}}) "
            f"SET e.name = $name, e.metadata = $metadata, e.original_type = $original_type"
        )
        tx.run(query, id=entity['id'], name=entity['name'],
               metadata=json.dumps(entity['metadata']),
               original_type=entity['type'])

    def load_relationships(self, relationships: List[Dict[str, Any]]):
        with self.driver.session() as session:
            for relationship in relationships:
                session.execute_write(self._create_relationship, relationship)
        logging.info(f"Loaded {len(relationships)} relationships")

    @staticmethod
    def _create_relationship(tx, rel):
        rel_type = rel['type'].upper().replace(' ', '_')
        query = (
            f"MATCH (source) WHERE source.id = $source_id "
            f"MATCH (target) WHERE target.id = $target_id "
            f"CREATE (source)-[r:{rel_type} {{metadata: $metadata}}]->(target)"
        )
        tx.run(query, source_id=rel['source'], target_id=rel['target'],
               metadata=json.dumps(rel['metadata']))

    def load_data(self, data: Dict[str, Any]):
        self.clear_database()
        self.create_constraints()
        self.load_entities(data['entities'])
        self.load_relationships(data['relationships'])
        logging.info("Data loaded successfully into Neo4j")
