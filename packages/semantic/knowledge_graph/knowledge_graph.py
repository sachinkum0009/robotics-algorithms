"""
Code for Knowledge Graph
"""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import yaml
from neo4j import GraphDatabase


@dataclass
class Pose:
    """
    position and orientation for the object
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float]


@dataclass
class Node:
    """Define node with id, name, position"""

    id: UUID
    name: str
    pose: Pose


class KnowledgeGraph:
    g: list[Node] = []

    def __init__(self, yaml_file_path: Path):
        """KnowledgeGraph to search objects and their relations"""
        self._load_from_yaml(yaml_file_path)

    def _load_from_yaml(self, file_path: Path):
        with open(file_path.absolute()) as f:
            data = yaml.safe_load(f)
            for item in data:
                print(item)

    def object_exists(self, name: str):
        for node in self.g:
            if node.name == name:
                return True
        else:
            return False

    def find_object(self, name: str):
        if name in self.g:
            return self.g[0]


knowledge_graph = Path("knowledge_graph.yaml")
kg = KnowledgeGraph(knowledge_graph)
camera_obj = kg.find_object("camera")

if camera_obj:
    print(f"camera object is found at position: {camera_obj.pose}")
