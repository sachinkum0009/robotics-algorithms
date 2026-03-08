from dataclasses import dataclass, field
import heapq
from itertools import count

@dataclass
class Node:
    """Represents a node in the graph"""
    name: str
    neighbors: dict['Node', float] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        return NotImplemented

@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)

    def add_edge(self, node_name: str, neighbor_name: str, weight: float):
        """Adds an undirected edge between two nodes in the graph"""
        if node_name not in self.nodes:
            self.nodes[node_name] = Node(name=node_name)
        if neighbor_name not in self.nodes:
            self.nodes[neighbor_name] = Node(name=neighbor_name)
        self.nodes[node_name].neighbors[self.nodes[neighbor_name]] = weight
        self.nodes[neighbor_name].neighbors[self.nodes[node_name]] = weight

@dataclass
class QueueItem:
    cost: float
    counter: count
    node: Node