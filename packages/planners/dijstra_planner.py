from .planner import BaseController

import heapq

class DijkstraPlanner(BaseController):
    def __init__(self, graph):
        self.graph = graph  # Graph represented as an adjacency list or matrix

    def plan(self, start_node, goal_node) -> list:

        # Dijkstra's algorithm implementation
        queue = [(0, start_node)]
        distances = {node: float('inf') for node in self.graph}
        distances[start_node] = 0
        previous_nodes = {node: None for node in self.graph}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        # Reconstruct path
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()

        return path
