import heapq
import os
import sys
from itertools import count

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

from planners.graph import Graph, Node, QueueItem
from planners.planner import BasePlanner
from planners.utils import plot_graph


class DijkstraPlanner(BasePlanner):
    def __init__(self, graph: Graph):
        self.graph = graph  # Graph represented as an adjacency list or matrix

    def plan(self, start_node: Node, goal_node: Node) -> list:
        # Dijkstra's algorithm implementation
        counter = count()  # tiebreaker to avoid comparing Node objects
        queue = [(0, next(counter), start_node)]  # (cost, tie, node)
        visited = set()
        parent_map = {start_node: None}  # To reconstruct the path
        while queue:
            cost, _, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            print(f"queue: {queue}")

            if node == goal_node:
                break  # Found the goal

            for neighbor, weight in node.neighbors.items():
                if neighbor not in visited:
                    heapq.heappush(
                        queue, (cost + weight, next(counter), neighbor)
                    )
                    if neighbor not in parent_map:
                        parent_map[neighbor] = node

        # Return empty list if goal was not reached
        if goal_node not in parent_map:
            return []

        # Reconstruct the path from goal to start
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node)
            current_node = parent_map[current_node]
        path.reverse()
        return path
    
    def plan2(self, start_node: Node, goal_node: Node) -> list:
        # Dijkstra's algorithm implementation
        queue = [(0, start_node)]  # (cost, node)
        visited = set()
        parent_map = {start_node: None}  # To reconstruct the path
        while queue:
            cost, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)

            if node == goal_node:
                break  # Found the goal

            for neighbor, weight in node.neighbors.items():
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor))
                    if neighbor not in parent_map:
                        parent_map[neighbor] = node

        # Return empty list if goal was not reached
        if goal_node not in parent_map:
            return []

        # Reconstruct the path from goal to start
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node)
            current_node = parent_map[current_node]
        path.reverse()
        return path
    
    def plan3(self, start_node: Node, goal_node: Node) -> list:
        counter = count()  # tiebreaker to avoid comparing Node objects
        queue: list[QueueItem] = [QueueItem(cost=0, counter=next(counter), node=start_node)]
        visited_nodes = set()
        parent_map = {start_node: None}

        while queue:
            cost, _, node = heapq.heappop(queue)
            if node in visited_nodes:
                continue
            visited_nodes.add(node)

            if node == goal_node:
                break

            for neighbor, weight in node.neighbors.items():
                if neighbor not in visited_nodes:
                    heapq.heappush(queue, QueueItem(cost=cost + weight, counter=next(counter), node=neighbor)) # type: ignore  # noqa: E501
                    if neighbor not in parent_map:
                        parent_map[neighbor] = node

        if goal_node not in parent_map:
            return []
        
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node)
            current_node = parent_map[current_node]
        path.reverse()

        return path


def main():
    graph = Graph()
    graph.add_edge("A", "B", 3)
    graph.add_edge("A", "C", 2)
    graph.add_edge("B", "D", 4)
    graph.add_edge("D", "E", 2)
    graph.add_edge("C", "E", 3)

    planner = DijkstraPlanner(graph)
    print("types:", type(graph.nodes["A"]), type(graph.nodes["E"]))
    path = planner.plan2(graph.nodes["A"], graph.nodes["E"])
    print("Shortest path from A to E:", [node.name for node in path])
    plot_graph(graph, path)


if __name__ == "__main__":
    main()
