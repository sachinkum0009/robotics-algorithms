import heapq
from itertools import count
import sys
import os

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )


from planners.graph import Node, Graph
from planners.planner import BasePlanner
from planners.utils import plot_graph


class AStarPlanner(BasePlanner):
    def __init__(self, graph: Graph, heuristic: dict[Node, float]):
        self.graph = graph
        self.heuristic = heuristic

    def plan(self, start_node: Node, goal_node: Node) -> list:
        counter = count()  # tiebreaker to avoid comparing Node objects
        queue = [
            (0 + self.heuristic[start_node], next(counter), start_node, 0)
        ]  # (f_cost, tie, node, g_cost)
        visited = set()
        parent_map = {start_node: None}  # To reconstruct the path
        while queue:
            f_cost, _, node, g_cost = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)

            if node == goal_node:
                break  # Found the goal

            for neighbor, weight in node.neighbors.items():
                if neighbor not in visited:
                    new_g_cost = g_cost + weight
                    new_f_cost = new_g_cost + self.heuristic[neighbor]
                    heapq.heappush(
                        queue,
                        (new_f_cost, next(counter), neighbor, new_g_cost),
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


def main():
    graph = Graph()
    graph.add_edge("A", "B", 1)
    graph.add_edge("A", "C", 4)
    graph.add_edge("B", "C", 2)
    graph.add_edge("B", "D", 5)
    graph.add_edge("C", "D", 1)

    heuristic = {
        graph.nodes["A"]: 7.0,
        graph.nodes["B"]: 4.0,
        graph.nodes["C"]: 6.0,
        graph.nodes["D"]: 0.0,
    }

    planner = AStarPlanner(graph, heuristic)
    path = planner.plan(graph.nodes["A"], graph.nodes["D"])
    print(path)
    print("Path:", [node.name for node in path])
    plot_graph(graph, path)
    # plt.show()


if __name__ == "__main__":
    main()
