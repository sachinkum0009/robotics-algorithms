import math

from matplotlib import pyplot as plt

from planners.graph import Graph


def plot_graph(graph: Graph, path: list):
    """Visualize the graph and the path using Matplotlib."""

    # Assign positions in a circle
    names = list(graph.nodes.keys())
    n = len(names)
    pos = {
        name: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, name in enumerate(names)
    }

    path_edges = set()
    if path:
        for i in range(len(path) - 1):
            path_edges.add((path[i].name, path[i + 1].name))
            path_edges.add((path[i + 1].name, path[i].name))

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw edges
    drawn_edges = set()
    for node in graph.nodes.values():
        for neighbor, weight in node.neighbors.items():
            edge = tuple(sorted([node.name, neighbor.name]))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            x = [pos[node.name][0], pos[neighbor.name][0]]
            y = [pos[node.name][1], pos[neighbor.name][1]]
            is_path_edge = (node.name, neighbor.name) in path_edges
            color = "tomato" if is_path_edge else "lightgray"
            lw = 3 if is_path_edge else 1.5
            ax.plot(x, y, color=color, linewidth=lw, zorder=1)
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            ax.text(
                mx,
                my,
                str(weight),
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
            )

    # Draw nodes
    path_names = {n.name for n in path}
    for name, (x, y) in pos.items():
        color = "tomato" if name in path_names else "steelblue"
        ax.scatter(x, y, s=600, color=color, zorder=2)
        ax.text(
            x,
            y,
            name,
            fontsize=12,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=3,
        )

    path_label = " → ".join(n.name for n in path) if path else "No path found"
    ax.set_title(f"Dijkstra's Shortest Path\n{path_label}", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
