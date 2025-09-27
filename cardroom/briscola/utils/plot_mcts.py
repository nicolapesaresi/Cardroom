import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from cardroom.briscola.agents.algorithms.mcts import InfoSetNode

def plot_info_node(root_node: InfoSetNode, action_labels:list|None = None):
    """
    Plot visit counts and Q-values for each action at a given InfoSetNode.
    
    Args:
        root_node: the root InfoSetNode from ISMCTS.
        action_labels: optional list of labels for each action (e.g., card names). If None, uses indices.
    """
    actions = root_node.legal_actions
    N_values = [root_node.N[a] for a in actions]
    Q_values = [root_node.Q[a] for a in actions]

    if action_labels is None:
        action_labels = [str(a) for a in actions]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Visit count (N)', color=color)
    ax1.bar(action_labels, N_values, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean value (Q)', color=color)
    ax2.plot(action_labels, Q_values, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('ISMCTS Root Node Statistics')
    plt.show()


def recap_info_node(root_node: InfoSetNode, top_k:list|None = None) -> str:
    """
    Create a textual summary of the InfoSetNode, showing action stats.
    
    Args:
        root_node: the root InfoSetNode to summarize.
        top_k: if set, show only the top_k actions by visit count.
    
    Returns:
        summary string.
    """
    actions = root_node.legal_actions
    summary_lines = [f"Player to move: {root_node.player}", f"Total visits: {root_node.total_visits}"]

    action_stats = [
        (a, root_node.N[a], root_node.Q[a]) for a in actions
    ]
    # sort by visit count descending
    action_stats.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        action_stats = action_stats[:top_k]

    summary_lines.append(f"{'Action':>6} | {'N':>6} | {'Q':>6}")
    summary_lines.append("-" * 22)
    for a, n, q in action_stats:
        summary_lines.append(f"{a:>6} | {n:>6} | {q:>6.2f}")

    return "\n".join(summary_lines)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_ismcts_tree(root_node: InfoSetNode, info_map: dict, max_depth: int = 2):
    """
    Plot the ISMCTS tree starting from the root node.
    
    Args:
        root_node: the InfoSetNode at the root.
        info_map: the full dictionary {obs_key -> InfoSetNode}.
        max_depth: how many levels of the tree to display (to avoid clutter).
    """
    G = nx.DiGraph()

    # recursive build
    def add_node_recursively(node: InfoSetNode, depth: int, parent_id: str = None, action: int = None):
        if depth > max_depth:
            return

        node_id = str(id(node))
        # label includes visits and mean value
        label = f"P{node.player}\nN={node.total_visits}\nQ={np.mean(list(node.Q.values())):.2f}"
        G.add_node(node_id, label=label, value=np.mean(list(node.Q.values())))

        if parent_id is not None:
            # add edge with action label
            G.add_edge(parent_id, node_id, action=action)

        # recurse on children
        for a, child_key in node.children.items():
            child = info_map[child_key]
            add_node_recursively(child, depth + 1, node_id, a)

    # build graph
    add_node_recursively(root_node, depth=0)

    # draw with spring layout
    pos = nx.spring_layout(G, seed=42)
    values = [G.nodes[n].get("value", 0.0) for n in G.nodes()]
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    edge_labels = {(u, v): f"a={d['action']}" for u, v, d in G.edges(data=True)}

    # map Q-values to colors (red = bad, green = good)
    cmap = cm.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=min(values), vmax=max(values))
    node_colors = [cmap(norm(v)) for v in values]

    plt.figure(figsize=(10, 7))
    ax = plt.gca()  # get current axes
    nx.draw(
        G, pos,
        labels=labels,
        node_color=node_colors,
        node_size=2000,
        font_size=8,
        with_labels=True,
        font_color="black",
        ax=ax
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    plt.title(f"ISMCTS Tree (depth ≤ {max_depth})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean Q-value")  # <-- attach to the Axes explicitly
    plt.show()


