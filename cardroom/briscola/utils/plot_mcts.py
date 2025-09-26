import matplotlib.pyplot as plt

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
