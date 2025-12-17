"""
Visualize GCN model outputs on graph
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def visualize_graph_output(g, output, adj_matrix=None, reduction='max', 
                          layout='spring', node_size=300, figsize=(12, 10),
                          save_path=None, title="Graph Output Visualization"):
    """
    Visualize graph with node features from model output
    
    Parameters
    ----------
    g : networkx.Graph
        Graph object
    output : np.ndarray
        Model output with shape (num_nodes, feature_dim)
        e.g., (1024, 32)
    adj_matrix : np.ndarray, optional
        Adjacency matrix (for layout reference)
    reduction : str, default='max'
        How to reduce feature dimension: 'max', 'mean', 'sum', 'norm'
    layout : str, default='spring'
        Graph layout: 'spring', 'circular', 'kamada_kawai', 'spectral'
    node_size : int, default=300
        Base node size
    figsize : tuple, default=(12, 10)
        Figure size
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    num_nodes, feature_dim = output.shape
    
    # Reduce feature dimension
    if reduction == 'max':
        node_features = np.max(output, axis=1)  # (num_nodes,)
    elif reduction == 'mean':
        node_features = np.mean(output, axis=1)
    elif reduction == 'sum':
        node_features = np.sum(output, axis=1)
    elif reduction == 'norm':
        node_features = np.linalg.norm(output, axis=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'max', 'mean', 'sum', or 'norm'")
    
    # Normalize features for visualization
    feature_min = node_features.min()
    feature_max = node_features.max()
    feature_range = feature_max - feature_min
    if feature_range > 0:
        node_features_norm = (node_features - feature_min) / feature_range
    else:
        node_features_norm = np.ones_like(node_features) * 0.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(g, k=1, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(g)
    elif layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(g)
        except:
            pos = nx.spring_layout(g, seed=42)
    elif layout == 'spectral':
        try:
            pos = nx.spectral_layout(g)
        except:
            pos = nx.spring_layout(g, seed=42)
    else:
        pos = nx.spring_layout(g, seed=42)
    
    # Create colormap
    cmap = cm.get_cmap('viridis')  # or 'plasma', 'coolwarm', 'RdYlBu'
    norm = Normalize(vmin=feature_min, vmax=feature_max)
    
    # Draw edges
    nx.draw_networkx_edges(g, pos, alpha=0.2, width=0.5, ax=ax)
    
    # Draw nodes with colors based on features
    nodes = list(g.nodes())
    node_colors = [cmap(norm(node_features[node])) for node in nodes]
    
    # Node size can also vary with feature value
    node_sizes = [node_size * (0.5 + 0.5 * node_features_norm[node]) for node in nodes]
    
    nx.draw_networkx_nodes(g, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8,
                          ax=ax)
    
    # Add colorbar
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label(f'Node Feature ({reduction})', rotation=270, labelpad=20)
    
    ax.set_title(f"{title}\n{reduction.capitalize()} of {feature_dim}D features", 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    return fig, ax


def visualize_graph_output_comparison(g, outputs_dict, reduction='max',
                                     layout='spring', node_size=300,
                                     figsize=(16, 12), save_path=None):
    """
    Compare multiple outputs side by side
    
    Parameters
    ----------
    g : networkx.Graph
        Graph object
    outputs_dict : dict
        Dictionary of {name: output_array} where output_array is (num_nodes, feature_dim)
    reduction : str
        How to reduce feature dimension
    layout : str
        Graph layout
    node_size : int
        Base node size
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    num_outputs = len(outputs_dict)
    cols = min(3, num_outputs)
    rows = (num_outputs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_outputs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Choose layout once (same for all subplots)
    if layout == 'spring':
        pos = nx.spring_layout(g, k=1, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(g)
    else:
        pos = nx.spring_layout(g, seed=42)
    
    # Get global feature range for consistent colormap
    all_features = []
    for output in outputs_dict.values():
        if reduction == 'max':
            features = np.max(output, axis=1)
        elif reduction == 'mean':
            features = np.mean(output, axis=1)
        elif reduction == 'sum':
            features = np.sum(output, axis=1)
        elif reduction == 'norm':
            features = np.linalg.norm(output, axis=1)
        all_features.append(features)
    
    all_features = np.concatenate(all_features)
    feature_min = all_features.min()
    feature_max = all_features.max()
    
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=feature_min, vmax=feature_max)
    
    for idx, (name, output) in enumerate(outputs_dict.items()):
        ax = axes[idx]
        
        # Reduce features
        if reduction == 'max':
            node_features = np.max(output, axis=1)
        elif reduction == 'mean':
            node_features = np.mean(output, axis=1)
        elif reduction == 'sum':
            node_features = np.sum(output, axis=1)
        elif reduction == 'norm':
            node_features = np.linalg.norm(output, axis=1)
        
        # Normalize
        feature_range = feature_max - feature_min
        if feature_range > 0:
            node_features_norm = (node_features - feature_min) / feature_range
        else:
            node_features_norm = np.ones_like(node_features) * 0.5
        
        # Draw
        nx.draw_networkx_edges(g, pos, alpha=0.2, width=0.5, ax=ax)
        
        nodes = list(g.nodes())
        node_colors = [cmap(norm(node_features[node])) for node in nodes]
        node_sizes = [node_size * (0.5 + 0.5 * node_features_norm[node]) for node in nodes]
        
        nx.draw_networkx_nodes(g, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        ax.set_title(f"{name}\n({reduction})", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_outputs, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
    #                    pad=0.05, aspect=40)
    # cbar.set_label(f'Node Feature ({reduction})', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison visualization to: {save_path}")
    
    return fig, axes


if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create a simple graph
    g = nx.erdos_renyi_graph(50, 0.1, seed=42)
    
    # Create dummy output (50 nodes, 32 features)
    output = np.random.randn(50, 32)
    
    # Visualize
    visualize_graph_output(g, output, reduction='max', 
                          save_path="example_output.png")
    
    print("Example visualization created!")

