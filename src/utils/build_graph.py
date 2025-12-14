import pandas as pd 
import numpy as np 
import torch 
import networkx as nx 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MultiLabelBinarizer

#1. SETUP PATHS 
METADATA_PATH = "../../data/raw/coin_metadata.csv"
OUTPUT_PATH = "../../data/processed/adjacency_matrix.pt"
GRAPH_PLOT_PATH = "../../data/processed/crypto_network.png"


def build_graph(threshold = 1):
    """
    Builds graph where nodes are coins and edges represent shared tags.
    
    threshold: Minimum number of shared tags to create an edge. # is this a good idea?
    """
    print(f"--- Building Graph (Threshold: {threshold} shared tags) ---")
    
    # load meta data: 
    try:
        df = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print("Error: Metadata file not found. Run fetch_crypto_data.py first.")
        return

    # 3. The CSV saves lists as strings like "['DeFi', 'DEX']", we need to clean that.
    df['tags_list'] = df['tags'].fillna("").apply(lambda x: x.split(",") if x else [])
    
    # 4. Multi Hot encoding: 
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(df["tags_list"])
    
    print(f"Found {len(mlb.classes_)} unique tags across {len(df)} coins.")
    print(f"Example tags: {mlb.classes_[:10]}")
    
    # 5. Compute Adjacency matrix: 
    adj_matrix = np.dot(tag_matrix, tag_matrix.T)
    # remove self loops: 
    np.fill_diagonal(adj_matrix, 0)
    
    # 6. apply threshold 
    adj_matrix[adj_matrix < threshold] = 0
    
    # Convert to Tensor for PyTorch
    adj_tensor = torch.FloatTensor(adj_matrix)
    torch.save(adj_tensor, OUTPUT_PATH)
    print(f"Graph Adjacency Matrix saved to {OUTPUT_PATH}")
    print(f"Shape: {adj_tensor.shape}")

    # 7. Visualization (Optional but cool)
    visualize_graph(adj_matrix, df['symbol'].tolist())
    

def visualize_graph(adj_matrix, labels):
    """
    Draws the network using NetworkX
    """
    plt.figure(figsize=(12, 12))
    
    # Create Graph Object
    G = nx.from_numpy_array(adj_matrix)
    
    # Remove nodes with no connections (to make chart cleaner)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    # Update labels to match remaining nodes
    active_labels = {i: label for i, label in enumerate(labels) if i not in isolated_nodes}
    
    # Layout (Spring layout pushes connected nodes together)
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color='skyblue', alpha=1)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='black')
    nx.draw_networkx_labels(G, pos, labels=active_labels, font_size=8)
    
    plt.title("Crypto Correlation Graph (Based on Shared Tags)")
    plt.axis('off')
    plt.savefig(GRAPH_PLOT_PATH)
    print(f"Graph visualization saved to {GRAPH_PLOT_PATH}")
    plt.show()  
    
if __name__ == "__main__":
    build_graph()
    