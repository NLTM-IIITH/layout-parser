import networkx as nx
import matplotlib.pyplot as plt

def create_graphs(euclidean):
    G = nx.Graph()
    for _, row in euclidean.iterrows():
        box_id = row['Id']
        right_box = row['Right_Box']
        left_box = row['Left_Box']
        top_box = row['Top_Box']
        bottom_box = row['Bottom_Box']
        right_box_id = right_box[1]
        left_box_id = left_box[1]
        top_box_id = top_box[1]
        bottom_box_id = bottom_box[1]
        # right_box_id = parse_string(right_box," ","]")
        # left_box_id = parse_string(left_box," ","]")
        # top_box_id = parse_string(top_box," ","]")
        # bottom_box_id = parse_string(bottom_box," ","]")
        G.add_node(box_id)
        if right_box[0] != -1 and right_box[1] != "-":
            G.add_edge(int(box_id), int(right_box_id))
        if left_box[0] != -1 and left_box[1] != "-":
            G.add_edge(int(box_id), int(left_box_id))
        if top_box[0] != -1 and top_box[1] != "-":
            G.add_edge(int(box_id), int(top_box_id))
        if bottom_box[0] != -1 and bottom_box[1] != "-":
            G.add_edge(int(box_id), int(bottom_box_id))
    connected_components = nx.connected_components(G)

    subgraphs = [G.subgraph(component) for component in connected_components]

    num_subgraphs = len(subgraphs)
    num_cols = 2
    num_rows = (num_subgraphs + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    axs = axs.flatten()

    for i, subgraph in enumerate(subgraphs):
        ax = axs[i]
        pos = nx.spring_layout(subgraph, k=0.5)
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=100, font_size=8, edge_color='gray', ax=ax)
        ax.set_title(f'Subgraph {i+1}')

    if num_subgraphs < len(axs):
        for j in range(num_subgraphs, len(axs)):
            fig.delaxes(axs[j])

    plt.savefig('subgraphs.png')
    plt.tight_layout()
    plt.show()

    

    return G