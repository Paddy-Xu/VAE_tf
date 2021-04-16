from GCN.data_loader import *
import networkx as nx
from skimage.util import montage
from data_loader import *

def draw_graph_mpl(g, pos=None, ax=None, layout_func=nx.drawing.layout.kamada_kawai_layout, draw_labels=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    else:
        fig = None
    if pos is None:
        pos = layout_func(g)
    node_color = []
    node_labels = {}
    shift_pos = {}
    for k in g:
        node_color.append(g.nodes[k].get('color', 'green'))
        node_labels[k] = g.nodes[k].get('label', k)
        shift_pos[k] = [pos[k][0], pos[k][1]]

    edge_color = []
    edge_width = []
    for e in g.edges():
        edge_color.append(g.edges[e].get('color', 'black'))
        edge_width.append(g.edges[e].get('width', 0.5))
    nx.draw_networkx_edges(g, pos, font_weight='bold', edge_color=edge_color, width=edge_width, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_shape='p', node_size=300, alpha=0.75, ax=ax)
    if draw_labels:
        nx.draw_networkx_labels(g, shift_pos, labels=node_labels, arrows=True, ax=ax)
    ax.autoscale()
    return fig, ax, pos

data_loader = data_loader()
adj = data_loader.get_adj()
x_single = data_loader.get_one_batch_x()

print(adj.shape, 'adjacency matrix')
plt.matshow(adj.todense())
xx, yy = np.meshgrid(np.arange(28), np.arange(28))
node_id = ['X:{:02d}_Y:{:02d}'.format(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

print(node_id[300], 'is connected to')
for row, col in zip(*adj[300].nonzero()):
    print(col, '->', node_id[col])

G = nx.from_scipy_sparse_matrix(adj[:10, :10])
for k, pix_val in zip(G.nodes, x_single[0]):
    G.nodes[k]['label'] = node_id[k]
draw_graph_mpl(G);

G = nx.from_scipy_sparse_matrix(adj)
for k, pix_val in zip(G.nodes, x_single[0]):
    G.nodes[k]['label'] = ''
    G.nodes[k]['color'] = 'red' if pix_val>0.5 else 'green'
draw_graph_mpl(G, pos=np.stack([xx.ravel(), yy.ravel()], -1));

adj, loader_tr, loader_va, loader_te = data_loader.getdata()



def show_intermediate(loader, model, layer_name = 'conv64'):
    layer_output = model.get_layer(layer_name).output

    model = Model(model.input, outputs=layer_output)

    step = 0

    # Training loop
    results_tr = []
    for batch in loader_tr:
        inputs, target = batch
        x, a = inputs
        gc1_out = model.predict(inputs, training=False)

        #i_model = Model(inputs=[X_in, A_in], outputs=[graph_conv_1, graph_conv_2])

        fig, m_axs = plt.subplots(4, 3, figsize=(20, 15))
        for i, (ax1, ax2, ax3) in enumerate(m_axs):
            ax1.imshow(x[i].reshape((28, 28)))
            gc_stack = gc1_out[i].reshape((28, 28, -1)).swapaxes(0, 2).swapaxes(1, 2)
            ax2.imshow(montage(gc_stack), vmin=-0.5, vmax=0.5, cmap='RdBu')
            ax2.set_title(i_model.output_names[0])
            gc_stack = gc2_out[i].reshape((28, 28, -1)).swapaxes(0, 2).swapaxes(1, 2)
            ax3.imshow(montage(gc_stack), vmin=-0.5, vmax=0.5, cmap='RdBu')
            ax3.set_title(model.output_names[1])
