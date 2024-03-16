### Vizualize.py
# Contains all functions making vizualizations using simulation data
###


"""
def viz_UV(UV_space):
    #\\FIXME Most likely some recording of the UV space is incorrect as there is no dynamic change in the UV space 
    
    t=0
    df = pd.DataFrame(UV_space[0][t], columns=['U'])
    df['V'] = UV_space[1][t]

    fig = px.scatter(df, x="U", y="V")
    fig = px.density_contour(df, x="U", y="V", marginal_x="histogram", marginal_y="histogram")
    #fig.update_traces(contours_coloring="fill", contours_showlabels = True)
    fig.show()
"""

def viz_deg_distr(G):

    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()



def viz_effect_clustering_coef(name, p, data):
    fig, ax = plt.subplots()
    ax.scatter(p, data, c="green", alpha=0.5, marker='x')
    ax.set_xlabel("Clustering Coefficient")
    ax.set_ylabel(name)
    ax.legend()
    plt.show()



"""                   
def vizualize_effect_of_rationality_on_QRE(df):
    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Mean Lambda'], df['Std Dev Lambda'], df['QRE Result'], c='blue', marker='o')

    ax.set_xlabel('Mean Lambda')
    ax.set_ylabel('Std Dev Lambda')
    ax.set_zlabel('QRE Result')

    path = utils.make_path("Figures", "GameChoice", "Effect_of_Rationality_on_QRE")
    plt.savefig(path)
    plt.close()
"""





    

