import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2
import operator


class Node:
    def __init__(self, position, i):
        self.position = position
        self.derivative_gradient = 0.0
        self.input_edge_cost = {}
        self.best_input_edge = None
        self.unary_cost = 0.0
        self.level = i

def display_image(image,title="random"):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gradient(image):
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image,-1,np.fliplr(np.flipud(sobel)))

def gradient_new(image):
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return cv2.filter2D(image,-1,np.fliplr(np.flipud(sobel)))

def edges(image):
    edges = cv2.Canny(image, 200, 450)
    return edges

def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=True):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

def internal_energy(curve,alpha,beta):
    return np.sqrt(np.sum(np.square(np.subtract(curve, np.roll(curve, 1)))))

def external_energy(image, curve):
    magnitude = gradient(image)
    return np.sum(-1.0*magnitude[curve.T[1],curve.T[0]]**2.0)

def local_neighbors(P):
    i = P[0]
    j = P[1]
    return np.array([[i,j-1],[i,j+1],[i-1,j-1],[i+1,j+1],[i-1,j],[i-1,j+1],[i+1,j],[i+1,j-1],[i,j]])

def generate_nodes(V,magnitude,node_dict):
    for i in range(2,len(V)):
        possible_positions = local_neighbors(V[i])
        input_edges = node_dict[i-1]
        node_list = []
        for position in possible_positions:
            current_position = position
            node = Node(current_position,i)
            node.derivative_gradient = -1.0*magnitude[current_position[1]][current_position[0]]**2
            edge_cost = list(np.square(np.sqrt(np.sum(np.square(current_position - np.array([edge.position for edge in input_edges])),axis=1))))
            node.input_edge_cost = dict(zip(input_edges,edge_cost))
            node_list.append(node)
        node_dict[i]=node_list
    return node_dict



def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    '''Load Image and edges'''
    Im, V = load_data(fpath, radius)
    plot_every = 5
    ''' Calculate magnitude '''
    if "ball" in fpath:
        magnitude = gradient(Im)
    elif "coffee" in fpath:
        magnitude = edges(Im)
    ''' Generate node dict for initial configuration '''
    node_dict = {k: [] for k in range(len(V))}
    ''' Fix a node 0 state  randomly '''
    possible_positions = local_neighbors(V[0])
    for position in possible_positions:
        current_position = position
        node = Node(current_position, 0)
        node.derivative_gradient = -1.0*magnitude[current_position[1]][current_position[0]]**2
        node.unary_cost = node.derivative_gradient
        node_dict[0].append(node)
    ''' Assign only randomly chosen state as previous vertex of level 1 '''
    possible_positions = local_neighbors(V[1])
    random_fixed_node = np.random.choice(node_dict[0])
    for position in possible_positions:
        current_position = position
        node = Node(current_position, 1)
        node.derivative_gradient = -1.0*magnitude[current_position[1]][current_position[0]]**2
        node.unary_cost = node.derivative_gradient
        node.input_edge_cost = {random_fixed_node:0}
        node_dict[1].append(node)
    node_dict = generate_nodes(V,magnitude,node_dict)
    ''' Start iteration '''
    n_steps = 70
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.pause(5)
    for t in range(n_steps):
        for i in range(1,len(V)):
            for node in node_dict[i]:
                possible_paths = {}
                input_edges = node.input_edge_cost
                for item in input_edges:
                    possible_paths[item] = item.unary_cost + input_edges[item] + node.derivative_gradient
                sorted_paths = sorted(possible_paths.items(), key=operator.itemgetter(1))
                node.best_input_edge = sorted_paths[0]
                node.unary_cost = node.best_input_edge[0].unary_cost + node.best_input_edge[1] + node.derivative_gradient
        new_nodes_list = []
        last_level = node_dict[len(V)-1]
        sorted_last_level = sorted(last_level, key=lambda x: x.unary_cost, reverse=False)
        new_nodes_list = new_nodes_list + [sorted_last_level[0]]
        for i in range(len(V)-1):
            new_nodes_list = [new_nodes_list[0].best_input_edge[0]] + new_nodes_list
        V = np.array([node.position for node in new_nodes_list])
        node_dict = generate_nodes(V,magnitude,node_dict)
        if t%plot_every==0:
            ax.clear()
            ax.imshow(Im, cmap='gray')
            ax.set_title('frame ' + str(t))
            plot_snake(ax, V)
            plt.pause(0.5)



if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=120)
