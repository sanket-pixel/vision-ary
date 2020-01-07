import cv2
import numpy as np
import maxflow
from matplotlib import pyplot as plt
from matplotlib import cm
def display_image(image,title="random"):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def mrf_binary(I,rho=0.7,pairwise_cost_same=0.005,pairwise_cost_diff=0.2):


    ### 1) Define Graph
    g = maxflow.Graph[float]()
    ### 2) Add pixels as nodes
    pixel_nodes = g.add_grid_nodes(I.shape)
    ### 3) Compute Unary cost
    P_given_0 = np.where(I==0,rho,1-rho)
    P_given_1 = np.where(I==255,rho,1-rho)
    unary_source = np.negative(np.log10(P_given_0))
    unary_sink = np.negative(np.log10(P_given_1))
    ### 4) Add terminal edges
    g.add_grid_tedges(pixel_nodes, unary_source+pairwise_cost_same, unary_sink-pairwise_cost_same)
    ### 5) Add Node edges
    ### Vertical Edges
    g.add_grid_edges(pixel_nodes,pairwise_cost_diff)
    ### 6) Maxflow
    g.maxflow()
    sgm = g.get_grid_segments(pixel_nodes)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.title.set_text("Original Image")
    ax1.imshow(I, cmap=cm.get_cmap("gray"))
    ax2.title.set_text("Pairwise-Cost-Diff = " + str(pairwise_cost_diff))
    Denoised_I = (np.int_(np.logical_not(sgm))*255).astype(np.uint8)
    ax2.imshow(Denoised_I, cmap=cm.get_cmap("gray"))
    plt.show()
    return



def alpha_expansion(I,rho=0.6):
    # I = np.array([[0,0,255],[255,0,0],[0,0,0]])
    labels = np.unique(I).tolist()
    label_dict = {1:0,2:128,3:255}
    original_image_unchanged = np.copy(I)
    original_image = np.copy(I)
    Denoised_I = np.zeros_like(I)
    for alpha in [1,2,3]:
        non_alpha=set([1,2,3]).difference(set([alpha]))
        ### Use Alpha expansion binary image for each label
        ### 1) Define Graph
        g = maxflow.Graph[float]()
        ### 2) Add pixels as nodes
        pixel_nodes = g.add_grid_nodes(I.shape)
        ### 3) Compute Unary cost
        P_given_alpha = np.where(original_image==label_dict[alpha],0.8,0.1)
        P_given_label = np.where(original_image==label_dict[alpha],0.1,0.8)
        unary_source = np.negative(np.log10(P_given_alpha))
        unary_sink = np.negative(np.log10(P_given_label))
        unary_sink[np.where(original_image == label_dict[alpha])]=np.inf
        ### 4) Add terminal edges
        g.add_grid_tedges(pixel_nodes, unary_source , unary_sink)
        ### 5) Add Node edges
        I[np.where(I == 0)] = 1
        I[np.where(I == 128)] = 2
        I[np.where(I == 255)] = 3
        for x in range(I.shape[1]):
            for y in range(I.shape[0]):
                pixel = I[y][x]
                if x + 1 < I.shape[1] :
                    pixel_right = I[y][x + 1]
                    if pixel == alpha and pixel_right in non_alpha:
                        g.add_edge(pixel_nodes[y][x],pixel_nodes[y][x+1],0,1)
                    elif pixel_right == alpha and pixel in non_alpha:
                        g.add_edge(pixel_nodes[y][x],pixel_nodes[y][x+1],0,1)
                    elif pixel in non_alpha and pixel_right in non_alpha and pixel==pixel_right:
                        g.add_edge(pixel_nodes[y][x],pixel_nodes[y][x+1],1,1)
                    elif pixel in non_alpha and pixel_right in non_alpha and pixel != pixel_right:
                        new_node = g.add_nodes(1)
                        g.add_edge(pixel_nodes[y][x],new_node[0],1,np.inf)
                        g.add_edge(new_node[0],pixel_nodes[y][x+1],np.inf,1)
                        g.add_tedge(new_node[0],0,1)
                if y + 1 < I.shape[0]:
                    pixel_bottom = I[y+1][x]
                    if pixel == alpha and pixel_bottom in non_alpha:
                        g.add_edge(pixel_nodes[y][x], pixel_nodes[y+1][x], 0, 1)
                    elif pixel_bottom == alpha and pixel in non_alpha:
                        g.add_edge(pixel_nodes[y][x],pixel_nodes[y+1][x], 0, 1)
                    elif pixel in non_alpha and pixel_bottom in non_alpha and pixel == pixel_bottom:
                        g.add_edge(pixel_nodes[y][x], pixel_nodes[y+1][x], 1, 1)
                    elif pixel in non_alpha and pixel_bottom in non_alpha and pixel != pixel_bottom:
                        new_node = g.add_nodes(1)
                        g.add_edge(pixel_nodes[y][x], new_node[0], 1, np.inf)
                        g.add_edge(new_node[0], pixel_nodes[y+1][x], np.inf, 1)
                        g.add_tedge(new_node[0], 0, 1)
        g.maxflow()
        sgm = g.get_grid_segments(pixel_nodes)
        original_image[np.where(sgm)]=label_dict[alpha]
        fig,(ax1,ax2)=plt.subplots(1,2)
        ax1.title.set_text("Original Image")
        ax1.imshow(original_image_unchanged,cmap=cm.get_cmap("gray"))
        ax2.title.set_text("Alpha-expansion for alpha="+str(alpha))
        ax2.imshow(original_image, cmap=cm.get_cmap("gray"))
        plt.show()
        # cv2.imshow('Original Img', original_image_unchanged)
        # cv2.imshow('Denoised Img ' + str(alpha), original_image), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    # ### Call solution for question 3
    mrf_binary(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    mrf_binary(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    mrf_binary(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    alpha_expansion(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()



