# vision-ary
This repository consists of my work on various topics in Computer Vision.

## 1. Level Set Based Contour Fitting
This project concerns with object segmentation the image using level-sets with an geodesic active contour. Firstly, a contour is initialized by a circle around the object and the a signed distance transform is evaluated to initialize the level-set function.Then on, the geodesic contour is optimized by gradient descent. The gradual change in the level-set function is visualized to show how the contour converges to the boundary of the object.
The final result looks like this
![Level Set in action](Level-sets.gif)

## 2. Snakes Based Contour Fitting (Dynamic Programming)
This project concerns with object segmentation of the object in the image using snakes ( Active Contours).Firstly, a snake is initiaslized  by a circle around the object and the energy function consisting of the elasticity and smoothness term is optimized using dynamic programming. The elastic term  is a pairwise cost, penalizing deviation from the average distance between pairs of nodes. Finally converges of the snake to the boundary of the object is visualized.
The final result looks like this
![Snakes in action](Snakes-contour.gif)

## 3. Markov Random Fields for Image Denoising (Graph-Cuts)
This project concerns with image denoising using Markov Random Fields which uses the Markov property that a nodes state only depends on its neighboring nodes. Markov Random Fields uses the prior that neighhboring pixels are smooth. If the neighboring pixels are different, they are penalized. The image is converted in a graph data structure using pixels as nodes. Each pixel (node) are connected to the “source node” and the “sink node” with directed edges as well as the directed edges between its left, top, right and bottom neighboring pixel. The uanry term is the likelihood term derived using Bernoulli distribuiton. While the pairwise terms are derived according the smoothness prior discussed above. If the pixels are same, the pairwise term is small, otherwise its large. Min-cut algorithm is then applied on this graph structure to ensure we get a MAP estimate. The idea is extended to more than one labels using alpha expansion, which approximates the solution for non-conve Potts model.
The final result looks like this
![MRF in action](denoise.gif)

## 4. Iterative Closest Point Algorithm for Template Shape Model
In this approach, we iteratively find the closest point on the edge. At each iteration, once we have found the
closest edge points, we apply an affine transformation on the original landmark points, to get new landmmark
points that are closer to the edge points. We claim to have converged when the psi stops changing.

1. Find edges of the hand image using Canny
2. Pre-compute the distance transform of the image
3. For each point w find the closest point in the edges.(Correspondence)
   - w = Point on the shape model. (trasnformed)
   - E = Point in the edge list.
   - D : Distance transform at point (w)
   - G = Find the gradient of the distance transform.
   - x = (w - (D/Magnitude(G))*(Gx*Gy))
4. Find an affine transformation using closed form solution.
![ICP in action](icp.gif)


## 4. Statistical Shape Modeling with Principal Component Analysis
In this problem, we have been given various samoples from the shape space, ( hands in this example ) and the goal is to learn the statistical properties of from these samples , more specifically, the mean and variance of the distribution of the landmark points of the shapes. This is then projected onto a sub-space using PCA such that this new model, captures the primary features of the shape and drops the noise from the training samples. Given a new sample, the PCA model can fit a shape using the sub-space model efficiently using ICP.
   ### A. Training the PCA Model : 
      1. In statistical shape modelling, we first calculate the mean shape. 
      2. Then find the covariance matrix of the points.
      3. Then we find the eigen value and eigen vector subspace.
      4. Finally we caclulate phi which represents set of eigen vectors.
 ![SSP in action](PCA_Shape.png)
   ### B. Inference on a new Sample :
      1. Calculate the w = mu + phi.h
      2. Find the psuedo inverse components. 3. Caclulate psi based on test points.
      4. Transform w using the new psi.
      5. Caclulate the components of h.
      6. Reiterate if shape not converged.
 ![SSP in action](SSP.gif)

## 5. SIFT Feature Matching
SIFT is a classic Feature Extraction Algorithm that produces a scale and rotation invariant, local feature descriptor of keypoints. These descriptors
can be used to find matches between two images of the same scene , which has further applications in depth calculations and image stitching. This project
uses 2 images of a scene containing mountains and finds keypoints with their respective SIFT descriptors. Then it finds matches using Euclidean distance
between features. If also eliminates ambiguity by using second-best match ratio.
![SIFT in Action](mapped_maches.png)

## 6. Harris and Forstner Corner Detection Algorithm
Corners are keypoints in image that show high change in intensity values if we move along any direction from that position. Corners help in finding
locally distinct points in an image, required for finding important features. The primary idea to detect if a given point is a corner, is to find the Jacobian of the local neighborhood around that point.
Then we evaluate  the Eigen Values of that Jacobian. If both these values are large, it indicates that the point is a corner. Harris and Forstner have proposed algorithms for the same, which have been implemented in this
project.
![Corner Detection in Action](corner.gif)