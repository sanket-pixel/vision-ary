# vision-ary
This repository consists of my work on various topics in Computer Vision.

# 1. Level Set Based Contour Fitting
This project concerns with object segmentation the image using level-sets with an geodesic active contour. Firstly, a contour is initialized by a circle around the object and the a signed distance transform is evaluated to initialize the level-set function.Then on, the geodesic contour is optimized by gradient descent. The gradual change in the level-set function is visualized to show how the contour converges to the boundary of the object.
The final result looks like this
![Level Set in action](Level-sets.gif)

# 1. Snakes Based Contour Fitting (Dynamic Programming)
This project concerns with object segmentation of the object in the image using snakes ( Active Contours).Firstly, a snake is initiaslized  by a circle around the object and the energy function consisting of the elasticity and smoothness term is optimized using dynamic programming. The elastic term  is a pairwise cost, penalizing deviation from the average distance between pairs of nodes. Finally converges of the snake to the boundary of the object is visualized.
The final result looks like this
![Level Set in action](Snakes.gif)

