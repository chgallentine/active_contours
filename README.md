# Active Contours

## Energies

Energies are calculated on a 7x7 grid centered on each contour point. The minimum energy is selected as the next coordinate for the contour point.

## Energy 1, Internal
### Distances

The first energy term encourages the contour to either spread or shrink. This calculates the squared distance from each potential energy point to each other contour point and sums them.

## Energy 2, Internal
### Deviation

This attempts to equalize the distance between each contour point and its neighbours. It is the square of the absolute difference between the distance between the contour point and its next and prior points.

## Energy 3, External
### Gradient

The gradient is formed by the Laplacian, Sobel, Canny, or other gradient forming convolutions. Inverting the normalized array's values 1.0 - (0.0-1.0) causes the gradient to pull toward brighter pixels.