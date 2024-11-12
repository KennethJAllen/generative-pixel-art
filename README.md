# PixelContraction
## Goal
The goal of this project is to create a scipt that takes text-to-art generated "pixel art" image as an input and downscales that image to the true resolution of the pixel art being depicted.

## Problems
There are many challenges involved in solving this problem.

1. There is a lot of noise in the text-to-art generated "pixel art" images. The edges of the pixels heuristically seem fairly well defined, that being said, random unrecognizable artifacts are fairly common. Aditionally, the parts of the images representing pixels may be wavy and not fit to a grid. A possible solution is to denoise with high-pass or low-pass filters.

2. What is the true resolution of the image? It is possible to check the true resolution manually by finding a parts of the image representing a group of pixels, and counting the number of actual pixels in that area. Some possible solutions for this are finding vertical and horizontal frequency peaks in the fourier transform, edge detection filters, or finding the rank of the image via singular value decomposition.

3. What should the color pallet of the image be? Pixel art images have a small set of colors to select from, and small variations of color should be avoided between pixels. At the same time, the distinct features of the pixel art image should not be lost when discretizing the colors. Possible solutions to these problems are to use clustering algorithms to find a given number of clusters in color space, but vectorizing the image may lose the structure of adjacent pixels. Another possible solution may be to randomly sample pixels in each row/column.
