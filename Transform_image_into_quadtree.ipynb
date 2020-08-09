{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Quad tree from image\n",
    "\n",
    "The main goal is to create a quad tree from an image and try to display it.\n",
    "\n",
    "## Load an image\n",
    "\n",
    "First thing we're going to do is to load an image and display it using matplotlib. It's a CC image from Flickr user yokok, source https://www.flickr.com/photos/yokok/7034360407"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1365, 2048, 3)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import numpy as np\n",
    "\n",
    "img = mpimg.imread('split-Croatia.jpg')\n",
    "img.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f4b01e7e550>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(img)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Split image in 4\n",
    "\n",
    "A big part of how the algorithm works is splitting the image into 4 quarters and calculate the mean color of each part to create a level of the tree. Let's split Split in 4 and calculate the mean color of each quarter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(683, 1024, 3)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from operator import add\n",
    "from functools import reduce\n",
    "\n",
    "def split4(image):\n",
    "    half_split = np.array_split(image, 2)\n",
    "    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)\n",
    "    return reduce(add, res)\n",
    "\n",
    "split_img = split4(img)\n",
    "split_img[0].shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f4b01ce7fd0>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 4 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axs = plt.subplots(2, 2)\n",
    "axs[0, 0].imshow(split_img[0])\n",
    "axs[0, 1].imshow(split_img[1])\n",
    "axs[1, 0].imshow(split_img[2])\n",
    "axs[1, 1].imshow(split_img[3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Reconstruct the full image from the split\n",
    "This will be useful when we want to display the image back, as the quadtree will store the images split into 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def concatenate4(north_west, north_east, south_west, south_east):\n",
    "    top = np.concatenate((north_west, north_east), axis=1)\n",
    "    bottom = np.concatenate((south_west, south_east), axis=1)\n",
    "    return np.concatenate((top, bottom), axis=0)\n",
    "\n",
    "full_img = concatenate4(split_img[0], split_img[1], split_img[2], split_img[3])\n",
    "plt.imshow(full_img)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calculate the mean\n",
    "\n",
    "Now we want to calculate the mean of all the parts of the split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 78 146 219]\n",
      "  [ 69 128 195]]\n",
      "\n",
      " [[ 58  76 109]\n",
      "  [135 113 108]]]\n"
     ]
    },
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def calculate_mean(img):\n",
    "    return np.mean(img, axis=(0, 1))\n",
    "\n",
    "means = np.array(list(map(lambda x: calculate_mean(x), split_img))).astype(int).reshape(2,2,3)\n",
    "print(means)\n",
    "plt.imshow(means)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## QuadTree data structure\n",
    "\n",
    "Now let's create a data structure that will allow us to construct our quad tree. It's a recursive calculation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def checkEqual(myList):\n",
    "    first=myList[0]\n",
    "    return all((x==first).all() for x in myList)\n",
    "\n",
    "class QuadTree:\n",
    "    \n",
    "    def insert(self, img, level = 0):\n",
    "        self.level = level\n",
    "        self.mean = calculate_mean(img).astype(int)\n",
    "        self.resolution = (img.shape[0], img.shape[1])\n",
    "        self.final = True\n",
    "        \n",
    "        if not checkEqual(img):\n",
    "            split_img = split4(img)\n",
    "            \n",
    "            self.final = False\n",
    "            self.north_west = QuadTree().insert(split_img[0], level + 1)\n",
    "            self.north_east = QuadTree().insert(split_img[1], level + 1)\n",
    "            self.south_west = QuadTree().insert(split_img[2], level + 1)\n",
    "            self.south_east = QuadTree().insert(split_img[3], level + 1)\n",
    "\n",
    "        return self\n",
    "    \n",
    "    def get_image(self, level):\n",
    "        if(self.final or self.level == level):\n",
    "            return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))\n",
    "        \n",
    "        return concatenate4(\n",
    "            self.north_west.get_image(level), \n",
    "            self.north_east.get_image(level),\n",
    "            self.south_west.get_image(level),\n",
    "            self.south_east.get_image(level))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "quadtree = QuadTree().insert(img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png":  "",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png":  ""
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(quadtree.get_image(1))\n",
    "plt.show()\n",
    "plt.imshow(quadtree.get_image(3))\n",
    "plt.show()\n",
    "plt.imshow(quadtree.get_image(7))\n",
    "plt.show()\n",
    "plt.imshow(quadtree.get_image(10))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
