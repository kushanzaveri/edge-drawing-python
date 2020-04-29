import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

VERTICAL = 1
HORIZONTAL = -1
anchor_thresh = 8
scan_interval = 3
ksize_gaussian = 5
sigma_gaussian = 1
ksize_sobel = 3
gradient_thresh = 36

class EdgeDrawing:
  def __init__(self):
    self.anchors = []
    self.G = []
    self.ED = []
    self.visited = []
    self.edgels = []
    self.ROWS = -1
    self.COLS = -1

  def GaussianFilter(self, filename: str): 
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    dst = cv2.GaussianBlur(src=img, ksize=(ksize_gaussian, ksize_gaussian), sigmaX=sigma_gaussian)
    [self.ROWS, self.COLS] = np.shape(dst)
    return dst

  def __GradientAndEdgeDirectionMap(self, G_r, G_c):
    self.G = np.hypot(G_r, G_c)
    self.G[self.G < gradient_thresh] = 0
    self.ED = np.sign(np.abs(G_r) - np.abs(G_c))
    self.ED[self.G < gradient_thresh] = 0

  def SobelOperator(self, image):
    G_r = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize_sobel) # going along the row - vert
    G_c = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize_sobel) # going along the col - horiz
    self.__GradientAndEdgeDirectionMap(G_r, G_c)
  
  def __isAnchor(self, row, col):
    current = self.G[row][col]
    if self.ED[row][col] == HORIZONTAL:
      top, bottom  = self.G[row - 1][col], self.G[row + 1][col]
      return current - top >= anchor_thresh and current - bottom >= anchor_thresh
    if self.ED[row][col] == VERTICAL:
      left, right = self.G[row][col-1], self.G[row][col + 1]
      return current - left >= anchor_thresh and current - right >= anchor_thresh
    return False

  def FindAnchors(self):
    for row in range(1, self.ROWS - 1, scan_interval):
      for col in range(1, self.COLS - 1, scan_interval):
        if self.__isAnchor(row, col):
          self.anchors.append([row, col])

  def __getLR(self, row, col):
    top, mid, bot = self.G[row - 1][col],  self.G[row][col], self.G[row + 1][col]
    if top > mid and top > bot:
      return row - 1
    if bot > mid and bot > top:
      return row + 1
    return row

  def __getUD(self, row, col):
    left, mid, right = self.G[row][col - 1],  self.G[row][col], self.G[row][col + 1]
    if left > mid and left > right:
      return col - 1
    if right > mid and right > left:
      return col + 1
    return col

  def __getL(self, row, col):
    return -1 if col == 0 else self.__getLR(row, col - 1)
  
  def __getR(self, row, col):
    return -1 if col + 1 == self.COLS else self.__getLR(row, col + 1)

  def __getD(self, row, col):
    return -1 if row == 0 else self.__getUD(row - 1, col)

  def __getU(self, row, col):
    return -1 if row + 1 == self.ROWS else self.__getUD(row + 1, col)

  def __proceedLR(self, row, col, row_fn, col_fn):
    while True:
      self.visited[row][col] = True
      next_row = row_fn(row, col)
      next_col = col_fn(col)
      if next_row == -1 or next_col == -1:
        return
      row = next_row
      col = next_col
      if self.G[row][col] <= 0 or self.visited[row][col]:
        return
      if self.ED[row][col] == VERTICAL:
        break
    self.__proceed(row, col)

  def __proceedUD(self, row, col, row_fn, col_fn):
    while True:
      self.visited[row][col] = True
      next_col = col_fn(row, col)
      next_row = row_fn(row)
      if next_col == -1 or next_row == -1:
        return
      col = next_col
      row = next_row
      if self.G[row][col] <= 0 or self.visited[row][col]:
        return
      if self.ED[row][col] == HORIZONTAL:
        break
    self.__proceed(row, col)

  def __proceed(self, row, col):
    if self.visited[row][col]:
      return
    inc = lambda a: a + 1
    dec = lambda a: a - 1
    if self.ED[row][col] == HORIZONTAL:
      self.__proceedLR(row, col, self.__getL, dec)
      self.__proceedLR(row, col, self.__getR, inc)

    if self.ED[row][col] == VERTICAL:
      self.__proceedUD(row, col, dec, self.__getD)
      self.__proceedUD(row, col, inc, self.__getU)

  def ConnectAnchors(self):
    self.visited = np.zeros(np.shape(self.G), dtype = bool)
    for [row, col] in self.anchors:
      if not self.visited[row][col]:
        self.__proceed(row, col)

    for row in range(self.ROWS):
      for col in range(self.COLS):
        if self.visited[row][col]:
          e.edgels.append([row, col])
    
if __name__=="__main__":
  e = EdgeDrawing()
  img = e.GaussianFilter("sample.png")
  e.SobelOperator(img) 
  e.FindAnchors()
  e.ConnectAnchors()
  
  x, y = [], []
  for [x_, y_] in e.edgels:
    x.append(y_)
    y.append(x_)

  plt.scatter(x, y, marker=',', color='r')
  plt.imshow(e.G, cmap='gray')
  #plt.show()
