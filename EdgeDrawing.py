import cv2
import numpy as np
from matplotlib import pyplot as plt

VERTICAL = 1
HORIZONTAL = -1
anchor_thresh = 8
scan_interval = 3
ksize_gaussian = 5
sigma_gaussian = 1
ksize_sobel = 3
gradient_thresh = 400
# f = open("out.txt", "x")
class EdgeDrawing:
  def __init__(self):
    self.anchors = []
    self.G = []
    self.ED = []
    self.included = []
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

  def __proceedLeft(self, row, col):
#    f.write("LSTART %d %d\n"%(row, col))
    while True:
      self.included[row][col] = True
      next_row = self.__getL(row, col)
      if next_row == -1:
        return
      row = next_row
      col -= 1
#      f.write("L --  %d %d\n"%(row, col))
      if self.G[row][col] <= 0 or self.included[row][col]:
        return
      if self.ED[row][col] == VERTICAL:
#        f.write("L // \n")
        break
    self.__proceed(row, col)

  def __proceedRight(self, row, col):
#    f.write("RSTART %d %d\n"%(row, col))
    while True:
      self.included[row][col] = True
      next_row = self.__getR(row, col)
      if next_row == -1:
        return
      row = next_row
      col += 1
#      f.write("R --  %d %d\n"%(row, col))
      if self.G[row][col] <= 0 or self.included[row][col]:
        return
      if self.ED[row][col] == VERTICAL:
#        f.write("R // \n")
        break
    self.__proceed(row, col)

  def __proceedUp(self, row, col):
#    f.write("USTART %d %d\n"%(row, col))
    while True:
      self.included[row][col] = True
      next_col = self.__getU(row, col)
      if next_col == -1:
        return
      col = next_col
      row += 1
#      f.write("U --  %d %d\n"%(row, col))
      if self.G[row][col] <= 0 or self.included[row][col]:
        return
      if self.ED[row][col] == HORIZONTAL:
#        f.write("U // \n")
        break
    self.__proceed(row, col)

  def __proceedDown(self, row, col):
#    f.write("DSTART %d %d\n"%(row, col))
    while True:
      self.included[row][col] = True
      next_col = self.__getD(row, col)
      if next_col == -1:
        return
      col = next_col
      row -= 1
      #f.write("D --  %d %d\n"%(row, col))
      if self.G[row][col] <= 0 or self.included[row][col]:
        return
      if self.ED[row][col] == HORIZONTAL:
#        f.write("D // \n")
        break
    self.__proceed(row, col)

  def __proceed(self, row, col):
    if self.included[row][col]:
      return
    if self.ED[row][col] == HORIZONTAL:
      self.__proceedLeft(row, col)
      self.__proceedRight(row, col)

    if self.ED[row][col] == VERTICAL:
      self.__proceedDown(row, col)
      self.__proceedUp(row, col)

  def ConnectAnchors(self):
    self.included = np.zeros(np.shape(self.G), dtype = bool)
    for [row, col] in self.anchors:
      if not self.included[row][col]:
        self.__proceed(row, col)

    for row in range(self.ROWS):
      for col in range(self.COLS):
        if self.included[row][col]:
          e.edgels.append([row, col])
    
if __name__=="__main__":
  e = EdgeDrawing()
#  img = e.GaussianFilter("DSC_0426.JPG")
  img = e.GaussianFilter("Mickey_Mouse.jpg")
  #img = e.GaussianFilter("lenna.png")
  e.SobelOperator(img) 
  e.FindAnchors()
  e.ConnectAnchors()
  x, y = [], []
  for [x_, y_] in e.edgels:
    x.append(y_)
    y.append(x_)
  
  x_2, x_3 = [], []
  y_2, y_3 = [], []
  for r in range(e.ROWS):
    for c in range(e.COLS):
      if e.ED[r][c] == HORIZONTAL:
        x_2.append(c)
        y_2.append(r)
      elif e.ED[r][c] == VERTICAL:
        x_3.append(c)
        y_3.append(r)

 # plt.figure(figsize=(10, 10))
#  f.close()
 # plt.scatter(x_2, y_2, marker = '3', color='g')
 # plt.scatter(x_3, y_3, marker = '2', color='b')
#  plt.scatter(x, y, marker = ',', color='y')
#  plt.gca().set_aspect('equal', adjustable='box')
#  ax = plt.gca()
#  ax.invert_yaxis()
 # plt.imshow(e.G, cmap='gray')
  fig = plt.figure()
  ax = fig.add_subplot(111)
  p = ax.plot(x[:100], y[:100], 'b')
  ax.invert_yaxis()
