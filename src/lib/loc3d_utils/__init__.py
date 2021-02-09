"""
written by jerry
2021/02/09
"""

class Loc3DParams:
    def __init__(fx, fy, img_size=(1088,608), H=1.7):
        self.fx, self.fy = fx, fy
        self.img_width, self.img_height = img_size
        self.H = H