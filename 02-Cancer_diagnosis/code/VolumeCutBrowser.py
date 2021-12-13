import numpy as np
import matplotlib.pyplot as plt
import sys


################################################################################
# 
# EXAMPLE:
# # Load NII Volume
# from BasicVisualization.DICOMViewer import VolumeSlicer
# import BasicIO.NiftyIO
# import os
# ServerDir='Y:\Shared\Guille'; NIIFile='LIDC-IDRI-0305_GT1_1.nii.gz'
# niivol,_=NiftyIO.readNifty(os.path.join(ServerDir,NIIFile))
#
# VolumeCutBrowser(niivol)


class VolumeCutBrowser:

    def __init__(self, ims: np.ndarray, idx: int = None, ims_seg=None, cut: str = 'SA'):
        self.IMS = ims
        self.idx = idx
        self.IMSSeg = ims_seg
        self.drawContour = True
        self.Cut = cut
        if ims_seg is None:
            self.drawContour = False

        if idx is None:
            if self.Cut == 'SA':
                self.idx = np.int(np.round(self.IMS.shape[2] / 2))

        elif self.Cut == 'Sag':
            self.idx = np.int(np.round(self.IMS.shape[0] / 2))

        elif self.Cut == 'Cor':
            self.idx = np.int(np.round(self.IMS.shape[1] / 2))

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.draw_scene()

    def press(self, event):

        sys.stdout.flush()
        if event.key == 'x':
            self.idx -= 1
            self.idx = max(0, self.idx)
            self.draw_scene()

        elif event.key == 'z':
            self.idx += 1
            if self.Cut is 'SA':
                mx = self.IMS.shape[2] - 1
            elif self.Cut is 'Sag':
                mx = self.IMS.shape[0] - 1
            elif self.Cut is 'Cor':
                mx = self.IMS.shape[1] - 1

            self.idx = min(mx, self.idx)
            self.draw_scene()

    def draw_scene(self):
        self.ax.cla()

        if self.Cut is 'Sag':
            img = np.squeeze(self.IMS[self.idx, :, :])

        elif self.Cut is 'Cor':
            img = np.squeeze(self.IMS[:, self.idx, :])

        else:   # self.Cut is 'SA'
            img = self.IMS[:, :, self.idx]

        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(f'Cut: {str(self.idx)}. Press "x" to decrease; "z" to increase')

        if self.drawContour:
            self.ax.contour(img, [0.5], colors='r')

            self.fig.canvas.draw()
