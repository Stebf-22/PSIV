import os
import IOFunctions as io
import copy
from VolumeCutBrowser import *


def crop_image(img: np.ndarray, affine: np.ndarray, center: np.ndarray, radius: np.ndarray, padding: int = 30,
               show: bool = True, log: bool = True, already_roi: bool = False, idx_roi: int = 0) -> np.ndarray:
    """
    Image cropping and plotting

    :param img: Image
    :param affine: Affine transform matrix
    :param center: Center point of crop region
    :param radius: Radius of crop region
    :param padding: Padding
    :param show: Display image
    :param log: Logging information
    :param already_roi: If ROI the area will be displayed
    :param idx_roi: Index of ROI to display

    :return: Returns the area of interest (roi)
    """

    if not already_roi:
        # Image transformations
        mat = np.linalg.inv(affine)
        point = mat.dot(center)
        point2 = radius
        if log:
            print(point, point2)

        point_a = point + point2
        point_b = point - point2

        if log:
            print(point_a, point_b, point2)

        # Borders of roi with padding
        x1, x2 = int(min([point_b[0], point_a[0]])), int(max([point_b[0], point_a[0]]))
        y1, y2 = int(min([point_b[1], point_a[1]])), int(max([point_b[1], point_a[1]]))
        z1, z2 = int(min([point_b[2], point_a[2]])), int(max([point_b[2], point_a[2]]))

        roi = img[x1 - padding:x2 + padding, y1 - padding: y2 - padding, z1:z2]

        # Display the area
        if show:
            mat = img[:, :, int(point[2])]
            mat2 = copy.deepcopy(img[:, :, int(point[2])])
            mat[x1 - padding:x2 + padding , y1 - padding:y2 + padding] = 0
            plt.figure()
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(mat)
            ax[1].imshow(mat2)
            plt.show()
            VolumeCutBrowser(roi)

        return roi

    else:
        if show:
            plt.imshow(img[:, :, idx_roi])
            plt.show()

        return img


def read_img_and_points(path: str, prefix: str = ''):
    img_path = [x for x in os.listdir(path) if 'nii' in x and prefix in x][0]
    img, trf = io.load_nifti_img(path + img_path)

    if 'Seg' not in prefix:
        acsv = [x for x in os.listdir(path) if 'acsv' in x][0]
        affine = trf['affine']
        with open(path + acsv, 'r') as hdlr:
            points = [x for x in hdlr.readlines() if 'point|' in x]
            separate = lambda x: np.asarray([int(float(y)) for y in x.split('|')[1:5]] )
            center = separate(points[0])
            radius = separate(points[1])

        return img, affine, center, radius

    return img, 1, 1, 1


if __name__ == '__main__':
    ### Per llegir FULL image no fer servir prefix no fer servir already_roy ni idx_roy
    ### Per llegir segmentacions fer servir el que no abans
    img1, affine1, center1, radius1 = read_img_and_points("../data/NoduleSegmentation/LENS_P3_16_07_2015/INSP_SIN/",
                                                      prefix='Seg')
    crop_image(img1, affine1, center1, radius1, padding=0, already_roi=True, idx_roi=455)

