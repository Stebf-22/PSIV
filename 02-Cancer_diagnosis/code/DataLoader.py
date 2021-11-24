from torch.utils.data import Dataset, DataLoader
import os
from ImageReader import read_img_and_points, crop_image
import matplotlib.pyplot as plt
from ImageTransformations import resize


class SegmentationDataset(Dataset):Cod
    """
    Pytorch Dataset Class for data handling.
    """

    def __init__(self, path_segmentation: str, path_diagnosi: str):

        self.path_segmentation = path_segmentation
        self.path_diagnosi = path_diagnosi
        self.sort = lambda y: sorted(y, key=lambda x: int(x.split('_')[1]) if not ('P' in x.split('_')[1]) else int(
            x.split('_')[1][1:]))
        self.segmentation_dirs = [x for x in os.listdir(self.path_segmentation) if 'txt' not in x]
        self.actual = None
        self.actual_offset = 0
        self.image_counter = 0

    def _read_images(self, idx: int) -> dict:

        patient = self.segmentation_dirs[idx]
        base = {'data': None, 'GT': None, 'GT2': None}
        affine = None

        for elements in sorted(os.listdir(self.path_segmentation + patient + '/INSP_SIN/')):
            if 'ROI' in elements:
                data, affine, center, radius = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'),
                                                                   prefix='CTD')
                roi = crop_image(data, affine, center, radius, show=False, padding=20, log=False)
                base['data'] = roi

            if 'Seg_Nodule1' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix='Seg_Nodule1')
                roi = crop_image(data[0], affine, center, radius, show=False, padding=20, log=False)
                base['GT'] = roi

            if 'Seg_Nodule2' in elements:
                data = read_img_and_points((self.path_segmentation + patient + '/INSP_SIN/'), prefix='Seg_Nodule2')
                roi = crop_image(data[0], affine, center, radius, show=False, padding=20, log=False)
                base['GT2'] = roi

        self.actual = base

        return base

    def __getitem__(self, idx):
        pure_idx = idx
        if self.actual is None:
            img = self._read_images(self.image_counter)
            print(f"idx is {idx}")
            res = {'img': resize(img['data'][:, :, idx], (240, 240)), 'GT': resize(img['GT'][:, :, idx], (64, 64))}

        else:
            print(idx)
            idx -= self.actual_offset
            print(idx, self.actual['data'].shape[2])
            if idx >= self.actual['data'].shape[2]:
                self.actual_offset += idx

                print(f"idx is :", idx, "actual offset: ", self.actual_offset)
                pure_idx -= self.actual_offset
                idx = pure_idx
                self._read_images(self.image_counter)
                self.image_counter += 1
                print(f"here")
            print(f"Here")
            img = self.actual
            res = {'img': resize(img['data'][:, :, idx], (240, 240)), 'GT': resize(img['GT'][:, :, idx], (64, 64))}

        return res

    def __len__(self):
        # return count_slices("/home/user/PSIV/02-Cancer_diagnosis/data/NoduleSegmentation", "CTD",segmentation=True)
        return 1098


if __name__ == '__main__':
    path_segmentation = "../data/NoduleSegmentation/"
    path_diagnosi = "../data/Diagnosis/"
    dataset = SegmentationDataset(path_segmentation, path_diagnosi)
    dataloader = DataLoader(dataset, batch_size=30)
    d = {}
    for x in range(30, 31):
        data = dataset[x]
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(data['img'])
        axarr[1].imshow(data['GT'])
        plt.show()
        break