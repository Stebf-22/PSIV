from nilearn import plotting
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)
fig, axs = plt.subplots(3)

plotting.plot_epi("/home/user/PSIV/02-Cancer_diagnosis/data/NoduleSegmentation/LENS_P15_13_11_2015/INSP_SIN/CTDataROI.nii", display_mode='ortho',
                       cut_coords=[36, -27, 60],
                       title="display_mode='ortho', cut_coords=[36, -27, 60]",
                       axes = axs[0])


plt.show()