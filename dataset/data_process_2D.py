import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image


def get_2d_images(ct_path, ct_label_path, test=False):
    k = 0
    file = "test" if test else "train"
    for i in range(len(ct_path)):

        nifti_ct = nib.load(ct_path[i])
        ct_3d = nifti_ct.get_fdata()
        nifti_ct_label = nib.load(ct_label_path[i])
        ct_label_3d = nifti_ct_label.get_fdata()

        for z in range(5, ct_3d.shape[2] - 5):
            ct_slice = ct_3d[:, :, z]
            ct_label_slice = ct_label_3d[:, :, z]
            if ct_label_slice.max() != ct_label_slice.min() and ct_slice.max() != ct_slice.min():

                ct_image = (((ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())) * 255).astype(np.uint8)

                ct_image = Image.fromarray(ct_image)
                ct_label = Image.fromarray(ct_label_slice)


                ct_image.save(f'/data/private/autoPET/autopet_3d_only_crop/{file}/image/slice_{k}.png')
                ct_label.save(f'data/private/autoPET/autopet_3d_only_crop/{file}/label/slice_{k}.png')
                k += 1

def list_images(path):
    image_path = []
    label_path = []
    # read files names
    image_names = os.listdir(path)
    #image_names = sorted(list(filter(lambda x: x.endswith('image.nii.gz'), names)))
    #label_names = list(filter(lambda x: x.endswith('label.nii.gz'), names))

    for i in range(len(image_names)):
        image_path.append(os.path.join(path, image_names[i], 'ct.nii.gz'))
        label_path.append(os.path.join(path, image_names[i], 'label.nii.gz'))

    return image_path, label_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/autopet_2d/train/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/val/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/val/label', exist_ok=True)

    path_image_train = "/data/private/autoPET/autopet_3d_only_crop/train"
    path_image_test = "/data/private/autoPET/autopet_3d_only_crop/test"

    ct_image_train, ct_label_train = list_images(path_image_train)
    ct_image_test, ct_label_test = list_images(path_image_test)

    get_2d_images(ct_image_test, ct_label_test)
    get_2d_images(ct_image_train, ct_label_train, test=True)
