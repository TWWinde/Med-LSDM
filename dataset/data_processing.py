import os
import nibabel as nib
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np


def crop_center(img, new_x, new_y):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, :]


def crop_block(img, new_x, new_y, start_z, length):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length]


def save_cropped(files, folder, crop_size, crop_2_block=False, length=32, stride=16):
    for file_path in files:
        img = nib.load(file_path)
        data = img.get_fdata()
        print(data.shape)
        if crop_2_block:
            for i in range(0, data.shape[2]-length, stride):
                cropped_data = crop_block(data, *crop_size, i, length)
                cropped_img = nib.Nifti1Image(cropped_data, affine=img.affine)
                output_path = os.path.join(folder, os.path.basename(file_path).split('.')[0]+f'_{i//length}.'+os.path.basename(file_path).split('.')[1]+'.' + os.path.basename(file_path).split('.')[2])
                nib.save(cropped_img, output_path)
                print('finished', output_path)
        else:
            cropped_data = crop_center(data, *crop_size)
            cropped_img = nib.Nifti1Image(cropped_data, affine=img.affine)
            output_path = os.path.join(folder, os.path.basename(file_path))
            nib.save(cropped_img, output_path)


def process_images(source_folder, train_folder, test_folder, crop_size=(256, 256)):

    ct_train_folder = os.path.join(train_folder, 'ct')
    ct_test_folder = os.path.join(test_folder, 'ct')
    label_train_folder = os.path.join(train_folder, 'label')
    label_test_folder = os.path.join(test_folder, 'label')

    os.makedirs(ct_train_folder, exist_ok=True)
    os.makedirs(ct_test_folder, exist_ok=True)
    os.makedirs(label_test_folder, exist_ok=True)
    os.makedirs(label_train_folder, exist_ok=True)

    ct_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('0001.nii.gz')]

    ct_train_files, ct_test_files = train_test_split(ct_files, test_size=0.1, random_state=42)
    label_train_files = [path.replace('0001.nii.gz', '0002.nii.gz') for path in ct_train_files]
    label_test_files = [path.replace('0001.nii.gz', '0002.nii.gz') for path in ct_test_files]

    crop_2_block = True
    save_cropped(ct_train_files, ct_train_folder, crop_size, crop_2_block=crop_2_block)
    save_cropped(ct_test_files, ct_test_folder, crop_size, crop_2_block=crop_2_block)
    save_cropped(label_train_files, label_train_folder, crop_size, crop_2_block=crop_2_block)
    save_cropped(label_test_files, label_test_folder, crop_size, crop_2_block=crop_2_block)
    print('all finished')


def remove_artifacts(in_file, out_path):
    image = sitk.ReadImage(in_file)
    img_3d = sitk.GetArrayFromImage(image)
    blurred_image = sitk.SmoothingRecursiveGaussian(image, sigma=[7.0, 7.0, 7.0])
    image_array = sitk.GetArrayFromImage(blurred_image)
    binary_image_array = (image_array > -800).astype(np.uint8)

    binary_image = sitk.GetImageFromArray(binary_image_array)
    binary_image.CopyInformation(image)

    label_image = sitk.ConnectedComponent(binary_image)
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(label_image)
    largest_label = max(label_stats.GetLabels(), key=lambda l: label_stats.GetPhysicalSize(l))
    largest_component = sitk.BinaryThreshold(label_image, lowerThreshold=largest_label, upperThreshold=largest_label)
    largest_component_array = sitk.GetArrayFromImage(largest_component)

    img_3d[largest_component_array == 0] = -1024

    modified_image = sitk.GetImageFromArray(img_3d)
    modified_image.CopyInformation(image)

    name = os.path.basename(in_file)
    out_file = os.path.join(out_path, name)
    sitk.WriteImage(modified_image, out_file)


def iterator(in_path, out_path):

    os.makedirs(out_path, exist_ok=True)
    files = [os.path.join(in_path, f) for f in os.listdir(source_folder) if f.endswith('0001.nii.gz')]
    for file_path in files:
        remove_artifacts(file_path, out_path)


if __name__ == '__main__':

    source_folder = '/data/private/autoPET/imagesTr'
    train_folder = '/data/private/autoPET/autopet_3d/train'
    test_folder = '/data/private/autoPET/autopet_3d/test'
    out_folder = '/data/private/autoPET/imagesTr_wo_artifacts'

    iterator(source_folder, out_folder)


    #process_images(source_folder, train_folder, test_folder)

