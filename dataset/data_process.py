import os
from glob import glob

import nibabel as nib
import SimpleITK as sitk
import numpy as np
import random

Totalsegmentator_ct_classes = {
                    #"0": "background",
                    "kidney": "1",
                    "vessels":"2",
                    "gallbladder":"3",
                     "liver":"4",
                     "stomach":"5",
                     "pancreas":"6",
                     "adrenal":"7",
                    "lung":"8" ,
                     "vertebrae":"9",
                     "esophagus":"10",
                     "trachea":"11",
                     "heart":"12",
                    "pulmonary_artery":"13" ,
                    "small_bowel":"14" ,
                    "duodenum":"15" ,
                     "colon":"16",
                     "ribs":"17",
                     "humerus":"18",
                    "scapula":"19" ,
                    "clavicula":"20" ,
                     "femur":"21",
                     "hip":"22",
                     "sacrum":"23",
                    "autochthon":"24" ,
                    "iliopsoas":"25" ,
                     "urinary_bladder":"26",
                    "skin":"27" ,
                     "spleen":"28",
                     "fat":"29",
                     "skeletal_muscle": "30"

}



def crop_center(img, label, new_x, new_y):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, :], label[start_x:start_x + new_x, start_y:start_y + new_y, :]


def crop_block(img, label, new_x, new_y, start_z, length):
    assert img.shape == label.shape
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length], \
        label[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length]


def crop_block_single(img, new_x, new_y, start_z, length):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length]


def is_all_zero(array1, array2):
    return np.all(array1 == 0) or np.all(array2 == 0)


def save_cropped_autopet(image_in_files, image_out_files, crop_size, crop_2_block=False, length=32, stride=16):
    for image_path in image_in_files:
        label_path = image_path.replace('0001.nii.gz', '0002.nii.gz')
        # label_path = label_path.replace('imagesTr_wo_artifacts', 'imagesTr')  # only for data remove artifacts
        img = nib.load(image_path)
        label = nib.load(label_path)
        img_data = img.get_fdata()
        #label_data = label.get_fdata()
        #assert img_data.shape == label_data.shape, "Error: The shapes of image data and label data do not match."
        n = 0
        if crop_2_block:
            for i in range(0, img_data.shape[2] // 4):
                if img_data.shape[2] < 32:
                    continue
                number = random.randint(0, img_data.shape[2] - length)
                cropped_image, cropped_label = crop_block(img_data, label_data, *crop_size, number, length)
                if is_all_zero(cropped_image, cropped_label):
                    print("Array is all zeros. Skipping rescaling.")
                    continue
                cropped_img = nib.Nifti1Image(cropped_image, affine=img.affine)
                name = os.path.basename(image_path).split('.')[0]
                name = name.split('_')[0]
                img_output_path = os.path.join(image_out_files, name + f'_{n}.' + 'nii.gz')
                label_output_path = img_output_path.replace('ct', 'label')
                nib.save(cropped_img, img_output_path)
                cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
                nib.save(cropped_label, label_output_path)
                print('finished', img_output_path)
                print('finished', label_output_path)
                n += 1
            cropped_image, cropped_label = crop_block(img_data, label_data, *crop_size, img_data.shape[2] - length,
                                                      length)

            if is_all_zero(cropped_image, cropped_label):
                print("Array is all zeros. Skipping rescaling.")
                continue
            cropped_img = nib.Nifti1Image(cropped_image, affine=img.affine)
            name = os.path.basename(image_path).split('.')[0]
            name = name.split('_')[0]
            img_output_path = os.path.join(image_out_files, name + f'_{n}.' + 'nii.gz')
            label_output_path = img_output_path.replace('ct', 'label')
            nib.save(cropped_img, img_output_path)
            cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
            nib.save(cropped_label, label_output_path)
            print('finished', img_output_path)
            print('finished', label_output_path)
        else:
            cropped_image, cropped_label = crop_center(img_data, label_data, *crop_size)
            if is_all_zero(cropped_image, cropped_label):
                print("Array is all zeros. Skipping rescaling.")
                continue
            cropped_img = nib.Nifti1Image(cropped_image, affine=img.affine)
            name = os.path.basename(image_path).split('.')[0]
            name = name.split('_')[0]
            os.makedirs(os.path.join(image_out_files, name), exist_ok=True)
            img_output_path = os.path.join(image_out_files, name, 'ct.nii.gz')
            label_output_path = os.path.join(image_out_files, name, 'label.nii.gz')
            nib.save(cropped_img, img_output_path)
            cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
            nib.save(cropped_label, label_output_path)
            print('finished', img_output_path)
            print('finished', label_output_path)


def process_autopet_onlycrop(source_folder, train_folder, test_folder, crop_size=(256, 256)):
    ct_train_folder = os.path.join(train_folder)
    ct_test_folder = os.path.join(test_folder)

    os.makedirs(ct_train_folder, exist_ok=True)
    os.makedirs(ct_test_folder, exist_ok=True)

    ct_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('0001.nii.gz')]

    ct_train_files, ct_test_files = train_test_split(ct_files)

    crop_2_block = False
    save_cropped_autopet(ct_train_files, ct_train_folder, crop_size, crop_2_block=crop_2_block)
    save_cropped_autopet(ct_test_files, ct_test_folder, crop_size, crop_2_block=crop_2_block)

    print('all finished')


def process_images_autopet(source_folder, train_folder, test_folder, crop_size=(256, 256)):
    ct_train_folder = os.path.join(train_folder, 'ct')
    ct_test_folder = os.path.join(test_folder, 'ct')
    label_train_folder = os.path.join(train_folder, 'label')
    label_test_folder = os.path.join(test_folder, 'label')

    os.makedirs(ct_train_folder, exist_ok=True)
    os.makedirs(ct_test_folder, exist_ok=True)
    os.makedirs(label_test_folder, exist_ok=True)
    os.makedirs(label_train_folder, exist_ok=True)

    ct_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('0001.nii.gz')]

    ct_train_files, ct_test_files = train_test_split(ct_files)

    crop_2_block = True
    save_cropped_autopet(ct_train_files, ct_train_folder, crop_size, crop_2_block=crop_2_block)
    save_cropped_autopet(ct_test_files, ct_test_folder, crop_size, crop_2_block=crop_2_block)
    print('all finished')


def crop_block_3(array1, array2, array3, new_x, new_y, start_z, length):
    assert array1.shape == array2.shape == array3.shape, "Error: The shapes of arrays do not match."
    x, y, z = array1.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return array1[start_x:(start_x + new_x), start_y:(start_y + new_y), start_z:(start_z + length)], \
        array2[start_x:(start_x + new_x), start_y:(start_y + new_y), start_z:(start_z + length)], \
        array3[start_x:(start_x + new_x), start_y:(start_y + new_y), start_z:(start_z + length)]


def save_cropped_synthrad2023(ct_in_files, image_out_files, crop_size, crop_2_block=False, length=32, stride=8):
    for ct_path in ct_in_files:
        name = ct_path.split('/')[-2]
        mr_path = ct_path.replace('ct.nii.gz', 'mr.nii.gz')
        label_path = ct_path.replace('ct.nii.gz', 'label.nii.gz')
        ct = nib.load(ct_path)
        mr = nib.load(mr_path)
        label = nib.load(label_path)
        ct_data = ct.get_fdata()
        mr_data = mr.get_fdata()
        label_data = label.get_fdata()

        assert ct_data.shape == mr_data.shape == label_data.shape, "Error: The shapes of arrayys do not match."
        n = 0
        if crop_2_block:
            for i in range(0, ct_data.shape[2] // 4):
                if ct_data.shape[2] < 32:
                    continue
                number = random.randint(0, ct_data.shape[2] - length)
                cropped_ct, cropped_mr, cropped_label = crop_block_3(ct_data, mr_data, label_data, *crop_size, number, length)
                if is_all_zero(cropped_mr, cropped_label):
                    print("Array is all zeros. Skipping rescaling.")
                    continue
                ct_output_path = os.path.join(image_out_files, 'ct', name + f'_{n}.' + 'nii.gz')
                mr_output_path = os.path.join(image_out_files, 'mr', name + f'_{n}.' + 'nii.gz')
                label_output_path = os.path.join(image_out_files, 'label', name + f'_{n}.' + 'nii.gz')
                cropped_ct = nib.Nifti1Image(cropped_ct, affine=ct.affine)
                cropped_mr = nib.Nifti1Image(cropped_mr, affine=mr.affine)
                cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
                nib.save(cropped_ct, ct_output_path)
                nib.save(cropped_mr, mr_output_path)
                nib.save(cropped_label, label_output_path)
                print('finished', ct_output_path)
                print('finished', mr_output_path)
                print('finished', mr_output_path)
                n += 1
            cropped_ct, cropped_mr, cropped_label = crop_block_3(ct_data, mr_data, label_data, *crop_size,
                                                                 ct_data.shape[2] - length, length)
            if is_all_zero(cropped_mr, cropped_label):
                print("Array is all zeros. Skipping rescaling.")
                continue
            ct_output_path = os.path.join(image_out_files, 'ct', name + f'_{n}.' + 'nii.gz')
            mr_output_path = os.path.join(image_out_files, 'mr', name + f'_{n}.' + 'nii.gz')
            label_output_path = os.path.join(image_out_files, 'label', name + f'_{n}.' + 'nii.gz')
            cropped_ct = nib.Nifti1Image(cropped_ct, affine=ct.affine)
            cropped_mr = nib.Nifti1Image(cropped_mr, affine=mr.affine)
            cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
            nib.save(cropped_ct, ct_output_path)
            nib.save(cropped_mr, mr_output_path)
            nib.save(cropped_label, label_output_path)
            print('finished', ct_output_path)
            print('finished', mr_output_path)
            print('finished', label_output_path)
            print(n, ct_path)



def process_images_synthrad2023(source_folder, train_folder, test_folder, crop_size=(256, 256)):
    ct_train_folder = os.path.join(train_folder, 'ct')
    ct_test_folder = os.path.join(test_folder, 'ct')
    mr_train_folder = os.path.join(train_folder, 'mr')
    mr_test_folder = os.path.join(test_folder, 'mr')
    label_train_folder = os.path.join(train_folder, 'label')
    label_test_folder = os.path.join(test_folder, 'label')

    os.makedirs(ct_train_folder, exist_ok=True)
    os.makedirs(ct_test_folder, exist_ok=True)
    os.makedirs(mr_train_folder, exist_ok=True)
    os.makedirs(mr_test_folder, exist_ok=True)
    os.makedirs(label_test_folder, exist_ok=True)
    os.makedirs(label_train_folder, exist_ok=True)

    ct_files = [os.path.join(source_folder, f, 'ct.nii.gz') for f in os.listdir(source_folder) if f != 'overview']

    ct_train_files, ct_test_files = train_test_split(ct_files, test_proportion=0.1)

    crop_2_block = True
    save_cropped_synthrad2023(ct_train_files, train_folder, crop_size, crop_2_block=crop_2_block)
    save_cropped_synthrad2023(ct_test_files, test_folder, crop_size, crop_2_block=crop_2_block)
    print('all finished')


def train_test_split(path, test_proportion=0.1):
    train_path = []
    test_path = []
    for i in range(int(test_proportion * len(path))):
        test_path.append(path[i])
    print('test length', len(test_path))
    for i in range(int(test_proportion * len(path)), len(path)):
        train_path.append(path[i])
    print('train length', len(train_path))

    return train_path, test_path


def remove_artifacts(in_file, out_path):
    os.makedirs(out_path, exist_ok=True)
    image = sitk.ReadImage(in_file)
    img_3d = sitk.GetArrayFromImage(image)
    min = img_3d.min()
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
    # random_noise = np.random.uniform(-1, 1, img_3d.shape)

    img_3d[largest_component_array == 0] = min

    modified_image = sitk.GetImageFromArray(img_3d)
    modified_image.CopyInformation(image)

    name = os.path.basename(in_file)
    out_file = os.path.join(out_path, name)
    sitk.WriteImage(modified_image, out_file)


def pad_and_concatenate_image(input_image_path, output_image_path):
    image = sitk.ReadImage(input_image_path)
    array = sitk.GetArrayFromImage(image)

    c, a, b = array.shape[0], array.shape[1], array.shape[2]

    max_length = max(a, b)

    if a < max_length:
        pad1 = np.tile(array[:, 0:1, :], (1, (max_length - a) // 2, 1))
        pad2 = np.tile(array[:, -1:, :], (1, (max_length - a) - (max_length - a) // 2, 1))
        padded_array = np.concatenate((pad1, array, pad2), axis=1)
    else:
        padded_array = array

    if b < max_length:
        pad1 = np.tile(padded_array[:, :, 0:1], (1, 1, (max_length - b) // 2))
        pad2 = np.tile(padded_array[:, :, -1:], (1, 1, (max_length - b) - (max_length - b) // 2))
        padded_array = np.concatenate((pad1, padded_array, pad2), axis=2)

    final_image = sitk.GetImageFromArray(padded_array)
    final_image.SetSpacing(image.GetSpacing())
    final_image.SetOrigin(image.GetOrigin())
    final_image.SetDirection(image.GetDirection())
    sitk.WriteImage(final_image, output_image_path)


def iterator(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    files = [os.path.join(in_path, f) for f in os.listdir(in_path) if f.endswith('0001.nii.gz')]
    for file_path in files:
        remove_artifacts(file_path, out_path)
        print('finished', file_path)


def add_mask_synthrad2024(ct_image, mr_image, mask):
    ct = sitk.GetArrayFromImage(ct_image)
    mr = sitk.GetArrayFromImage(mr_image)
    mask = sitk.GetArrayFromImage(mask)
    ct[mask == 0.0] = ct.min()
    mr[mask == 0.0] = mr.min()
    masked_ct = sitk.GetImageFromArray(ct)
    masked_mr = sitk.GetImageFromArray(mr)
    masked_ct.CopyInformation(ct_image)
    masked_mr.CopyInformation(mr_image)

    return masked_ct, masked_mr


def pad_and_rescale(image):
    array = sitk.GetArrayFromImage(image)

    c, a, b = array.shape[0], array.shape[1], array.shape[2]
    max_length = max(a, b)

    if a < max_length:
        pad1 = np.tile(array[:, 0:1, :], (1, (max_length - a) // 2, 1))
        pad2 = np.tile(array[:, -1:, :], (1, (max_length - a) - (max_length - a) // 2, 1))
        padded_array = np.concatenate((pad1, array, pad2), axis=1)
    else:
        padded_array = array

    if b < max_length:
        pad1 = np.tile(padded_array[:, :, 0:1], (1, 1, (max_length - b) // 2))
        pad2 = np.tile(padded_array[:, :, -1:], (1, 1, (max_length - b) - (max_length - b) // 2))
        padded_array = np.concatenate((pad1, padded_array, pad2), axis=2)

    final_image = sitk.GetImageFromArray(padded_array)
    final_image.SetSpacing(image.GetSpacing())
    final_image.SetOrigin(image.GetOrigin())
    final_image.SetDirection(image.GetDirection())

    # rescale
    shape = padded_array.shape
    assert shape[1] == shape[2]

    scale_factor = shape[1] / 256.0

    original_spacing = final_image.GetSpacing()
    original_size = final_image.GetSize()

    new_spacing = [osz * scale_factor for osz in original_spacing]
    new_size = [int(round(osz / scale_factor)) for osz in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(final_image.GetDirection())
    resampler.SetOutputOrigin(final_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_image = resampler.Execute(final_image)

    return resampled_image


def image_process_synthrad2023(in_file, out_path):
    os.makedirs(os.path.join(out_path, os.path.basename(in_file)), exist_ok=True)
    ct_path = os.path.join(in_file, 'ct.nii.gz')
    mr_path = os.path.join(in_file, 'mr.nii.gz')
    # mask_path = os.path.join(in_file, 'mask.nii.gz')
    label_path = os.path.join('/misc/data/private/autoPET/synth2023_label',
                              os.path.basename(in_file) + '_ct_label.nii.gz')
    ct_image = sitk.ReadImage(ct_path)
    mr_image = sitk.ReadImage(mr_path)
    # mask = sitk.ReadImage(mask_path)
    label = sitk.ReadImage(label_path)
    # add mask
    # masked_ct, masked_mr = add_mask_synthrad2024(ct_image, mr_image, mask)
    # pad and rescale
    final_ct = pad_and_rescale(ct_image)
    final_mr = pad_and_rescale(mr_image)
    final_label = pad_and_rescale(label)
    sitk.WriteImage(final_ct, os.path.join(out_path, os.path.basename(in_file), 'ct.nii.gz'))
    # print('finish ', os.path.join(out_path, os.path.basename(in_file), 'ct.nii.gz'))
    sitk.WriteImage(final_mr, os.path.join(out_path, os.path.basename(in_file), 'mr.nii.gz'))
    sitk.WriteImage(final_label, os.path.join(out_path, os.path.basename(in_file), 'label.nii.gz'))
    # print('finish', os.path.join(out_path, os.path.basename(in_file), 'mr.nii.gz'))


def image_process_total_mri(root_path, label_in_path, out_path):
    out_mr = os.path.join(out_path, 'mr')
    out_label = os.path.join(out_path, 'label')
    os.makedirs(out_mr, exist_ok=True)
    os.makedirs(out_label, exist_ok=True)
    names = os.listdir(label_in_path)
    for item in names:
        x = item.split('.')[0]
        mr_path = os.path.join(root_path, x, 'mri.nii.gz')
        label_path = os.path.join(label_in_path, f'{x}.nii.gz')

        try:
            mr_image = sitk.ReadImage(mr_path)
            label = sitk.ReadImage(label_path)
            final_mr = pad_and_rescale(mr_image)
            final_label = pad_and_rescale(label)
            sitk.WriteImage(final_mr, os.path.join(out_mr, f'{item}.nii.gz'))
            sitk.WriteImage(final_label, os.path.join(out_label, f'{item}.nii.gz'))
        except:
            mr_image = sitk.ReadImage(mr_path, sitk.sitkFloat32)
            mr_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

            print('error', x )


def iterator_synthrad2023(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    files = [os.path.join(in_path, f) for f in os.listdir(in_path) if f != 'overview']
    for file_path in files:
        image_process_synthrad2023(file_path, out_path)
        print('finished', file_path)


txt = "/no_backups/d1502/medicaldiffusion/dataset/total_mri.txt"


def read_labels(file_path):
    name = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                file = 's'+str(line).zfill(4)
                name.append(file)
    return name


def iterator_total_mri_combine_label(in_path, output_path):
    name = read_labels(txt)
    os.makedirs(output_path, exist_ok=True)
    files = [os.path.join(in_path, f) for f in name]
    for item in files:
        number = item.split('/')[-1]
        print(number)
        seg_path = os.path.join(item, 'segmentations')
        organs_path = os.listdir(seg_path)
        mr = nib.load(os.path.join(seg_path, organs_path[0]))
        mr_example = mr.get_fdata()
        mr_affine = mr.affine
        merged_data = np.zeros_like(mr_example)
        for organ in organs_path:
            if organ.split('.')[0] in Totalsegmentator_ct_classes or organ.split('_')[0] in Totalsegmentator_ct_classes:
                nii_path = os.path.join(seg_path, organ)
                anatomy = nib.load(nii_path)
                data_anatomy = anatomy.get_fdata()
                try:
                    label = Totalsegmentator_ct_classes[organ.split('.')[0]]
                except:
                    label = Totalsegmentator_ct_classes[organ.split('_')[0]]
                merged_data[data_anatomy != 0] = int(label)
        merged_label = nib.Nifti1Image(merged_data, affine=mr_affine)
        print(output_path)
        nib.save(merged_label, os.path.join(output_path, f'{number}.nii.gz'))
        print('finished', os.path.join(output_path, f'{number}.nii.gz'))


def save_cropped_total_mr(image_in_files, image_out_files, crop_size, length=32):
    image_in = os.path.join(image_in_files, "mr")
    label_in = os.path.join(image_in_files, "label")
    image_in_paths = os.listdir(image_in)
    os.makedirs(os.path.join(image_out_files, 'mr'), exist_ok=True)
    os.makedirs(os.path.join(image_out_files, 'label'), exist_ok=True)
    for image_path in image_in_paths:
        path_image = os.path.join(image_in, image_path)
        path_label = path_image.replace(label_in, image_path)
        img = nib.load(path_image)
        label = nib.load(path_label)
        img_data = img.get_fdata()
        label_data = label.get_fdata()
        assert img_data.shape == label_data.shape, "Error: The shapes of image data and label data do not match."
        n = 0
        for i in range(0, img_data.shape[2] // 8):
            if img_data.shape[2] < 32:
                continue
            number = random.randint(0, img_data.shape[2] - length)
            cropped_image, cropped_label = crop_block(img_data, label_data, *crop_size, number, length)
            if is_all_zero(cropped_image, cropped_label):
                print("Array is all zeros. Skipping rescaling.")
                continue
            cropped_img = nib.Nifti1Image(cropped_image, affine=img.affine)
            name = image_path.split('.')[0]

            img_output_path = os.path.join(image_out_files, 'mr', name + f'_{n}.' + 'nii.gz')
            label_output_path = os.path.join(image_out_files, 'label', name + f'_{n}.' + 'nii.gz')
            nib.save(cropped_img, img_output_path)
            cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
            nib.save(cropped_label, label_output_path)
            print('finished', img_output_path)
            print('finished', label_output_path)
            n+=1


def dicom_serie2nifti(dicom_folder, nifti_save_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image_3d = reader.Execute()
    image_size = image_3d.GetSize()
    #image_3d = rescale(image_3d)
    sitk.WriteImage(image_3d, nifti_save_path)
    return image_size


def resample_image(image, reference_image):
    """
    Resample the image to match the spatial properties of the reference_image.
    """
    # Create resampler
    resampler = sitk.ResampleImageFilter()

    # Set the reference image properties
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())

    # Set the interpolator (linear interpolation)
    resampler.SetInterpolator(sitk.sitkLinear)

    # Execute resampling
    resampled_image = resampler.Execute(image)

    return resampled_image


def combine_label_duke(in_path, out_path, item):

    breast_seg_name = 'Segmentation_' + item + '_Breast.seg.nrrd'
    vessels_seg_name = 'Segmentation_' + item + '_Dense_and_Vessels.seg.nrrd'
    seg_name = item + '.nii.gz'
    breast_seg_path = os.path.join(in_path, item, breast_seg_name)
    vessels_seg_path = os.path.join(in_path, item, vessels_seg_name)
    seg_breast = sitk.ReadImage(breast_seg_path)
    seg_vessels = sitk.ReadImage(vessels_seg_path)
    seg_vessels_array = sitk.GetArrayFromImage(seg_vessels)
    seg_breast_array = sitk.GetArrayFromImage(seg_breast)
    if len(seg_vessels_array.shape) == 4:  # 0： breast， 1： vessel， 3： fibroglandular/dense tissue
        seg_vessels_array = seg_vessels_array[:, :, :, 1]
        seg_dense_array = seg_vessels_array[:, :, :, 0]
        seg_breast_array = seg_breast_array

    elif len(seg_vessels_array.shape) == 3:
        #seg_vessels_array = seg_vessels_array
        seg_dense_array = np.zeros(seg_vessels_array.shape, dtype=np.uint8)
        #seg_breast_array = seg_breast_array


    assert seg_vessels_array.shape == seg_breast_array.shape == seg_dense_array.shape, f"{seg_vessels_array.shape},{seg_breast_array.shape},{seg_dense_array.shape}"

    #combined_label = np.concatenate((seg_breast_array, seg_vessels_array, seg_dense_array), axis=-1)
    combined_label = np.zeros_like(seg_vessels_array)
    combined_label[seg_breast_array == 1] = 1
    combined_label[seg_vessels_array == 1] = 2
    combined_label[seg_dense_array == 1] = 3
    combined_label = combined_label.transpose(1, 2, 0)
    print(combined_label.shape)


    # Create a NIfTI image
    combined_label_nifti = nib.Nifti1Image(combined_label, affine=np.eye(4))

    # Save the NIfTI image
    seg_output_path = os.path.join(out_path, seg_name)
    nib.save(combined_label_nifti, seg_output_path)

    return combined_label.shape[1:]  # Return the shape of the 3D volume (x, y, z)


def stack_mr_combine_labels_duck_breast(input_root, output_root):
    input_mr_root = os.path.join(input_root, 'Duke-Breast-Cancer-MRI')
    input_seg_root = os.path.join("/data/private/autoPET/duke/SEG_raw")
    seg_path_list = os.listdir(input_seg_root)
    output_path_labeled_mr = os.path.join(output_root, 'labeled_MR')
    output_path_unlabeled_mr = os.path.join(output_root, 'unlabeled_MR')
    output_path_seg = os.path.join(output_root, 'SEG')
    os.makedirs(output_path_labeled_mr, exist_ok=True)
    os.makedirs(output_path_unlabeled_mr, exist_ok=True)
    os.makedirs(output_path_seg, exist_ok=True)
    for item in sorted(seg_path_list): # different patients
        i = 0
        patient_mr_path = os.path.join(input_mr_root, item)
        patient_path_list = os.listdir(patient_mr_path)
        try:
            seg_shape = combine_label_duke(input_seg_root, output_path_seg, item)
            print("finished label", item)
        except:
            print("label mistake",  item) # even label has problem, save mr to unlabeled one
        """
            for x in patient_path_list:  # the unuseful middle path
                for mr_dir in os.listdir(os.path.join(patient_mr_path, x)):  # different image of same patient
                    found = False
                    mr_path_ab = os.path.join(patient_mr_path, x, mr_dir)
                    num = len(os.listdir(mr_path_ab))
                    if num < 10:
                        continue
                    else:
                        output_name = item + f'_{i}.nii.gz'
                        output_path = os.path.join(output_path_unlabeled_mr, output_name)
                        mr_size = dicom_serie2nifti(mr_path_ab, output_path)
                        i += 1
            continue
        print(seg_shape)
        length = seg_shape[2]
        for x in patient_path_list:    # the unuseful middle path
            for mr_dir in os.listdir(os.path.join(patient_mr_path, x)):  # different image of same patient
                found = False
                mr_path_ab = os.path.join(patient_mr_path, x, mr_dir)
                num = len(os.listdir(mr_path_ab))
                if num < 10:  # filter the slice label file.
                    continue
                if num == length:
                    output_name = item + '.nii.gz'
                    output_path = os.path.join(output_path_labeled_mr, output_name)
                    mr_size = dicom_serie2nifti(mr_path_ab, output_path)
                    assert seg_shape == mr_size, f"{seg_shape},{mr_size}"
                    found = True
                else:
                    output_name = item + f'_{i}.nii.gz'
                    output_path = os.path.join(output_path_unlabeled_mr, output_name)
                    mr_size = dicom_serie2nifti(mr_path_ab, output_path)
                    i += 1
            break
        print("finished", item)
        if not found:
            print("not find labeled mr for", item)
        """
    print('finished all')


def rescale_crop_duke(root_path, both_label_image=False):

    label_input = os.path.join(root_path, 'SEG')
    labeled_mr_input = os.path.join(root_path, 'labeled_MR')
    unlabeled_mr_input = os.path.join(root_path, 'unlabeled_MR')

    label_output = os.path.join(root_path, 'label')
    labeled_mr_output = os.path.join(root_path, 'labeled_mr_bspline')
    mr_output = os.path.join(root_path, 'mr_bspline')

    os.makedirs(label_output, exist_ok=True)
    os.makedirs(labeled_mr_output, exist_ok=True)
    os.makedirs(mr_output, exist_ok=True)

    labeled_mr_files = [os.path.join(labeled_mr_input, f) for f in os.listdir(labeled_mr_input)]
    labeled_files = [os.path.join(label_input, f) for f in os.listdir(label_input)]
    unlabeled_mr_files = [os.path.join(unlabeled_mr_input, f) for f in os.listdir(unlabeled_mr_input)]
    if both_label_image:
        for label_path in sorted(labeled_files):
            name = label_path.split('/')[-1]
            image_path = label_path.replace('SEG', 'labeled_MR')
            label = sitk.ReadImage(label_path)
            image = sitk.ReadImage(image_path)
            image = rescale(image)
            label = rescale(label, label=True)
            sitk.WriteImage(image, os.path.join(labeled_mr_output, f'scaled_{name}'))
            sitk.WriteImage(label, os.path.join(label_output, f'scaled_{name}'))
            crop_save(name, os.path.join(labeled_mr_output, f'scaled_{name}'), labeled_mr_output, label_path=os.path.join(label_output, f'scaled_{name}'), label_out_files=label_output, crop_size=(256, 256), length=32, labelandimage=True)
            print("finished", name)
    else:
        path = labeled_mr_files + unlabeled_mr_files
        for item in path:  # get all mr blocks from all mr raw data
            name = item.split('/')[-1]
            mr = sitk.ReadImage(item)
            mr = rescale(mr)
            sitk.WriteImage(mr, os.path.join(mr_output, f'scaled_{name}'))
            crop_save(name, os.path.join(mr_output, f'scaled_{name}'), mr_output)
            print("finished", name)

    print('all finished')


def crop_save(name, image_path, image_out_files,  label_path=None, label_out_files=None, crop_size=(256, 256), length=32, labelandimage=False):

    niffti_data = nib.load(image_path)
    image_data = niffti_data.get_fdata()
    n = 0
    for i in range(0, image_data.shape[2] // 4):
        if image_data.shape[2] < 32:
            continue
        number = random.randint(0, image_data.shape[2] - length)
        if labelandimage:
            label_niffti_data = nib.load(label_path)
            label_data = label_niffti_data.get_fdata()
            print("label shape",label_data.shape)
            print("image shape", image_data.shape)
            assert image_data.shape == label_data.shape, f"Error: The shapes of arrayys do not match.{image_data.shape},{label_data.shape},{name}"
            cropped_image, cropped_label = crop_block(image_data, label_data, *crop_size, number, length)
            if is_all_zero(cropped_image, cropped_label):
               print("Array is all zeros. Skipping rescaling.")
               continue
            label_output_path = os.path.join(label_out_files, name + f'_{n}.' + 'nii.gz')
            cropped_label = nib.Nifti1Image(cropped_label, affine=label_niffti_data.affine)
            nib.save(cropped_label, label_output_path)
        else:
            cropped_image = crop_block_single(image_data, *crop_size, number, length)

        image_output_path = os.path.join(image_out_files,  name + f'_{n}.' + 'nii.gz')
        cropped_image = nib.Nifti1Image(cropped_image, affine=niffti_data.affine)
        nib.save(cropped_image, image_output_path)
        n += 1


def rescale(image, label=False):

    # rescale
    shape = image.GetSize()
    print(shape)
    assert shape[0] == shape[1]

    scale_factor = shape[1] / 256.0

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [osz * scale_factor for osz in original_spacing]
    new_size = [int(round(osz / scale_factor)) for osz in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    if label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)   # sitk.sitkBSpline
    resampled_image = resampler.Execute(image)

    return resampled_image


if __name__ == '__main__':

    source_folder1 = '/data/private/autoPET/imagesTr'
    out_folder = '/data/private/autoPET/imagesTr_wo_artifacts'
    train_folder = '/data/private/autoPET/autopet_3d/train'
    test_folder = '/data/private/autoPET/autopet_3d/test'

    source_folder2 = '/data/private/autoPET/imagesTr_wo_artifacts'
    train_folder2 = '/data/private/autoPET/autopet_3d_wo_artifacts/train'
    test_folder2 = '/data/private/autoPET/autopet_3d_wo_artifacts/test'

    source_folder3 = '/misc/data/private/autoPET/Task1/pelvis'
    out_folder2 = '/misc/data/private/autoPET/Processed_SynthRad2024_raw_withoutmask'
    train_folder3 = '/data/private/autoPET/SynthRad2024_withoutmask/train'
    test_folder3 = '/data/private/autoPET/SynthRad2024_withoutmask/test'
    train_folder4 = '/data/private/autoPET/autopet_3d_only_crop/train'
    test_folder4 = '/data/private/autoPET/autopet_3d_only_crop/test'
    Total_label_out = '/data/private/autoPET/Totalsegmentator_mri_cutted/label_'
    Total_mri_root = "/misc/data/private/autoPET/TotalSegmentator"
    Total_out = '/data/private/autoPET/Totalsegmentator_mri_cutted/'
    croped_total_mr = '/data/private/autoPET/Totalsegmentator_mri_croped/'
    duke_input_root = '/data/private/autoPET/duke/'
    duke_output_root = '/data/private/autoPET/duke/'
    autopet = False
    if autopet:
        preprocess_raw = False
        crop = True
        if preprocess_raw:
            iterator(source_folder1, source_folder2)
        if crop:
            process_autopet_onlycrop(source_folder1, train_folder4, test_folder4, crop_size=(256, 256))

    sythrad2023 = False
    if sythrad2023:
        preprocess_raw = False
        cut = True
        if preprocess_raw:
            iterator_synthrad2023(source_folder3, out_folder2)
        if cut:
            process_images_synthrad2023(out_folder2, train_folder3, test_folder3)
    total_mri = False
    if total_mri:
        os.makedirs(Total_label_out, exist_ok=True)
        pad_rescale = False
        cut = False
        combine_label = False
        crop = True
        if combine_label:
            iterator_total_mri_combine_label(Total_mri_root, Total_label_out)
            print('finished label combine')
        if pad_rescale:
            image_process_total_mri(Total_mri_root, Total_label_out, Total_out)
        if crop:
            save_cropped_total_mr(Total_out, croped_total_mr, (256, 256), length=32)

    duke = True
    combine_label_and_dicom2niffti = True
    rescale_crop2blocks = True
    if duke:
        if combine_label_and_dicom2niffti:
            stack_mr_combine_labels_duck_breast(duke_input_root, duke_output_root)
        if rescale_crop2blocks:
            #rescale_crop_duke(duke_output_root)
            rescale_crop_duke(duke_output_root, both_label_image=True)




