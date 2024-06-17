import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np


def crop_center(img, new_x, new_y):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, :]


def crop_block(img, label, new_x, new_y, start_z, length):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length], \
        label[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length]


def is_all_zero(array1, array2):

    return np.all(array1 == 0) or np.all(array2 == 0)


def save_cropped_autopet(image_in_files, image_out_files, crop_size, crop_2_block=False, length=32, stride=16):
    for image_path in image_in_files:
        label_path = image_path.replace('0001.nii.gz', '0002.nii.gz')
        label_path = label_path.replace('imagesTr_wo_artifacts', 'imagesTr')  # only for data remove artifacts
        img = nib.load(image_path)
        label = nib.load(label_path)
        img_data = img.get_fdata()
        label_data = label.get_fdata()
        assert img_data.shape == label_data.shape, "Error: The shapes of image data and label data do not match."
        if crop_2_block:
            for i in range(0, img_data.shape[2]-length, stride):
                cropped_image, cropped_label = crop_block(img_data, label_data,  *crop_size, i, length)
                if is_all_zero(cropped_image, cropped_label):
                    print("Array is all zeros. Skipping rescaling.")
                    continue
                cropped_img = nib.Nifti1Image(cropped_image, affine=img.affine)
                name = os.path.basename(image_path).split('.')[0]
                name = name.split('_')[0]
                img_output_path = os.path.join(image_out_files, name + f'_{i//length}.' + 'nii.gz')
                label_output_path = img_output_path.replace('ct', 'label')
                nib.save(cropped_img, img_output_path)
                cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
                nib.save(cropped_label, label_output_path)
                print('finished', img_output_path)
                print('finished', label_output_path)


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
    return array1[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length], \
        array2[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length], \
        array3[start_x:start_x + new_x, start_y:start_y + new_y, start_z:start_z + length]


def save_cropped_synthrad2023(ct_in_files, image_out_files, crop_size, crop_2_block=False, length=32, stride=16):

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
        if crop_2_block:
            for i in range(0, ct_data.shape[2]-length, stride):
                cropped_ct, cropped_mr, cropped_label = crop_block_3(ct_data, mr_data, label_data,  *crop_size, i, length)
                if is_all_zero(cropped_mr, cropped_label):
                    print("Array is all zeros. Skipping rescaling.")
                    continue
                ct_output_path = os.path.join(image_out_files, 'ct', name + f'_{i//length}.' + 'nii.gz')
                mr_output_path = os.path.join(image_out_files, 'mr', name + f'_{i // length}.' + 'nii.gz')
                label_output_path = os.path.join(image_out_files, 'label', name + f'_{i // length}.' + 'nii.gz')
                cropped_ct = nib.Nifti1Image(cropped_ct, affine=ct.affine)
                cropped_mr = nib.Nifti1Image(cropped_mr, affine=mr.affine)
                cropped_label = nib.Nifti1Image(cropped_label, affine=label.affine)
                nib.save(cropped_ct, ct_output_path)
                nib.save(cropped_mr, mr_output_path)
                nib.save(cropped_label, label_output_path)
                print('finished', ct_output_path)
                print('finished', mr_output_path)
                print('finished', mr_output_path)


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

    ct_files = [os.path.join(source_folder, f, 'ct.nii.gz') for f in os.listdir(source_folder)]

    ct_train_files, ct_test_files = train_test_split(ct_files)

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


def add_mask_synthrad2024(ct_image, mr_image,  mask):

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
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(final_image)

    return resampled_image


def image_process_synthrad2023(in_file, out_path):

    os.makedirs(os.path.join(out_path, os.path.basename(in_file)), exist_ok=True)
    ct_path = os.path.join(in_file, 'ct.nii.gz')
    mr_path = os.path.join(in_file, 'mr.nii.gz')
    mask_path = os.path.join(in_file, 'mask.nii.gz')
    label_path = os.path.join('/misc/data/private/autoPET/synth2023_label', os.path.basename(in_file) + '_ct_label.nii.gz')
    ct_image = sitk.ReadImage(ct_path)
    mr_image = sitk.ReadImage(mr_path)
    mask = sitk.ReadImage(mask_path)
    label = sitk.ReadImage(label_path)
    # add mask
    masked_ct, masked_mr = add_mask_synthrad2024(ct_image, mr_image, mask)
    # pad and rescale
    final_ct = pad_and_rescale(masked_ct)
    final_mr = pad_and_rescale(masked_mr)
    final_label = pad_and_rescale(label)
    sitk.WriteImage(final_ct, os.path.join(out_path, os.path.basename(in_file), 'ct.nii.gz'))
    # print('finish ', os.path.join(out_path, os.path.basename(in_file), 'ct.nii.gz'))
    sitk.WriteImage(final_mr, os.path.join(out_path, os.path.basename(in_file), 'mr.nii.gz'))
    sitk.WriteImage(final_label, os.path.join(out_path, os.path.basename(in_file), 'label.nii.gz'))
    # print('finish', os.path.join(out_path, os.path.basename(in_file), 'mr.nii.gz'))


def iterator_synthrad2023(in_path, out_path):

    os.makedirs(out_path, exist_ok=True)
    files = [os.path.join(in_path, f) for f in os.listdir(in_path) if f !='overview']
    for file_path in files:
        image_process_synthrad2023(file_path, out_path)
        print('finished', file_path)


if __name__ == '__main__':

    source_folder1 = '/data/private/autoPET/imagesTr'
    out_folder = '/data/private/autoPET/imagesTr_wo_artifacts'
    train_folder = '/data/private/autoPET/autopet_3d/train'
    test_folder = '/data/private/autoPET/autopet_3d/test'

    source_folder2 = '/data/private/autoPET/imagesTr_wo_artifacts'
    train_folder2 = '/data/private/autoPET/autopet_3d_wo_artifacts/train'
    test_folder2 = '/data/private/autoPET/autopet_3d_wo_artifacts/test'

    source_folder3 = '/misc/data/private/autoPET/Task1/pelvis'
    out_folder2 = '/misc/data/private/autoPET/Processed_SynthRad2024'
    train_folder3 = '/data/private/autoPET/SynthRad2024/train'
    test_folder3 = '/data/private/autoPET/SynthRad2024/test'
    #iterator(source_folder1, source_folder2)
    #process_images_autopet(source_folder2, train_folder2, test_folder2, crop_size=(256, 256))
    iterator_synthrad2023(source_folder3, out_folder2)
    process_images_synthrad2023(out_folder2, train_folder3, test_folder3)




