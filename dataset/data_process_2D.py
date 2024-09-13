import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image


def get_2d_images(ct_path, ct_label_path, file="train"):

    image_name ="tr" if file=="train" else "val"
    for i in range(len(ct_path)):
        k = 0
        nifti_ct = nib.load(ct_path[i])
        ct_3d = nifti_ct.get_fdata()
        nifti_ct_label = nib.load(ct_label_path[i])
        ct_label_3d = nifti_ct_label.get_fdata()
        n = ct_3d.shape[2]-ct_3d.shape[2] % 32
        for z in range(5, n - 5):
            ct_slice = ct_3d[:, :, z]
            ct_label_slice = ct_label_3d[:, :, z].astype(np.int32)
            #print(ct_label_slice)
            #if ct_label_slice.max() != ct_label_slice.min() and ct_slice.max() != ct_slice.min():
            ct_image = (((ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())) * 255).astype(np.uint8)

            ct_image = Image.fromarray(ct_image)
            ct_label = Image.fromarray(ct_label_slice)
            if file == "train":
                ct_image.save(f'/data/private/autoPET/autopet_2d/image/train/tr_{k}.png')
                ct_label.save(f'/data/private/autoPET/autopet_2d/label/train/tr_{k}.png')
            else:
                ct_image.save(f'/data/private/autoPET/autopet_2d/image/test/ts_{i}_{k}.png')
                ct_label.save(f'/data/private/autoPET/autopet_2d/mask/all/test/ts_{i}_{k}.png')
                #ct_image.save(f'/data/private/autoPET/autopet_2d/image/val/val_{k}.png')
                #ct_label.save(f'/data/private/autoPET/autopet_2d/label/val/val_{k}.png')

                k += 1
        print("finished", ct_path[i])

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


def png_to_3D_npy(input1, input2, output1, output2):

    for i in range(1000):
        name = f"condon_ts_{i}"
        path_list =[k for k in os.listdir(input1) if k.startswith(name)]
        length = len(path_list)-len(path_list) % 32
        images_real = []
        images_fake = []
        for f in range(length):
            full_name_fake = f"condon_ts_{i}_{f}.png"
            full_name_real = f"ts_{i}_{f}.png"
            img_path_real = os.path.join(input1, full_name_real)
            img_path_fake = os.path.join(input2, full_name_fake)
            if os.path.exists(img_path_real) and os.path.exists(img_path_fake) :
                img_real = Image.open(img_path_real)
                img_real = np.array(img_real)
                img_real = img_real / 255.0
                img_fake = Image.open(img_path_fake)
                img_fake = np.array(img_fake)
                img_fake = img_fake / 255.0
                if img_real.ndim == 2:
                    img_real = np.expand_dims(img_real, axis=0)
                if img_fake.ndim == 2:
                    img_fake = np.expand_dims(img_fake, axis=0)
                # 将处理后的图片添加到列表
                images_real.append(img_real)
                images_fake.append(img_fake)
                if len(images_real) == 32 and len(images_fake) == 32:
                    # 拼接图片为一个四维数组 (batch_size, height, width, channels)
                    images_batch_real = np.stack(images_real, axis=0)
                    images_batch_fake = np.stack(images_fake, axis=0)

                    npy_filename_real = os.path.join(output1, f'image_real_{i}.npy')
                    np.save(npy_filename_real, images_batch_real)
                    print(f'Saved {npy_filename_real}')
                    npy_filename_fake = os.path.join(output2, f'image_fake_{i}.npy')
                    np.save(npy_filename_fake, images_batch_fake)
                    print(f'Saved {npy_filename_fake}')
                    images_real = []
                    images_fake = []


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/autopet_2d/image/train/', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/image/test/', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/image/val/', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/label/train/', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/mask/all/test/', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/autopet_2d/label/val/', exist_ok=True)

    path_image_train = "/data/private/autoPET/autopet_3d_only_crop/train"
    path_image_test = "/data/private/autoPET/autopet_3d_only_crop/test"

    #ct_image_train, ct_label_train = list_images(path_image_train)
    #ct_image_test, ct_label_test = list_images(path_image_test)

    #get_2d_images(ct_image_test, ct_label_test, file="test")
    #get_2d_images(ct_image_train, ct_label_train, file="train")
    input1 = "/data/private/autoPET/autopet_2d/image/npy"
    input2 = "/data/private/autoPET/ddim-AutoPET-256-segguided/samples_many_32000"
    output1 = "/data/private/autoPET/ddim-AutoPET-256-segguided/real_npy"
    output2 = "/data/private/autoPET/ddim-AutoPET-256-segguided/fake_npy"
    os.makedirs(output1, exist_ok=True)
    os.makedirs(output2, exist_ok=True)
    png_to_3D_npy(input1, input2, output1, output2)
