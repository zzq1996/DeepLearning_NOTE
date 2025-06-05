import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def visualize(original, augmented):
    """
    打印显示图像
    :param original:
    :param augmented:
    :return:
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original image")
    # plt.axis("off")  # 关闭坐标轴显示
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title("Augmented image")
    # plt.axis("off")  # 关闭坐标轴显示
    plt.imshow(augmented)

    plt.tight_layout()
    plt.show()



def Image_Enhancement():
    dirs = os.listdir('./input/corn_leaf/valid')
    print(dirs)
    for classes in dirs:
        classes_pathname = os.path.join('./input/corn_leaf/valid',classes)
        for filepath, dirnames, filenames in os.walk(classes_pathname):
            for filename in filenames:
                filename_path = os.path.join(filepath, filename)
                """
                # 随机图像增强，不可控
                datagen = ImageDataGenerator(
                    rescale=1. / 255,  # 图片中的每个像素点都乘以1/255
                    rotation_range=40,  # 把图形随机旋转一个角度，在0-40度之间
                    width_shift_range=0.2,  # 位移，0-20%之间选择做偏移
                    height_shift_range=0.2,  # 垂直方向位移，如果是0-1之间的数，就是比例，大于1就是像素
                    shear_range=0.2,  # 剪切强度（逆时针剪切角，以度为单位）
                    zoom_range=0.2,  # 缩放强度
                    horizontal_flip=True,  # 水平随机翻转
                    fill_mode='nearest',  # 图形放大后，有些地方需要填充，
                    #  可以是“常数”，“最近”，“反射”或“环绕”之一。默认值为“最近”。输入边界之外的点将根据给定模式进行填充
                    )
                l = []
                img = Image.open(filename_path)  # this is a PIL image
                img_array = np.array(img, dtype=np.float32)
                l.append(img_array)
                l = np.array(l)
                i = 0
                for batch in datagen.flow(l, batch_size=1,
                                          save_to_dir=filepath, save_format='jpg'):
                    i += 1
                    if i > 6:
                        break  # 用i来控制得到的图片张数
                """


                # 根据路径读取图片
                img = tf.io.read_file(filename_path)
                # 解码图片
                img = tf.image.decode_jpeg(img, channels=3)

                flipped_h = tf.image.flip_left_right(img)  # 水平翻转
                flipped_v = tf.image.flip_up_down(img)  # 垂直翻转

                rotated_90 = tf.image.rot90(img)  # 旋转90度
                rotated_180 = tf.image.rot90(rotated_90)
                rotated_270 = tf.image.rot90(rotated_180)

                img_array1 = np.array(flipped_v, dtype=np.float32)
                img_array1 = np.expand_dims(img_array1, axis=0)
                img_array2 = np.array(flipped_h, dtype=np.float32)
                img_array2 = np.expand_dims(img_array2, axis=0)
                # img_array3 = np.array(rotated_90, dtype=np.float32)
                # img_array3 = np.expand_dims(img_array3, axis=0)
                # img_array4 = np.array(rotated_180, dtype=np.float32)
                # img_array4 = np.expand_dims(img_array4, axis=0)
                # img_array5 = np.array(rotated_270, dtype=np.float32)
                # img_array5 = np.expand_dims(img_array5, axis=0)


                i = 0
                datagen = ImageDataGenerator()
                for batch in datagen.flow(img_array1, batch_size=1,
                                          save_to_dir=filepath, save_format='jpg'):
                    i += 1
                    if i > 0:
                        break  # otherwise the generator would loop indefinitely
                i = 0
                for batch in datagen.flow(img_array2, batch_size=1,
                                          save_to_dir=filepath, save_format='jpg'):
                    i += 1
                    if i > 0:
                        break  # otherwise the generator would loop indefinitely
                # i = 0
                # for batch in datagen.flow(img_array3, batch_size=1,
                #                           save_to_dir=filepath, save_format='jpg'):
                #     i += 1
                #     if i > 0:
                #         break  # otherwise the generator would loop indefinitely
                # i = 0
                # for batch in datagen.flow(img_array4, batch_size=1,
                #                           save_to_dir=filepath, save_format='jpg'):
                #     i += 1
                #     if i > 0:
                #         break  # otherwise the generator would loop indefinitely
                # i = 0
                # for batch in datagen.flow(img_array5, batch_size=1,
                #                           save_to_dir=filepath, save_format='jpg'):
                #     i += 1
                #     if i > 0:
                #         break  # otherwise the generator would loop indefinitely


Image_Enhancement()


