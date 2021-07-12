import numpy as np
from scipy.misc import imread, imresize, imsave

#----------------------------------------------------------------------------
# Linear interpolations
def lerp(a, b, t): return a + (b - a) * t
#----------------------------------------------------------------------------
# Loads images, normalizes to (-1, 1) and returns as numpy arrays
def load_image(
        image_path,  # path of a image
        lod=0.0,
        image_size=256,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
):
    if is_gray:
        image = imread(image_path, mode='L').astype(np.float32)
    else:
        image = imread(image_path, mode='RGB').astype(np.float32)
    image = imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    image = np.asarray(image)
    image = image.transpose([2, 0, 1]) # HWC => CHW
    
    # Smooth crossfade between consecutive levels-of-detail.
    s = image.shape
    y = np.reshape(image, [s[0], s[1]//2, 2, s[2]//2, 2])
    y = np.mean(y, axis=(2, 4), keepdims=True)
    y = np.tile(y, [1, 1, 2, 1, 2])
    y = np.reshape(y, [s[0], s[1], s[2]])
    image = lerp(image, y, lod - np.floor(lod))
    
    # Upscale to match the expected input/output size of the networks.
    s = image.shape
    factor = int(2 ** np.floor(lod))
    image = np.reshape(image, [s[0], s[1], 1, s[2], 1])
    image = np.tile(image, [1, 1, factor, 1, factor])
    image = np.reshape(image, [s[0], s[1] * factor, s[2] * factor])
    
    return image

#----------------------------------------------------------------------------
# recives a numpy array, converts from (-1, 1) to (0, 255), and saves as image
def save_batch_images(
        batch_images,   # a batch of images
        save_path,  # path to save the images
        image_value_range=(-1,1),   # value range of the input batch images
        size_frame=None     # size of the image matrix, number of images in each row and column
):
    # transform the pixcel value to 0~1
    batch_images = batch_images.transpose([0, 2, 3, 1]) 
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    imsave(save_path, frame)
#----------------------------------------------------------------------------
    