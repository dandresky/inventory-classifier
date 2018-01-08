'''
This file provides fuunctions to facilitate resizing of images to a consistent
size needed for the inputs of the neural network.
'''
from PIL import Image
from resizeimage import resizeimage
from explore_dataset import get_image_file_names
import sys

IMAGE_DATA_PATH = '../data/bin-images/'

def resize_images(size=(120,120)):
    '''
    Iterate through all image files in the data folder and resize
    '''
    pass

def resize_image(path, filename, size=(120,120)):
    '''
    resize an image so that it can fit in the specified size, keeping the ratio
    and without cropping.
    '''
    width, height = size
    with open(path + filename, 'r+b') as f:
        with Image.open(f) as image:
            img = resizeimage.resize_contain(image, [int(width), int(height)])
            img = img.convert("RGB")
            img.save(filename, img.format)
    # fd_img = open(path + filename, 'r')
    # img = Image.open(fd_img)
    # img = resizeimage.resize_contain(img, [width, height])
    # img.save('test-image-contain.jpg', img.format)
    # fd_img.close()
    pass

def main():
    # test for first command line argument
    if len(sys.argv) < 2:
        print("Provide arguments as follows: \n")
        print("python resize-image.py <file name> [width] [height]")
        return
    if sys.argv[1] == 'all':
        resize_images(IMAGE_DATA_PATH, (sys.argv[2], sys.argv[3]))
    else:
        resize_image(IMAGE_DATA_PATH, sys.argv[1], (sys.argv[2], sys.argv[3]))
    pass


if __name__ == '__main__':
    main()
