from image_segmentation import ImageSegmenter
from PIL import Image

def main():
    segmenter = ImageSegmenter(15, 10)
    im = Image.open("test.png")
    segmented = segmenter.segment(get_pixels(im))
    
def get_pixels(im):
    return [[im.load()[x, y] for y in range(im.size[1])]for x in range(im.size[0])]

if __name__ == "__main__":
    main()
