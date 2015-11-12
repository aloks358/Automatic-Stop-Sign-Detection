import image_segmenter

def main():
    segmenter = ImageSegmenter(15, 10)
    segmented = segmenter.segment("")
    

if __name__ == "__main__":
    main()
