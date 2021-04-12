import cv2

from pathlib import Path


# data_dir = Path("./images/")
# # Get list of all the images
# images = list(map(str, list(data_dir.glob("*.jpg"))))

# for image in images:
#     tokens = image.split("/")
#     filename = tokens[len(tokens)-1]
#     img = cv2.imread(image)
#     print(img.shape)


data_dir = Path("/Users/dodo/Downloads/extracted_lp_CCPD2019/")
# Get list of all the images
images = list(map(str, list(data_dir.glob("*.jpg"))))

for image in images:
    tokens = image.split("/")
    filename = tokens[len(tokens)-1]
    img = cv2.imread(image)
    # dsize
    dsize = (470, 110)
    # resize image
    img = cv2.resize(img, dsize)
    name = filename.split(".jpg")[0].upper() 
    cv2.imwrite("./chinese_lp/{}.jpg".format(name), img)