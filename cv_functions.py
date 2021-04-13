import numpy as np
import cv2
import skimage.filters as filters
from skimage import img_as_ubyte
from PIL import Image

# Find all the images inside the folder (only the name)
def loadImg(path):
        plate_image = cv2.imread(path)
        license_plate, alpha, beta = automatic_brightness_and_contrast(plate_image)
        if len(license_plate.shape) == 3:
            license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        adaptive = apply_adaptive_threshold(license_plate)
        binary = cv2.threshold(adaptive, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel4 = getKernelBasedOnMedianValue(np.median(license_plate.ravel()))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel4)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel4)
        thre_mor = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel4)
        thre_mor = erode(thre_mor, kern_size=1)
        if "chinese" in path:
            img = cv2.cvtColor(thre_mor, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(cv2.bitwise_not(thre_mor), cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        tf_image = np.array(im_pil)
        return tf_image


def automatic_brightness_and_contrast(image, clip_hist_percent=5):
    if len(image.shape)==3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0], None, [256], [0,256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def getKernelBasedOnMedianValue(plate_image_median):
    if plate_image_median > 170:
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    else:
        kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return kernel4

def preprocess_image(plate_image):
    plate_image = cv2.imread(plate_image)
    license_plate, alpha, beta = automatic_brightness_and_contrast(plate_image)
    if len(license_plate.shape) == 3:
        # auto_result, alpha, beta = automatic_brightness_and_contrast(license_plate)
        license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    adaptive = apply_adaptive_threshold(license_plate)
    binary = cv2.threshold(adaptive, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel4 = getKernelBasedOnMedianValue(np.median(license_plate.ravel()))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel4)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel4)
    thre_mor = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel4)
    thre_mor = erode(thre_mor, kern_size=1)
    img = cv2.cvtColor(cv2.bitwise_not(thre_mor), cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil
    

def erode(img, kern_size=2):
    kern = np.ones((kern_size, kern_size), np.uint8) # make a kernel for erosion based on given kernel size.
    eroded = cv2.erode(img, kern, 1) # erode your image to blobbify black areas
    y, x = eroded.shape # get shape of image to make a white boarder around image of 1px, to avoid problems with find contours.
    return cv2.rectangle(eroded, (0, 0), (x, y), (0, 0, 0), 1)

def apply_adaptive_threshold(license_plate):
    channel_0 = license_plate[:, 0]
    channel_1 = license_plate[:, 1]
    channel_0_mean = np.mean(channel_0)
    channel_1_mean = np.mean(channel_1)

    if channel_0_mean <= 30 and channel_1_mean <= 30:
        threshold = filters.threshold_local(license_plate, block_size=81, offset=5)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    elif channel_0_mean > 30 and channel_0_mean <= 60 and channel_1_mean > 30 and channel_1_mean <= 60:
        threshold = filters.threshold_local(license_plate, block_size=71, offset=13)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    elif channel_0_mean > 60 and channel_0_mean <= 90 and channel_1_mean > 60 and channel_1_mean <= 90:
        threshold = filters.threshold_local(license_plate, block_size=125, offset=9)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    elif channel_0_mean > 90 and channel_0_mean <= 100 and channel_1_mean > 90 and channel_1_mean <= 100:
        threshold = filters.threshold_local(license_plate, block_size=125, offset=1)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    elif channel_0_mean > 100 and channel_1_mean > 100 and channel_0_mean < 130 and channel_1_mean < 130:
        threshold = filters.threshold_local(license_plate, block_size=115, offset=17)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    elif channel_0_mean >= 130 and channel_1_mean >= 130 and channel_0_mean < 160 and channel_1_mean < 160:
        threshold = filters.threshold_local(license_plate, block_size=41, offset=7)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    else:
        threshold = filters.threshold_local(license_plate, block_size=131, offset=22)
        image = license_plate > threshold
        binary = img_as_ubyte(image)
    return binary