import cv2
import scipy.io as sio
from numpy import sqrt, float32, hstack


def generate_codebook(feature, codebook_size):
    retval, bestLabels, centers = cv2.kmeans(
        data=float32(feature),
        K=codebook_size,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER, 100, 0),
        attempts=1,
        flags=cv2.KMEANS_PP_CENTERS)
    return centers


if __name__ == "__main__":
    sp_feature = sio.loadmat('sp_feature_sqrt_patch.mat')

    hsv = sp_feature['hsv']
    cn = sp_feature['cn']
    hog = sp_feature['hog']
    siltp = sp_feature['siltp']
    all = hstack(hsv, cn, hog, siltp)

    codebook_HSV = generate_codebook(hsv, 350)
    codebook_CN = generate_codebook(cn, 350)
    codebook_HOG = generate_codebook(hog, 350)
    codebook_SILTP = generate_codebook(siltp, 350)
    codebook_ALL = generate_codebook(all, 350*4)

    mdict = dict(
        codebook_HSV=codebook_HSV,
        codebook_CN=codebook_CN,
        codebook_HOG=codebook_HOG,
        codebook_SILTP=codebook_SILTP,
        codebook_ALL=codebook_ALL
    )
    sio.savemat('codebook.mat', mdict)
