import cv2
import scipy.io as sio
from numpy import sqrt, float32, hstack, transpose


def generate_codebook(feature, codebook_size):
    retval, bestLabels, centers = cv2.kmeans(
        data=float32(feature),
        K=codebook_size,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER, 100, 0),
        attempts=1,
        flags=cv2.KMEANS_PP_CENTERS)
    return centers


if __name__ == "__main__":
    feature = sio.loadmat('TUDpositive_feature_500_1.0.mat')

    hsv = transpose(feature['hsv'])
    cn = transpose(feature['cn'])
    hog = transpose(feature['hog'])
    siltp = transpose(feature['siltp'])
    all = hstack((hsv, cn, hog, siltp))

    print 'hsv'
    codebook_HSV = generate_codebook(hsv, 350)

    print 'cn'
    codebook_CN = generate_codebook(cn, 350)

    print 'hog'
    codebook_HOG = generate_codebook(hog, 350)

    print 'siltp'
    codebook_SILTP = generate_codebook(siltp, 350)

    print 'all'
    codebook_ALL = generate_codebook(all, 350*4)

    mdict = dict(
        codebook_HSV=codebook_HSV,
        codebook_CN=codebook_CN,
        codebook_HOG=codebook_HOG,
        codebook_SILTP=codebook_SILTP,
        codebook_ALL=codebook_ALL
    )
    sio.savemat('codebook.mat', mdict)
