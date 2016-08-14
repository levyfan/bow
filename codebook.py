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
    feature = sio.loadmat('TUDpositive_parse_500_20.0.mat')

    # hsv = transpose(feature['hsv'])
    # cn = transpose(feature['cn'])
    # hog = transpose(feature['hog'])
    # siltp = transpose(feature['siltp'])
    # all = hstack((hsv, cn, hog, siltp))

    p0 = transpose(feature['p0'])
    p1 = transpose(feature['p1'])
    p2 = transpose(feature['p2'])
    p3 = transpose(feature['p3'])
    p4 = transpose(feature['p4'])

    # print 'hsv'
    # codebook_HSV = generate_codebook(hsv, 1024)

    print 'p0'
    codebook_p0 = generate_codebook(p0, 1024)

    # print 'cn'
    # codebook_CN = generate_codebook(cn, 1024)

    print 'p1'
    codebook_p1 = generate_codebook(p1, 1024)

    # print 'hog'
    # codebook_HOG = generate_codebook(hog, 1024)

    print 'p2'
    codebook_p2 = generate_codebook(p2, 1024)

    # print 'siltp'
    # codebook_SILTP = generate_codebook(siltp, 1024)

    print 'p3'
    codebook_p3 = generate_codebook(p3, 1024)

    # print 'all'
    # codebook_ALL = generate_codebook(all, 1024*4)

    print 'p4'
    codebook_p4 = generate_codebook(p4, 1024)

    # mdict = dict(
    #     codebook_HSV=codebook_HSV,
    #     codebook_CN=codebook_CN,
    #     codebook_HOG=codebook_HOG,
    #     codebook_SILTP=codebook_SILTP,
    #     codebook_ALL=codebook_ALL
    # )

    mdict = dict(
        codebook_p0=codebook_p0,
        codebook_p1=codebook_p1,
        codebook_p2=codebook_p2,
        codebook_p3=codebook_p3,
        codebook_p4=codebook_p4,
    )

    sio.savemat('codebook.mat', mdict)
