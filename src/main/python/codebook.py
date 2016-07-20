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


def run_and_write(feature, codebook_size, codebook_name):
    print codebook_name
    codebook = generate_codebook(feature, codebook_size)
    sio.savemat('sp_' + codebook_name + '.mat', {codebook_name: codebook})


if __name__ == "__main__":
    sp_feature = sio.loadmat('sp_feature_sqrt_patch.mat')

    mixFeature = hstack(sp_feature['hsvFeature'], sp_feature['cnFeature'], sp_feature['hogFeature'], sp_feature['siltpFeature'])

    run_and_write(sp_feature['hsvFeature'], 350, 'codebook_hsv')
    run_and_write(sp_feature['cnFeature'], 350, 'codebook_cn')
    run_and_write(sp_feature['hogFeature'], 350, 'codebook_hog')
    run_and_write(sp_feature['siltpFeature'], 350, 'codebook_siltp')
    run_and_write(mixFeature, 1400, 'codebook_mix_0')

    # mdict = dict(
    #     codebook_HSV=codebook_HSV,
    #     codebook_CN=codebook_CN,
    #     codebook_HOG=codebook_HOG,
    #     codebook_SILTP=codebook_SILTP
    # )
    # sio.savemat('sp_codebook.mat', mdict)
