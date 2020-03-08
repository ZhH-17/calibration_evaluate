import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from images import Images
from camera import Camera
from utility import triangulation_two, read_fs, plot_tripoints

def gamma_ss(s, s_, f):
    # return c_s_/c_s
    return (s-f)/(s_-f)*s_/s

def compute_brown(s, s1, K1, dist1, s2, K2, dist2):
    # compute params at s3 based on s1 & s2
    f = 6.e-3
    alpha = (s2-s)/(s2-s1)*(s1-f)/(s-f)
    dist = alpha*dist1 + (1-alpha)*dist2

    if True:
        ga_s1_s = gamma_ss(s1, s, f)
        K = K1.copy()
        K[0, 0] = ga_s1_s * K[0, 0]
        K[1, 1] = ga_s1_s * K[1, 1]
    else:
        ga_s2_s = gamma_ss(s2, s, f)
        K = K2.copy()
        K[0, 0] = ga_s2_s * K[0, 0]
        K[1, 1] = ga_s2_s * K[1, 1]

    return K, dist



if __name__ == "__main__":
    f = 6.e-3
    w, h = 1920, 1080
    params_33cm = read_fs("./datasets/data/image_cap_mono_33cm/camera.yaml", "matrix", "dist")
    s1 = 0.33
    K_33 = params_33cm[0]
    dist_33 = params_33cm[1]
    # camera_33 = Camera((w, h), K_33, dist_33)
    # TV_u, TV_v, _, _ = camera_33.TV_distortion()
    # print("TV distortion: ", TV_u, TV_v)

    params_3m = read_fs("./datasets/data/image_cap_mono_3m/camera0.yml", "matrix", "dist")
    s2 = 3.
    K_3 = params_3m[0]
    dist_3 = params_3m[1]
    # camera_3 = Camera((w, h), K_3, dist_3)
    # TV_u, TV_v, _, _ = camera_3.TV_distortion()
    # print("TV distortion: ", TV_u, TV_v)

    params_1m = read_fs("./datasets/data/image_cap_mono_1m/camera.yaml", "matrix", "dist")
    s3 = 1.
    K_1 = params_1m[0]
    dist_1 = params_1m[1]
    # camera_1 = Camera((w, h), K_1, dist_1)
    # TV_u, TV_v, _, _ = camera_1.TV_distortion()
    # print("TV distortion: ", TV_u, TV_v)

    # plt.figure()
    # plt.plot([0.33, 1, 3], [(K_33[0, 0]+K_33[1,1])/2.,
    #     (K_1[0, 0]+K_1[1, 1])/2., (K_3[0, 0]+K_3[1, 1])/2.], '-o', label="f")
    # plt.legend()
    # plt.savefig("../../1.png")
    # plt.figure()
    # plt.plot([0.33, 1, 3], [K_33[0, 2], K_1[0, 2], K_3[0, 2]], '-o', label="cx")
    # plt.legend()
    # plt.savefig("../../2.png")
    # plt.figure()
    # plt.plot([0.33, 1, 3], [K_33[1, 2], K_1[1, 2], K_3[1, 2]], '-o', label="cy")
    # plt.legend()
    # plt.savefig("../../3.png")
    # # plt.show()

    # g1 = gamma_ss(0.33, 1, f)
    # g2 = gamma_ss(0.33, 3, f)
    # g3 = gamma_ss(1., 3., f)
    # print(g1, g2, g3)

    def camera_trianular(fns, K, dist, pattern_size):
        images_test = Images(fns)
        images_test.pattern_size = pattern_size
        obj_pnts, img_pnts = images_test.find_chess_img_pnts()
        tripoints3d = triangulation_two(img_pnts, K)
        plot_tripoints(tripoints3d)
        plt.savefig("../tri.png")

        camera_test = Camera((w, h), K, dist)
        img_pnts_undistort = camera_test.undistort_points(img_pnts)
        tripoints3d = triangulation_two(img_pnts_undistort, K)
        plot_tripoints(tripoints3d)
        plt.savefig("../tri_undistort.png")

        plt.show()
        


    # fns_test = ["./datasets/data/image_cap_mono_3m_test/1.png", "./datasets/data/image_cap_mono_3m_test/2.png"]
    # camera_trianular(fns_test, K_3, dist_3, (7, 11))
    # K_3_c, dist_3_c = compute_brown(s2, s1, K_33, dist_33, s3, K_1, dist_1)
    # camera_trianular(fns_test, K_3_c, dist_3_c, (7, 11))

    # fns_test = ["./datasets/data/image_cap_mono_1m_test/1582705634.57.png", 
    #     "./datasets/data/image_cap_mono_1m_test/1582706309.93.png"]
    fns_test = ["./datasets/data/image_cap_mono_1m_test/10.png", 
        "./datasets/data/image_cap_mono_1m_test/11.png"]
    # camera_trianular(fns_test, K_1, dist_1, (7, 11))
    K_1_c, dist_1_c = compute_brown(s3, s1, K_33, dist_33, s2, K_3, dist_3)
    camera_trianular(fns_test, K_1_c, dist_1_c, (7, 11))

    # fns_test = ["./datasets/data/image_cap_mono_33cm_test/3.png", 
    #     "./datasets/data/image_cap_mono_33cm_test/5.png"]
    # camera_trianular(fns_test, K_33, dist_33, (6, 9))
    # K_33_c, dist_33_c = compute_brown(s1, s2, K_3, dist_3, s3, K_1, dist_1)
    # camera_trianular(fns_test, K_33_c, dist_33_c, (6, 9))

    # fns_test = ["./datasets/data/image_cap_mono_33cm_test/10.png", 
    #     "./datasets/data/image_cap_mono_33cm_test/11.png"]
    # images_test = Images(fns_test)
    # img1 = cv.imread(fns_test[0], -1)
    # img1 = img1[:, 500:-500]
    # img2 = cv.imread(fns_test[1], -1)
    # img2 = img2[:, 500:-500]

    # img1_pnts, img2_pnts, num1, num2 = images_test.find_img_pnts_manual(2, img1, img2)
    # tripoints3d = triangulation_two([img1_pnts, img2_pnts], K_33, cv.RANSAC, True)
    # plot_tripoints(tripoints3d, num1, num2)
    


    # images_test = Images(fns_test)
    # img1_pnts, img2_pnts, num1, num2 = images_test.find_img_pnts_manual(1)
    # tripoints3d = triangulation_two([img1_pnts, img2_pnts], K_1, cv.RANSAC, True)
    # plot_tripoints(tripoints3d, num1, num2)

    # img_pnts_undistort = camera_1.undistort_points([img1_pnts, img2_pnts])
    # tripoints3d = triangulation_two([img1_pnts, img2_pnts], K_1, cv.RANSAC, True)
    # plot_tripoints(tripoints3d, num1, num2)

    plt.show()

    # K_1_c, dist_1_c = compute_brown(s, s1, K_33, dist_33, s2, K_3, dist_3)
    # tripoints3d = triangulation_two([img1_pnts, img2_pnts], K, cv.RANSAC, True)
    # plot_tripoints(tripoints3d, num1, num2)
    # plt.show()