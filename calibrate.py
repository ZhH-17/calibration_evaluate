import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from camera import Camera
from images import Images
from utility import reconstruct_pose_cv, triangulation_two
from utility import plot_tripoints
from utility import write_fs, read_fs

if __name__ == "__main__":
    # datapath = "./datasets/oppo"
    # datapath = "./datasets/data/image_cap_mono_1m"
    # pattern_size = (11, 7)

    datapath = "./datasets/data/image_cap_mono_33cm"
    pattern_size = (6, 9)

    fns = [os.path.join(datapath, fn) for fn in os.listdir(datapath)
        if "png" in fn or "jpg" in fn]

    camera_tan = Camera()
    flag_calib = cv.CALIB_FIX_K3+cv.CALIB_ZERO_TANGENT_DIST
    obj_pnts, img_pnts, rms = camera_tan.calibrate(fns, pattern_size, flag_calib, True)
    print("rms: ", rms)
    write_fs(os.path.join(datapath, "camera.yaml"), matrix=camera_tan.K,
        dist=camera_tan.dist, rms=rms, rvecs=camera_tan.rvecs, tvecs=camera_tan.tvecs)

    TV_u, TV_v, _, _ = camera_tan.TV_distortion()
    print("TV distortion: ", TV_u, TV_v)
    img_pnts_undistort = camera_tan.undistort_points(img_pnts)

    # calibration after undistort
    rms, matrix, dist, rvecs, tvecs = cv.calibrateCamera(obj_pnts.transpose(0, 2, 1),
        img_pnts_undistort.transpose(0, 2, 1), (camera_tan.w, camera_tan.h), None, None, flags=flag_calib)
    print("rms:", rms)
    camera_undistort = Camera((camera_tan.w, camera_tan.h), matrix, dist)
    TV_u, TV_v, _, _ = camera_undistort.TV_distortion()
    print("TV distortion: ", TV_u, TV_v)

    # tripoints3d = triangulation_two(img_pnts_undistort, camera_tan.K, cv.RANSAC, True)
    # plot_tripoints(tripoints3d)
    # plt.show()

    w, h = 1920, 1080
    params = read_fs("./datasets/data/image_cap_mono_3m/camera0.yml", "matrix", "dist")
    fns = ["./datasets/data/image_cap_mono_3m_test/1582708301.25.png", 
            "./datasets/data/image_cap_mono_3m_test/1582708321.31.png"]

    params = read_fs("./datasets/data/image_cap_mono_33cm/camera.yaml", "matrix", "dist")
    fns = ["./datasets/data/image_cap_mono_33cm_test/2.png", 
            "./datasets/data/image_cap_mono_33cm_test/5.png"]
    camera_33 = Camera((w, h), params[0], params[1])

    K = params[0]
    dist = params[1]

    images_test = Images(fns)
    images_test.pattern_size = (6, 9)
    obj_pnts, img_pnts = images_test.find_chess_img_pnts(True)
    tripoints3d = triangulation_two(img_pnts, K, cv.RANSAC, True)
    tripoints3d = triangulation_two(img_pnts, K, cv.RANSAC, False)
    plot_tripoints(tripoints3d)

    # img1 = cv.imread(fns[0], -1)
    # img1_un = cv.undistort(img1, camera_33.K, camera_33.dist, None, None)
    # img2 = cv.imread(fns[1], -1)
    # img2_un = cv.undistort(img2, camera_33.K, camera_33.dist, None, None)
    imgs_undistort = camera_33.undistort_imgs(fns)
    for i, img in enumerate(imgs_undistort):
        name = os.path.basename(fns[i])[:-4]
        cv.imwrite("%s_undistort.jpg" %name, img)

    _, img_pnts_undistort = images_test.find_chess_img_pnts(True, 1, imgs_undistort)
    # tripoints3d = triangulation_two(img_pnts_undistort, K, cv.RANSAC, True)
    tripoints3d = triangulation_two(img_pnts_undistort, K, cv.RANSAC, False)

    # img1_pnts, img2_pnts, num1, num2 = images_test.find_img_pnts_manual()
    # tripoints3d = triangulation_two([img1_pnts, img2_pnts], K, cv.RANSAC, True)


    # fns = ["./datasets/data/image_cap_mono_33cm_test/1582692002.72.png", 
    #         "./datasets/data/image_cap_mono_33cm_test/1582692020.34.png"]
    # images_test = Images(fns)
    # images_test.pattern_size = (6, 9)
    # obj_pnts, img_pnts = images_test.find_chess_img_pnts()
    # tripoints3d = triangulation_two(img_pnts, camera_tan.K, cv.RANSAC, True)

    plot_tripoints(tripoints3d)
    plt.show()


    def sfm_2plane():
        print("3D images")
        fns = ["./datasets/oppo/test/3d_5.jpg", "./datasets/oppo/test/3d_6.jpg"]
        images_3d = Images(fns)
        points1, points2, num_plane1, num_plane2 = images_3d.find_img_pnts_manual()
        # tripoints3d = triangulation_two([points1[:, :num_plane1], points2[:, :num_plane1]], camera_tan.K)
        tripoints3d = triangulation_two([points1, points2], camera_tan.K)
        vec1, vec2 = plot_tripoints(tripoints3d, num_plane1, num_plane2)
        plt.show()
        angle = np.arccos(np.dot(vec1[:3], vec2[:3]) / np.linalg.norm(vec1) / np.linalg.norm(vec2))
        print(np.dot(vec1[:3], vec2[:3]))
        print(angle*180/np.pi)
    # sfm_2plane()

