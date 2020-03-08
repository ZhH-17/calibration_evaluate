''' camera class '''
import numpy as np
import cv2 as cv
from images import Images

class Camera:
    ''' describe camera'''
    def __init__(self, resolution=None, intrinsic=None, dist=None):
        if resolution is None:
            self.w, self.h = None, None
        else:
            self.w, self.h = resolution
        self.K = intrinsic
        self.dist = dist
        self.rvecs = None
        self.tvecs = None
        self.extrinsics = None
    
    def calibrate(self, fns, pattern_size, flag_calib=None, debug=False, square_size=1.):
        # calibrate camera
        images = Images(fns)
        self.w, self.h = images.w, images.h
        images.pattern_size = pattern_size
        obj_pnts, img_pnts = images.find_chess_img_pnts(debug)
        print("find points pair: ", len(img_pnts))

        flag_calib = cv.CALIB_FIX_K3+cv.CALIB_ZERO_TANGENT_DIST
        rms, matrix, dist, rvecs, tvecs = cv.calibrateCamera(obj_pnts.transpose(0, 2, 1),
            img_pnts.transpose(0, 2, 1), (images.w, images.h), None, None, flags=flag_calib)
        self.K = matrix
        self.dist = dist
        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)
        return obj_pnts, img_pnts, rms

    def project(self, points):
        ''' project 3D homogenous points 4 x n
            to 2D camera coordinate 2 x n '''
        if self.P is None:
            print("Pose P is not given")
            return None
        x = np.dot(self.P, points)
        x[0, :] /= x[2, :]
        x[1, :] /= x[2, :]
        return x
    
    def project_pixel(self, points):
        ''' project 3D homogenous points 4 x n
            to 2D camera coordinate 2 x n'''
        if self.T is not None:
            image_points, _ = cv.projectPoints(points[:3, :].T.reshape(1, -1, 3), 
                                            self.R, self.t, self.K, self.dist)
        return image_points.reshape(-1, 2).T

    def undistort_points(self, points_imgs):
        # undistort points in some images
        # points_imgs: N x n x 2
        # return: N x 2 x n
        import pdb
        # pdb.set_trace()
        # pnts_undistort = cv.undistortPoints(points_imgs.transpose(0, 2, 1), self.K, self.dist, None, self.K)
        # return pnts_undistort.transpose(0, 2, 1)
        points_undistort = []
        for pnts in points_imgs:
            pnts_undistort = self.undistort_point(pnts)
            points_undistort.append(pnts_undistort)
        return np.array(points_undistort)

    def undistort_point(self, points):
        pnts_undistort = cv.undistortPoints(np.expand_dims(points.T, 0), 
            self.K, self.dist, None, self.K)
        return np.squeeze(pnts_undistort).T

    def undistort_imgs(self, fns):
        # undistort images
        imgs = [cv.imread(fn, -1) for fn in fns]
        imgs_undistort = [cv.undistort(img, self.K, self.dist) for img in imgs]
        return imgs_undistort

    def pixel2normal(self, points):
        # project pixel points to normal camera coordinate
        # points: 2 x n
        one_vec = np.ones((1, points.shape[1]))
        pnts = np.vstack((points, one_vec))
        K_inv = np.linalg.inv(self.K)
        return np.dot(K_inv, pnts)

    def TV_distortion(self):
        # compute distortion pixel shift surface image in u,v direction
        mtx, dist = self.K, self.dist.ravel()
        h, w = self.h, self.w
        u0, v0 = mtx[0, 2], mtx[1, 2]
        alpha, beta = mtx[0, 0], mtx[1, 1]
        k1, k2 = dist[0], dist[1]
        p1, p2 = dist[2], dist[3]
        u = np.linspace(0, w, 2500)
        v = np.linspace(0, h, 2500)
        U, V = np.meshgrid(u, v)
        X = (U - u0)/alpha
        Y = (V - v0)/beta
        r_2 = X**2 + Y**2
        delta_u = (U-u0)*(k1*r_2 + k2*r_2**2) + \
            alpha*(2.*p1*X*Y + p2*(r_2 + 2.*X**2))
        delta_v = (V-v0)*(k1*r_2 + k2*r_2**2) + \
            beta*(p1*(r_2 + 2.*Y**2) + 2*p2*X*Y)
        delta_u_max = np.max(abs(delta_u))
        delta_v_max = np.max(abs(delta_v))
        TV_u = delta_u_max / w * 100.
        TV_v = delta_v_max / h * 100.
        return TV_u, TV_v, delta_u, delta_v

        # fig = plt.figure()
        # ax = fig.gca(projection="3d")
        # surf = ax.plot_surface(U, V, np.squeeze(delta_u))
        # delta_u_max = np.max(abs(delta_u))
        # ax.text(0.8*w, 0.8*h, delta_u_max, "max delta u: %.3f" %delta_u_max, color="blue")

        # ax_v = fig.gca(projection="3d")
        # surf_v = ax_v.plot_surface(U, V, np.squeeze(delta_v))
        # delta_v_max = np.max(abs(delta_v))
        # ax_v.text(0.8*w, 0.8*h, delta_v_max, "max delta v: %.3f" %delta_v_max, color="red")

        # ax.set_xlabel('u')
        # ax.set_ylabel('v')
        # ax.set_zlabel('delta')
        # debug_dir = "./output/"
        # if debug_dir and not os.path.isdir(debug_dir):
        #     os.mkdir(debug_dir)
        # filename = os.path.join(debug_dir, "pixel_shift.png")
        # i = 0
        # while os.path.isfile(filename):
        #     i += 1
        #     filename = os.path.join(debug_dir, "pixel_shift_%d.png" % i)
        # plt.savefig(filename)
        # plt.clf()

class Points:
    ''' Points '''
    def __init__(self, num):
        self.num = num
        self.points3D = None
        self.pointsPixel = None
        self.one_vec = np.ones(1, self.num) # for homogenous points