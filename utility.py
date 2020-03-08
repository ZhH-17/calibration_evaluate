import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import structure
import pdb

def find_img_pnts(img_names, pattern_size, square_size=1, debug=False):
    # compute object and find image points of img_names
    threads = 2

    debug_dir = os.path.join(os.path.dirname(img_names[0]), './output_debug/')
    if debug and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

    def processImage(fn):
        img = cv.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0],\
            ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            name = (os.path.basename(fn)).split('.')[0]
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('           %s chessboard not found' % fn)
            return None
        print('           %s... OK' % fn)
        return (corners.reshape(-1, 2), pattern_points)

    threads_num = threads
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    return obj_points, img_points, w, h

def reconstruct_pose_cv(points1, points2, camera_matrix, method=cv.RANSAC):
    """ Computes the fundamental or essential matrix from corresponding points
    using opencv function
    :input p1, p2: corresponding pixel points with shape n x 2
    :returns: fundamental or essential matrix with shape 3 x 3
    pose1 and pose2 with shape 3 x 4
    """
    F_mat, mask = cv.findFundamentalMat(points1, points2)

    E_mat, mask = cv.findEssentialMat(points1, points2, camera_matrix, method, 0.99, 3)
    # print("Essential matrix is: ", E_mat)
    
    retval, R, t, mask = cv.recoverPose(E_mat, points1, points2, camera_matrix, 10)
    T1 = np.zeros((3, 4), np.float32)
    for i in range(3): T1[i, i] = 1.0
    T2 = np.zeros((3, 4), np.float32)
    T2[:3, :3] = R
    T2[:, 3] = t.reshape(-1)
    return F_mat, E_mat, T1, T2

# def plot_tripoints(tripoints3d):
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)

def fit_3dplane(points3d):
    # ax+by+d=z, return a,b,d
    A = np.ones((len(points3d[0]), 3))
    A[:, 0] = points3d[0]
    A[:, 1] = points3d[1]
    B = points3d[2]
    A_pinv = np.linalg.pinv(A)
    c_a, c_b, c_d = coefs = np.dot(A_pinv, B)
    res = np.dot(A, coefs) - B
    print("plane fit: %", abs(res).mean()/abs(B.mean())*100)
    r_ab = np.sqrt(c_a**2 + c_b**2)
    dist = np.array([(c_a*pnt[0]+c_b*pnt[1]-pnt[2]+c_d)/r_ab for pnt in points3d.T])
    print("mean distance from fitting plane: ", abs(dist).mean())
    minx, maxx = points3d[0].min(), points3d[0].max()
    miny, maxy = points3d[1].min(), points3d[1].max()
    return (c_a, c_b, c_d), (minx, maxx, miny, maxy)

def plot_tripoints(points3d, num_plane1=-1, num_plane2=-1):
    # plot 3d points and fitting plane in a figure
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    vec1, vec2 = None, None
    if num_plane1 <= 0 and num_plane2 <= 0:
        ax.plot(points3d[0], points3d[1], points3d[2], 'b.')
        (c_a, c_b, c_d), (minx, maxx, miny, maxy) = fit_3dplane(points3d)
        xx, yy = np.meshgrid(np.linspace(minx, maxx, 40), np.linspace(miny, maxy, 40))
        zz = c_a*xx + c_b*yy + c_d
        ax.plot_surface(xx, yy, zz, alpha=0.2)
    elif num_plane1 > 0:
        ax.plot(points3d[0, :num_plane1], points3d[1, :num_plane1], points3d[2, :num_plane1], 'b.')
        (c_a, c_b, c_d), (minx, maxx, miny, maxy) = fit_3dplane(points3d[:, :num_plane1])
        xx, yy = np.meshgrid(np.linspace(minx, maxx, 40), np.linspace(miny, maxy, 40))
        zz = c_a*xx + c_b*yy + c_d
        ax.plot_surface(xx, yy, zz, alpha=0.2)
        vec1 = np.array([c_a, c_b, -1, c_d])
    elif num_plane2 > 0:
        ax.plot(points3d[0, num_plane1:], points3d[1, num_plane1:], points3d[2, num_plane1:], 'r.')
        (c_a, c_b, c_d), (minx, maxx, miny, maxy) = fit_3dplane(points3d[:, -num_plane2:])
        xx, yy = np.meshgrid(np.linspace(minx, maxx, 40), np.linspace(miny, maxy, 40))
        zz = c_a*xx + c_b*yy + c_d
        ax.plot_surface(xx, yy, zz, alpha=0.2)
        vec2 = np.array([c_a, c_b, -1, c_d])

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    return vec1, vec2

def find_correspondence_points(img1, img2):
    # find the keypoints and descriptors with ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    if len(matches)>30: good = matches[:30]
    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T

def triangulatation_self(points1n, points2n):
    # achieved self
    # points1 = img_pnts[0]
    # points2 = img_pnts[1]

    # one_vec = np.ones((1, points1.shape[1]))
    # K_inv = np.linalg.inv(K)
    # pnts1 = np.vstack((points1, one_vec))
    # pnts2 = np.vstack((points2, one_vec))
    # points1n = np.dot(K_inv, pnts1) # 3 x n
    # points2n = np.dot(K_inv, pnts2) # 3 x n

    E = structure.compute_essential_normalized(points1n, points2n)
    print('Computed essential matrix:', (-E / E[0][1]))

    # Given we are at camera 1, calculate the parameters for camera 2
    # Using the essential matrix returns 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = structure.reconstruct_one_point(
            points1n[:, 0], points2n[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)
    return tripoints3d, E, P1, P2


def triangulation_two(img_pnts, K, method=cv.RANSAC, myself=False):
    """ Computes 3D points from corresponding image points
    :input img_pnts: corresponding points with shape 2 x 2 x n
    :returns: 3D points 3 x n
    """
    points1 = img_pnts[0]
    points2 = img_pnts[1]

    one_vec = np.ones((1, points1.shape[1]))
    K_inv = np.linalg.inv(K)
    pnts1 = np.vstack((points1, one_vec))
    pnts2 = np.vstack((points2, one_vec))
    points1n = np.dot(K_inv, pnts1) # 3 x n
    points2n = np.dot(K_inv, pnts2) # 3 x n

    if myself:
        tripoints3d, E_mat, T1, T2 = triangulatation_self(points1n, points2n)
    else:
        P_mat, E_mat, T1, T2 = reconstruct_pose_cv(points1.T, points2.T, K)
        tripoints3d = cv.triangulatePoints(T1, T2, points1n[:2, :], points2n[:2, :])
    return tripoints3d / tripoints3d[3]
    # return tripoints3d

def cart2hom(arr):
    """ Convert catesian to homogenous points by appending a row of 1s
    :param arr: array of shape (num_dimension x num_points)
    :returns: array of shape ((num_dimension+1) x num_points) 
    """
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

def write_fs(fn, **kwargs):
    fs = cv.FileStorage(fn, cv.FILE_STORAGE_WRITE)
    for key in kwargs.keys():
        fs.write(key, kwargs[key])
    fs.release()

def read_camera_fs(fn):
    fs = cv.FileStorage(fn, cv.FILE_STORAGE_READ)
    cameramatrix = fs.getNode('matrix').mat()
    dist = fs.getNode('distoration').mat()
    extrinsics = fs.getNode('extrinsics').mat()
    rvecs = extrinsics[:, :3]
    tvecs = extrinsics[:, 3:]
    fs.release()
    return cameramatrix, dist, rvecs, tvecs

def read_fs(fn, *args):
    # read specified nodes' mat
    fs = cv.FileStorage(fn, cv.FILE_STORAGE_READ)
    values = [fs.getNode(arg).mat() for arg in args]
    fs.release()
    return values