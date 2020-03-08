# for images
import cv2 as cv
import numpy as np
import os

class Images:
    def __init__(self, fns, img_type=0, camera_type=1):
        ''' some images '''
        if len(fns) == 0:
            print("No images")
        self.fns = fns
        images = []
        for fn in fns:
            img = cv.imread(fn, -1)
            # 0: bgr image
            if img_type == 1:
                # bayer BG iamge
                img = cv.cvtColor(img, cv.COLOR_BayerBG2BGR)
            elif img_type == 2:
                # gray image
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            images.append(img)
        self.images = images
        self.num = len(images)
        # TODO: monocular or stereo, not considered now
        self.camera_type = camera_type 
        self.h, self.w = images[0].shape[:2]
        self.pattern_size = None

    def find_correspondence_points(self, img1=None, img2=None):
        # find 2 images correspond points
        # return: pnts1, pnts2 with shape 2 x n
        if img1 is None:
            img1 = self.images[0]
            img2 = self.images[1]
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        distances = [m.distance for m in matches]
        if len(matches) == 0:
            print("No matches!!!")
            return None, None
        good = matches
        if len(matches) > 80:
            good = matches[:80]
        src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

        return src_pts.T, dst_pts.T, distances

    def find_chess_img_pnts(self, debug=False, square_size=1., imgs=None):
        # compute object and find image points for chessborad
        # retrun N x 2 x n points
        if imgs is None:
            imgs = self.images

        self.debug_dir = None
        if debug:
            self.debug_dir = os.path.join(os.path.dirname(self.fns[0]), 'debug')
            if not os.path.isdir(self.debug_dir):
                os.mkdir(self.debug_dir)
        threads = 2

        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        obj_points = []
        img_points = []
        h, w = self.h, self.w

        threads_num = threads
        if threads_num <= 1:
            chessboards = [self.__processImage(self, i, fn) for i, fn 
                                            in enumerate(imgs)]
        else:
            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(threads_num)
            chessboards = pool.starmap(self.__processImage, enumerate(imgs))

        chessboards = [x for x in chessboards if x is not None]
        for corners in chessboards:
            img_points.append(corners.T)
            obj_points.append(pattern_points.T)

        return np.array(obj_points), np.array(img_points)

    def __processImage(self, i, img):
        corners = self.processImage(img, self.fns[i])
        return corners

    def processImage(self, img, fn):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img is None:
            print("image %s is None" %fn)
            return None

        assert self.w == img.shape[1] and self.h == img.shape[0],\
            ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, self.pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if self.debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, self.pattern_size, corners, found)
            name = (os.path.basename(fn)).split('.')[0]
            outfile = os.path.join(self.debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('           %s chessboard not found' % fn)
            return None
        print('           %s... OK' % fn)
        return corners.reshape(-1, 2)

    def find_img_pnts_manual(self, nplane=2, img1=None, img2=None):
        # find correspond points manually
        # return 2 x n
        if img1 is None:
            img1=self.images[0]
            img2=self.images[1]
        pts1, pts2, distances = self.find_correspondence_points(img1, img2)

        mask_inplane1 = verify_matches(img1, img2, pts1.T, pts2.T)
        pnts1_plane1 = pts1[:, mask_inplane1]
        pnts2_plane1 = pts2[:, mask_inplane1]
        retval, mask_h = cv.findHomography(pnts1_plane1.T, pnts2_plane1.T, cv.RANSAC, 20.0)
        pnts1_plane1 = pnts1_plane1[:, mask_h.ravel()>0]
        pnts2_plane1 = pnts2_plane1[:, mask_h.ravel()>0]
        print("%d points in on plane1" %(pnts1_plane1.shape[1]))
        # pnts1 = np.hstack((pnts1_plane1, pnts1_plane2))
        # pnts2 = np.hstack((pnts2_plane1, pnts2_plane2))

        if nplane==2:
            pts1_tmp = pts1[:, ~mask_inplane1]
            pts2_tmp = pts2[:, ~mask_inplane1]
            mask_inplane2 = verify_matches(img1, img2, pts1_tmp.T, pts2_tmp.T)
            pnts1_plane2 = pts1_tmp[:, mask_inplane2]
            pnts2_plane2 = pts2_tmp[:, mask_inplane2]
            if pnts1_plane2.shape[1] > 0:
                retval, mask_h = cv.findHomography(pnts1_plane2.T, pnts2_plane2.T, cv.RANSAC, 20.0)
                pnts1_plane2 = pnts1_plane2[:, mask_h.ravel()>0]
                pnts2_plane2 = pnts2_plane2[:, mask_h.ravel()>0]
                print("%d points in on plane2" %(pnts1_plane2.shape[1]))
            return np.hstack((pnts1_plane1, pnts1_plane2)), np.hstack((pnts2_plane1, pnts2_plane2)), \
                pnts1_plane1.shape[1], pnts1_plane2.shape[1]
        elif nplane==1:
            return pnts1_plane1, pnts2_plane1, pnts1_plane1.shape[1], -1

    def find_img_pnts_blob(self, debug=False, square_size=1.):
        # find LED blob
        th_dist = 90
        nrows, ncols = self.pattern_size
        ids_verify_pnt = [0, 1, 2, noclos, 2*ncols]

        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        verify_pnts = pattern_points[ids_verify_pnt]

        for img in self.images:
            detected_pnts = detect_blob_weak(img)
            flag, img_pnts, pnts_prj, homography = \
                project_by_h(detected_pnts, pattern_points, verify_pnts, th_dist)
            if flag:
                print("success to detect %s" %fn)
                pnts_id_closest, pnts_id_obj_closest, H_new = find_best_H(
                    detected_pnts, pattern_points, homography, th_dist)


        # detected_pnts = detect_blob_weak(img)
        # flag, img_pnts, pnts_prj, homography = \
        #     project_by_h(detected_pnts, obj_pnts, obj_pnts_used, th_dist)
        # if flag:
        #     if debug:
        #         # check if initial H is right
        #         save_img_pnts(img, img_pnts, pnts_prj,
        #                       os.path.join(debug_path, fn[:-4]+"_prj.png"))
        #     print("success to detect %s" %fn)
        #     pnts_id_closest, pnts_id_obj_closest, H_new = find_best_H(
        #         detected_pnts, obj_pnts, homography, th_dist)

        #     pnts_closest = np.array([detected_pnts[id] for id in pnts_id_closest],
        #                             dtype=np.float32)
        #     pnts_img_filtered[0].append(pnts_closest)
        #     pnts_obj_closest = np.array([obj_pnts[id] for id in pnts_id_obj_closest],
        #                                 dtype=np.float32)
        #     pnts_obj_filtered[0].append(pnts_obj_closest)

        #     if debug:
        #         pnts_prj_new = cv.perspectiveTransform(
        #             pnts_obj_closest[:, :2].reshape(-1, 1, 2), H_new).reshape(-1, 2)
        #         save_img_pnts(img, pnts_closest, pnts_prj_new,
        #                       os.path.join(debug_path, fn[:-4]+"_prj_filtered.png"))
        # else:
        #     print("   fail to detect %s left" %fn)
        #     continue




num_choose = 0
px, py = -1, -1
pnts_rec = np.zeros((2, 2), np.int)
img_tmp = None
def onMouse(event, x, y, flags, param):
    global num_choose, px, py, pnts_rec
    global img_tmp
    if event == cv.EVENT_LBUTTONDOWN and num_choose == 0:
        pnts_rec[0] = x, y
        num_choose += 1
        img_tmp1 = img_tmp.copy()
        cv.drawMarker(img_tmp, (x, y), (255, 0, 0), 0, 40, 2)
    elif event == cv.EVENT_LBUTTONDOWN and num_choose == 1:
        pnts_rec[1] = x, y
        num_choose += 1
        cv.drawMarker(img_tmp, (x, y), (255, 0, 0), 0, 40, 2)
    elif event == cv.EVENT_RBUTTONDBLCLK:
        px, py = x, y
def in_rect(pnt0, pnt1, pnts):
    # find pnts in rectangle
    from shapely.geometry import Polygon, Point
    x0, y0 = pnt0; x1, y1 = pnt1
    polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    mask = list(map(polygon.contains, [Point(x, y) for (x,y) in pnts]))
    return np.array(mask)
    
def verify_matches(img1, img2, pnts1, pnts2):
    # pnts1, pnts2 with shape n x 2
    from shapely.geometry import Polygon, Point
    global num_choose, px, py, pnts_rec
    global img_tmp
    h, w =img1.shape[:2]
    img_tmp = np.zeros((h, 2*w, 3), np.uint8)
    img_tmp[:, :w] = img1
    img_tmp[:, w:] = img2
    img_copy = img_tmp.copy()
    windowname = "img"
    cv.namedWindow("img", cv.WINDOW_NORMAL)
    for i in range(len(pnts1)):
        pnt0 = tuple(pnts1[i].astype(int))
        pnt1 = tuple((pnts2[i]+(w, 0)).astype(int))
        cv.drawMarker(img_tmp, pnt0, (0, 0, 255), 1, 20, 2)
        cv.drawMarker(img_tmp, pnt1, (0, 0, 255), 1, 20, 2)
        cv.line(img_tmp, pnt0, pnt1, [0,0,255], 2)
    cv.imshow(windowname, img_tmp)

    mask_in = np.full((len(pnts1)), True, bool)
    radius = 40
    cv.setMouseCallback(windowname, onMouse)
    while True:
        key = cv.waitKey(1)
        redraw = False
        if key == 27:
            break
        elif key == ord("d"):
            if num_choose == 2:
                # delete points in polygon
                x0, y0 = pnts_rec[0]; x1, y1 = pnts_rec[1]
                if x0 > w:
                    x0 -= w; x1 -= w
                    pnts = pnts2
                else:
                    pnts = pnts1
                mask = in_rect((x0, y0), (x1, y1), pnts)
                mask_in &= ~mask
                redraw = True
                num_choose = 0
            elif px > 0:
                pnt_del = [px, py]
                if px > w:
                    pnt_del[0] = px - w
                    p_shift = pnts2 - pnt_del
                else:
                    p_shift = pnts1 - pnt_del
                distances = np.power(p_shift[:, 0], 2) + np.power(p_shift[:, 1], 2)
                mask = distances > 400
                mask_in &= mask
                redraw = True
        elif key == 13 and num_choose == 2:
            x0, y0 = pnts_rec[0]; x1, y1 = pnts_rec[1]
            if x0 > w:
                x0 -= w; x1 -= w
                pnts = pnts2
            else:
                pnts = pnts1
            mask = in_rect((x0, y0), (x1, y1), pnts)
            # polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
            # mask = list(map(polygon.contains, [Point(x, y) for (x,y) in pnts]))
            mask_in &= mask
            redraw = True
            num_choose = 0
        elif key == ord("r"):
            num_choose = 0
            px = -1
            redraw = True

        # for i in range(num_choose):
        #     cv.drawMarker(img_tmp, tuple(pnts_rec[i]), (255, 0, 0), 2, 40, 2)
        if num_choose == 2:
            cv.rectangle(img_tmp, tuple(pnts_rec[0]), tuple(pnts_rec[1]), [255, 0, 0], 4)

        if redraw:
            img_tmp = img_copy.copy()
            for p1, p2 in zip(pnts1[mask_in], pnts2[mask_in]):
                pnt0 = tuple(p1.astype(int))
                pnt1 = tuple((p2+(w, 0)).astype(int))
                cv.drawMarker(img_tmp, pnt0, (0, 0, 255), 1, 20, 2)
                cv.drawMarker(img_tmp, pnt1, (0, 0, 255), 1, 20, 2)
                cv.line(img_tmp, pnt0, pnt1, [0,0,255], 2)
            redraw = False
        if px>0:
            cv.circle(img_tmp, (px, py), radius, [255, 0, 0], 2)

        cv.imshow(windowname, img_tmp)
    cv.destroyAllWindows()
    return mask_in
