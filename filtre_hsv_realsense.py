import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

def nothing(x):
    pass

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

def determine_mask_hsv(nameimageexportee='image.png'):
     """
    Permet de déterminer le masque hsv d'une image avec une interface graphique.

    Parameters:
    - nameimageexportee (str): Le nom du fichier image pour enregistrer le résultat. Par défaut, 'image.png'.

    Returns:
    - Tuple: Un tuple contenant les plages HSV inférieures et supérieures déterminées lors de l'ajustement dans l'interface graphique.
    """
    state = AppState()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()

    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L-H","Trackbars",0,179,nothing)
    cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
    cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
    cv2.createTrackbar("U-H","Trackbars",179,179,nothing)
    cv2.createTrackbar("U-S","Trackbars",255,255,nothing)
    cv2.createTrackbar("U-V","Trackbars",255,255,nothing)

    def project(v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h) / w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w * view_aspect, h) + (w / 2.0, h / 2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def view(v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

    def pointcloud(out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front
            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5 ** state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)

        out[i[m], j[m]] = color[u[m], v[m]][:, ::-1]

    out = np.empty((h, w, 3), dtype=np.uint8)

    lecture = True

    while lecture == True:
        # Grab camera data
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            frame = np.asanyarray(color_frame.get_data())
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
            l_h = cv2.getTrackbarPos("L-H", "Trackbars")
            l_s = cv2.getTrackbarPos("L-S", "Trackbars")
            l_v = cv2.getTrackbarPos("L-V", "Trackbars")
            u_h = cv2.getTrackbarPos("U-H", "Trackbars")
            u_s = cv2.getTrackbarPos("U-S", "Trackbars")
            u_v = cv2.getTrackbarPos("U-V", "Trackbars")
        
            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])

            mask = cv2.inRange(hsv,lower_blue,upper_blue)
            result_BGR = cv2.bitwise_and(frame, frame, mask=mask)
            result = cv2.cvtColor(result_BGR, cv2.COLOR_BGR2RGB)

            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            depth_frame = decimate.process(depth_frame)

            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            
            
            depth_frame = decimate.process(depth_frame)

            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if state.color:
                color = color_image[:, 1]
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

        now = time.time()

        out.fill(0)


        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                            (w, h, 1.0 / dt, dt * 1000, "PAUSED" if state.paused else ""))
        

        cv2.imshow(state.WIN_NAME, out)

        
        key = cv2.waitKey(1)

        if key == ord("p"):
            state.paused = True

        if key == ord('s'):
            cv2.imwrite(nameimageexportee, result)
            break
        
        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

    print("Masque exporté")
    pipeline.stop()
    cv2.destroyAllWindows()
    return lower_blue,upper_blue
