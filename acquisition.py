import math
import time
import cv2
import pyrealsense2 as rs
import numpy as np

from typing import Tuple


class AppState:

    def __init__(self,*args, **kwargs) -> None:
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

    def reset(self) -> None:
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self) -> np.ndarray:
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self) -> np.ndarray:
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


def run_acquisition(point_cloud: str, image: str) -> None:
    """
    Acquiert des données à partir de la caméra RealSense.

    Parameters:
    - point_cloud (str): Le chemin du fichier PLY pour enregistrer le nuage de points.
    - image (str): Le chemin du fichier image pour enregistrer la capture couleur.

    Returns:
    None
    """
    state = AppState()

# Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
    pipeline.start(config)

# Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(
        profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()

    def mouse_cb(event: int, x: int, y: int, flags: int, param: dict) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True
        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True
        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:
            h, w = out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)

    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

    def project(v: np.ndarray) -> np.ndarray:
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def view(v: np.ndarray) -> np.ndarray:
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

    def line3d(out: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, color: np.ndarray = (0x80, 0x80, 0x80), thickness: int = 1) -> int:
        p0 = project(pt1.reshape(-1, 3))[0]
        p1 = project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    def grid(out: np.ndarray, pos: np.ndarray, rotation: np.ndarray = np.eye(3), size: float = 1, n: np.ndarray = 10, color: np.ndarray = (0x80, 0x80, 0x80)) -> None:
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
                   view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
                   view(pos + np.dot((s2, 0, z), rotation)), color)

    def axes(out: np.ndarray, pos: np.ndarray, rotation: np.ndarray = np.eye(3), size: float = 0.075, thickness: int = 2) -> None:
        line3d(out, pos, pos +
               np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        line3d(out, pos, pos +
               np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        line3d(out, pos, pos +
               np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)

    def frustum(out: np.ndarray, intrinsics, color: np.ndarray = (0x40, 0x40, 0x40)) -> None:

        orig = view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                line3d(out, orig, view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            line3d(out, view(top_left), view(top_right), color)
            line3d(out, view(top_right), view(bottom_right), color)
            line3d(out, view(bottom_right), view(bottom_left), color)
            line3d(out, view(bottom_left), view(top_left), color)

    def pointcloud(out: np.ndarray, verts: np.ndarray, texcoords: np.ndarray, color: np.ndarray, painter: bool = True) -> None:
        if painter:
            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68

        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = out.shape[:2]

    # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]

    out = np.empty((h, w, 3), dtype=np.uint8)

    while True:
        # Grab camera data
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_frame = decimate.process(depth_frame)

            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # Render
        now = time.time()

        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
        cv2.imshow("Color Image", color_image)
        cv2.imshow(state.WIN_NAME, out)
        key = cv2.waitKey(1)

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            state.decimate = (state.decimate + 1) % 3
            decimate.set_option(rs.option.filter_magnitude,
                                2 ** state.decimate)

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("s"):
            cv2.imwrite(image, color_image)
            points.export_to_ply(point_cloud, mapped_frame)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

    # Stop streaming
    pipeline.stop()


def points_and_colors_realsense(image_name: str = "image.png") -> Tuple[np.ndarray]:
    """
    Capturer les coordonnées 3D et les couleurs associées à partir de la caméra RealSense.

    Parameters:
    - image_name (str): Nom du fichier pour enregistrer l'image couleur.

    Returns:
    - vertices (np.array): Coordonnées 3D des points.
    - color_image (np.array): Image couleur correspondante.
    - depth_image (np.array): Image de profondeur associée.
    """
    try:
        # Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming
        pipeline.start(config)
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Get the 3D and 2D coordinates
        pc = rs.pointcloud()
        pc.map_to(depth_frame)
        points = pc.calculate(depth_frame)

        # Convert the coordinates to NumPy arrays
        # Les vertices correpondent à nos coordonnées 3D
        vertices = np.array(points.get_vertices())
        color_image = np.array(color_frame.get_data())
        color_image_rgb = color_image[:, :, [2, 1, 0]]
        cv2.imwrite(image_name, color_image_rgb)

    except Exception as e:
        print(e)
        pass
    return vertices, color_image

if __name__ == '__main__':
    ### Pour faire des acquisitions en masse

    import utils.convert as cv

    def colors_relasense_sofa(colors):
        # Permet de convertir les couleurs de la caméra realsense  en un format lisible par Sofa (ie on met l'image en ligne)
        l=len(colors)*len(colors[0])
        new_colors=np.asarray([(0,0,0) for i in range(l)])
        indice=0
        for i in range(len(colors)):
            for j in range(len(colors[0])):
                new_colors[indice]=colors[i][j]
                indice+=1
        return np.array(new_colors)

    name_object="_test_"
    name_folder="labo_biologie/acquisition_en_masse/"
    name_acquisition_thibaud=name_folder+name_object+"_Thibaud"
    name_acquisition_tinhinane=name_folder+name_object+"_Tinhinane"
    i=0
    while True:
        run_acquisition(name_acquisition_tinhinane+str(i)+".ply",name_acquisition_tinhinane+str(i)+".png")
        points,couleurs=points_and_colors_realsense(name_acquisition_thibaud+str(i)+".png")
        couleurs=colors_relasense_sofa(couleurs)
        cv.create_ply_file(points, couleurs, name_acquisition_thibaud+str(i)+".ply")
        i+=1
