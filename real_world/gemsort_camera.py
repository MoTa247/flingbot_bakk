"""Camera backends for the GEMSORT cell.

FlingBot's two cameras are network clients (KinectClient over HTTP, RealSense over a TCP server). Ours is a RealSense
mounted on Bernoulli's wrist and plugged into this PC, so `GemsortRealSense` talks to it directly with pyrealsense2
while exposing the same `get_rgbd(repeats) -> (rgb uint8 HxWx3, depth float32 HxW in metres)` interface, plus
`color_intr` (3x3) which FlingBot uses for pixel → world.

Which camera is used is stored in real_world/gemsort_calibration.json ("camera": {...}) and chosen by
`setup.get_top_cam()`, so nothing is hard-coded:
    {"camera": {"type": "realsense_local", "serial": null, "width": 1280, "height": 720, "fps": 30}}
    {"camera": {"type": "kinect", "ip": "192.168.0.10", "port": 8080}}
    {"camera": {"type": "realsense_tcp", "ip": "127.0.0.1", "port": 50010, "width": 1280, "height": 720}}
"""
import numpy as np

DEFAULT = dict(type='realsense_local', serial=None, width=1280, height=720, fps=30, align=True)


class GemsortRealSense:
    """Wrist RealSense on Bernoulli, read directly over USB (pyrealsense2), depth aligned to colour."""

    def __init__(self, serial=None, width=1280, height=720, fps=30, align=True, warmup=15):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color) if align else None
        self.scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(warmup):   # let auto-exposure settle
            self.pipeline.wait_for_frames()

    @property
    def color_intr(self):
        i = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile().get_intrinsics()
        return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], float)

    def get_rgbd(self, repeats=1):
        """Median over `repeats` frames (FlingBot calls this with repeats>1 to denoise)."""
        colours, depths = [], []
        for _ in range(max(1, int(repeats))):
            frames = self.pipeline.wait_for_frames()
            if self.align is not None:
                frames = self.align.process(frames)
            colours.append(np.asanyarray(frames.get_color_frame().get_data()))
            depths.append(np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self.scale)
        return np.median(colours, axis=0).astype(np.uint8), np.median(depths, axis=0).astype(np.float32)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def make_camera(config=None):
    """Build the configured top camera; `config` defaults to the calibration file's "camera" entry."""
    if config is None:
        from real_world.gemsort_calibrate import load
        config = load().get('camera', DEFAULT)
    config = {**DEFAULT, **(config or {})}
    kind = config.get('type', 'realsense_local')
    if kind == 'realsense_local':
        return GemsortRealSense(serial=config.get('serial'), width=config['width'], height=config['height'],
                                fps=config['fps'], align=config.get('align', True))
    if kind == 'kinect':
        from real_world.kinect import KinectClient
        return KinectClient(ip=config.get('ip', '127.0.0.1'), port=int(config.get('port', 8080)))
    if kind == 'realsense_tcp':
        from real_world.realsense import RealSense
        return RealSense(tcp_ip=config.get('ip', '127.0.0.1'), tcp_port=int(config.get('port', 50010)),
                         im_h=config['height'], im_w=config['width'], max_depth=float(config.get('max_depth', 3.)))
    raise ValueError(f'unknown camera type {kind!r} (realsense_local | kinect | realsense_tcp)')


def list_realsense():
    """Connected RealSense devices as [(serial, name)] — for the GUI dropdown."""
    try:
        import pyrealsense2 as rs
        return [(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name))
                for d in rs.context().devices]
    except Exception as error:
        return [('', f'pyrealsense2 unavailable: {error}')]
