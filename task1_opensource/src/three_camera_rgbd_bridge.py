#!/usr/bin/env python3
"""Grouped RGB-D bridge for head, left-gripper and right-gripper cameras."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from orcalab.protos.edit_service_wrapper import EditServiceWrapper
from orcalab_depth import load_depth_npy


CAMERAS = ("camera_head", "camera_left", "camera_right")


class ThreeCameraBridge(Node):
    def __init__(self, args):
        super().__init__("task1_three_camera_rgbd_bridge")
        self.args = args
        self.cameras = ("camera_head",) if args.head_only else CAMERAS
        self.output = (args.output_dir or Path(tempfile.mkdtemp(prefix="task1_3rgbd_"))).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.camera_publishers = {}
        for camera in self.cameras:
            base = f"/{camera}"
            self.camera_publishers[camera] = (
                self.create_publisher(Image, base + "/color/image_raw", 10),
                self.create_publisher(Image, base + "/depth/image_raw", 10),
                self.create_publisher(CameraInfo, base + "/color/camera_info", 10),
                self.create_publisher(CameraInfo, base + "/depth/camera_info", 10),
            )
        # Compatibility aliases keep RTAB-Map on the head camera.
        self.head_alias = (
            self.create_publisher(Image, "/camera/color/image_raw", 10),
            self.create_publisher(Image, "/camera/depth/image_raw", 10),
            self.create_publisher(CameraInfo, "/camera/color/camera_info", 10),
            self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10),
        )
        self.lock = threading.Lock()
        self.latest = None
        self.last_published = -1
        # The OrcaLab GetCameraDataPNG RPC does not expose a simulator/frame
        # timestamp.  The official OrcaGym GetTimeStamp RPC is deliberately
        # not called here: in the current move.json scene the native camera
        # frame index is -1 (native camera stream disabled), and calling
        # GetTimeStamp in that state crashes OrcaLab 26.7.1.  Anchor the
        # monotonic acquisition clock to the ROS clock
        # once, then stamp each group with the time at which the RPC/files
        # completed.  This makes the image timestamp describe the captured
        # state (and lets TF/odom history be queried at that state) instead of
        # the later executor publish callback.
        self._clock_anchor_monotonic_ns = time.monotonic_ns()
        self._clock_anchor_ros_ns = int(self.get_clock().now().nanoseconds)
        self._last_stamp_ns = 0
        self.capture_publish_lags = []
        self.stop_event = threading.Event()
        self.started = time.monotonic()
        self.successes = self.failures = 0
        self.publish_times = []
        self.last_error = None
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        self.create_timer(0.005, self._publish)

    def _thread_main(self):
        asyncio.run(self._capture())

    async def _capture(self):
        services = {}
        for camera in self.cameras:
            service = EditServiceWrapper(); service.init_grpc(self.args.edit_address)
            services[camera] = service
        index = self.args.start_index
        period = 1.0 / self.args.fps
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    capture_request_started = time.monotonic()
                    # One logical group: all three captures finish before any
                    # image is published. ROS stamps are assigned as a group.
                    results = []
                    for camera in self.cameras:
                        results.append(await services[camera].get_camera_data_png(
                            camera, str(self.output / camera), index
                        ))
                    if any(not x.has_color or not x.has_depth for x in results):
                        raise RuntimeError("one camera response lacks RGB or Depth")
                    frames = {}
                    for camera in self.cameras:
                        rgb_path = self.output / camera / "color" / f"{camera}_color_{index}.png"
                        depth_path = self.output / camera / "depth" / f"{camera}_depth_{index}.npy"
                        rgb, depth, meta = await asyncio.to_thread(
                            self._load_pair, rgb_path, depth_path
                        )
                        frames[camera] = (rgb, depth, meta)
                        if not self.args.keep_frames:
                            rgb_path.unlink(missing_ok=True); depth_path.unlink(missing_ok=True)
                    capture_request_finished = time.monotonic()
                    # The RPC has no frame timestamp.  The midpoint of the
                    # request/file interval is a less biased estimate of the
                    # render instant than the later file-decoding completion.
                    captured_monotonic = 0.5 * (
                        capture_request_started + capture_request_finished
                    )
                    with self.lock:
                        self.latest = (index, frames, captured_monotonic)
                    self.successes += 1
                except Exception as exc:
                    self.failures += 1; self.last_error = f"{type(exc).__name__}: {exc}"
                    if self.failures == 1 or self.failures % 10 == 0:
                        self.get_logger().warning(self.last_error)
                index += 1
                delay = period - (time.monotonic() - started)
                if delay > 0: await asyncio.sleep(delay)
        finally:
            await asyncio.gather(*[service.destroy_grpc() for service in services.values()])

    @staticmethod
    def _load_pair(rgb_path, depth_path):
        deadline = time.monotonic() + 3.0; last = None
        while time.monotonic() < deadline:
            try:
                with PilImage.open(rgb_path) as opened:
                    opened.load(); rgb = np.asarray(opened.convert("RGB"), np.uint8).copy()
                depth, meta = load_depth_npy(depth_path, (480, 640), 0.2, 1024.0)
                return rgb, depth, meta
            except (FileNotFoundError, EOFError, OSError, SyntaxError, ValueError) as exc:
                last = exc; time.sleep(0.01)
        raise TimeoutError(f"grouped RGB-D files incomplete: {last}")

    def _info(self, stamp, frame):
        fy = 480 / (2 * math.tan(math.radians(self.args.vertical_fov) / 2)); fx = fy
        msg = CameraInfo(); msg.header.stamp = stamp; msg.header.frame_id = frame
        msg.width = 640; msg.height = 480; msg.distortion_model = "plumb_bob"; msg.d = [0.0]*5
        msg.k = [fx,0,319.5,0,fy,239.5,0,0,1]; msg.r = [1,0,0,0,1,0,0,0,1]
        msg.p = [fx,0,319.5,0,0,fy,239.5,0,0,0,1,0]
        return msg

    @staticmethod
    def _image(stamp, frame, array, encoding):
        array = np.ascontiguousarray(array); msg = Image(); msg.header.stamp = stamp
        msg.header.frame_id = frame; msg.height, msg.width = array.shape[:2]
        msg.encoding = encoding; msg.is_bigendian = False; msg.step = array.strides[0]
        msg.data = array.tobytes(); return msg

    def _publish(self):
        with self.lock: group = self.latest
        if group is None or group[0] == self.last_published: return
        index, frames, captured_monotonic = group
        stamp_ns = self._stamp_from_capture(captured_monotonic)
        stamp = self._time_msg_from_ns(stamp_ns)
        self.capture_publish_lags.append(max(0.0, time.monotonic() - captured_monotonic))
        for camera, (rgb, depth, _meta) in frames.items():
            frame = camera + "_optical_frame"; info = self._info(stamp, frame)
            rgb_msg = self._image(stamp, frame, rgb, "rgb8")
            depth_msg = self._image(stamp, frame, depth, "32FC1")
            pubs = self.camera_publishers[camera]
            if camera == "camera_head":
                # The mapping pipeline consumes the compatibility /camera/*
                # aliases. Publishing the same 2.2 MB pair again on the
                # unused /camera_head/* topics doubles DDS serialization and
                # was enough to starve exact synchronizers during motion.
                if not self.args.head_only:
                    pubs[0].publish(rgb_msg); pubs[1].publish(depth_msg)
                    pubs[2].publish(info); pubs[3].publish(info)
                self.head_alias[0].publish(rgb_msg); self.head_alias[1].publish(depth_msg)
                self.head_alias[2].publish(info); self.head_alias[3].publish(info)
            else:
                pubs[0].publish(rgb_msg); pubs[1].publish(depth_msg)
                pubs[2].publish(info); pubs[3].publish(info)
        self.last_published = index; self.publish_times.append(time.monotonic())

    def _stamp_from_capture(self, captured_monotonic: float) -> int:
        """Map acquisition monotonic time into the node's ROS clock domain."""
        if self._clock_anchor_ros_ns > 0:
            stamp_ns = self._clock_anchor_ros_ns + int(
                (captured_monotonic * 1e9) - self._clock_anchor_monotonic_ns
            )
            # Never publish a future-stamped image if the executor was delayed
            # or the wall clock was adjusted while the bridge was running.
            stamp_ns = min(stamp_ns, int(self.get_clock().now().nanoseconds))
        else:
            # Formal OrcaLab runs use wall/system time (there is no /clock),
            # but retain a useful fallback for isolated unit/smoke tests.
            stamp_ns = time.time_ns()
        stamp_ns = max(stamp_ns, self._last_stamp_ns + 1)
        self._last_stamp_ns = stamp_ns
        return stamp_ns

    @staticmethod
    def _time_msg_from_ns(stamp_ns: int):
        from builtin_interfaces.msg import Time
        msg = Time()
        msg.sec = int(stamp_ns // 1_000_000_000)
        msg.nanosec = int(stamp_ns % 1_000_000_000)
        return msg

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=5)
        rate = ((len(self.publish_times)-1)/(self.publish_times[-1]-self.publish_times[0])
                if len(self.publish_times)>1 else 0.0)
        report = {"cameras": list(self.cameras), "elapsed_seconds": time.monotonic()-self.started,
                  "target_group_fps": self.args.fps, "successful_groups": self.successes,
                  "failed_groups": self.failures, "published_groups": len(self.publish_times),
                  "group_publish_fps": rate,
                  "timestamp_source": "rpc/file_capture_completion_mapped_to_ros_clock",
                  "official_orcagym_camera_timestamp": {
                      "status": "disabled_for_current_move_json",
                      "reason": "GetCurrentFrameIndex=-1; GetTimeStamp is unsafe when native camera stream is disabled",
                      "frame_index_source": "not queried by bridge; EditService.GetCameraDataPNG is the image source",
                  },
                  "capture_to_publish_latency_s": {
                      "mean": (float(np.mean(self.capture_publish_lags))
                                if self.capture_publish_lags else None),
                      "max": (float(np.max(self.capture_publish_lags))
                              if self.capture_publish_lags else None),
                      "samples": len(self.capture_publish_lags),
                  },
                  "all_images_same_ros_timestamp": True,
                  "all_six_images_same_ros_timestamp": len(self.cameras) == 3,
                  "last_error": self.last_error}
        self.args.report.parent.mkdir(parents=True, exist_ok=True)
        self.args.report.write_text(json.dumps(report, indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--edit-address",default="127.0.0.1:50151")
    p.add_argument("--fps",type=float,default=15); p.add_argument("--vertical-fov",type=float,default=90)
    p.add_argument("--start-index",type=int,default=5000000); p.add_argument("--output-dir",type=Path)
    p.add_argument("--report",type=Path,default=Path("task_1/data/three_camera_rgbd_report.json"))
    p.add_argument("--keep-frames",action="store_true")
    p.add_argument("--head-only", action="store_true",
                   help="capture only camera_head RGB-D for arms-down navigation")
    args=p.parse_args()
    rclpy.init(); node=ThreeCameraBridge(args)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.close(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__": main()
