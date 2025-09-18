#!/usr/bin/env python3

import gi
import rospy
import numpy as np
import signal
import netifaces
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

# GStreamer 초기화
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib, GObject

Gst.init(None)

def get_local_ip():
    interface_name = "wlx503eaa86b3e9"  # Wi-Fi 인터페이스 이름
    try:
        ifaddresses = netifaces.ifaddresses(interface_name)
        if netifaces.AF_INET in ifaddresses:
            for link in ifaddresses[netifaces.AF_INET]:
                ip = link.get('addr')
                if ip and not ip.startswith("127."):
                    return ip
    except Exception as e:
        rospy.logwarn(f"❌ IP 탐색 실패: {e}")
    return "127.0.0.1"

class ResultImageRtspFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, bridge):
        super(ResultImageRtspFactory, self).__init__()
        self.bridge = bridge
        self.frame = np.zeros((int(480 * 1.5), 640), dtype=np.uint8)  # 초기 프레임
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND
        self.number_frames = 0

        self.launch_string = (
            "appsrc name=source is-live=true block=true format=time do-timestamp=true "
            "caps=video/x-raw,format=I420,width=640,height=480,framerate=30/1 "
            "! x264enc speed-preset=ultrafast tune=zerolatency key-int-max=15 intra-refresh=true threads=1 "
            "! rtph264pay config-interval=1 name=pay0 pt=96"
        )

        self.sub = rospy.Subscriber(
            "/pepper_detection/output_image", Image, self.image_callback, queue_size=1
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            resized = cv2.resize(cv_image, (640, 480))
            yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
            self.frame = yuv
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge 변환 실패: {e}")

    def on_need_data(self, src, length):
        if self.frame is None:
            return

        data = self.frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        buf.pts = buf.dts = int(self.number_frames * self.duration)
        self.number_frames += 1
        retval = src.emit("push-buffer", buf)
        if retval != Gst.FlowReturn.OK:
            print(f"⚠️ push-buffer 실패: {retval}")

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        appsrc = rtsp_media.get_element().get_child_by_name("source")
        appsrc.connect("need-data", self.on_need_data)

class ResultImageRtspServer:
    def __init__(self):
        self.bridge = CvBridge()
        self.server = GstRtspServer.RTSPServer()
        self.server.props.service = "8554"

        factory = ResultImageRtspFactory(self.bridge)
        factory.set_shared(True)
        mount_points = self.server.get_mount_points()
        mount_points.add_factory("/live", factory)

        self.server.attach(None)

        ip = get_local_ip()
        rospy.loginfo(f"✅ RTSP 서버 실행됨: rtsp://{ip}:8554/live")

        self.loop = GLib.MainLoop()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        rospy.loginfo("🛑 RTSP 서버 종료 중...")
        self.loop.quit()

    def run(self):
        self.loop.run()

if __name__ == '__main__':
    rospy.init_node("result_image_rtsp_server")
    server = ResultImageRtspServer()
    server.run()

