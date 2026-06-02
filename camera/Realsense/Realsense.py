import pyrealsense2 as rs
import numpy as np
import depthai as dai
import matplotlib.pyplot as plt
import cv2


class RealsenseCamera:
    def __init__(self, width=1280, height=720):
        self.im_height = 720
        self.im_width = 1280
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        self.color_profile = profile.get_stream(rs.stream.color)
        self.align = rs.align(rs.stream.color)

    def get_intrinsics(self):
        intr = self.color_profile.as_video_stream_profile().get_intrinsics()
        camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        coeffs = intr.coeffs
        return camera_matrix, np.array(coeffs)

    def get_color_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def get_data0(self):
        # Return color image and depth image
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        depth_img = np.asarray(depth.get_data())
        color_img = np.asarray(color.get_data())
        # Get camera intrinsics
        self.intrinsics = [918.27160645, 0., 643.14483643, 0., 918.02313232, 357.28491211, 0., 0.,
                           1]  # Change me!!!!!!!

        return color_img, depth_img

    def get_data1(self):
        # 初始化管道和配置
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, 30)

        # 启动并等待对齐帧
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        for _ in range(5):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

        # 取对齐后的图像
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("对齐失败，未获取到帧")

        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        # 使用手动设置的内参
        self.intrinsics = [918.27160645, 0., 643.14483643,
                           0., 918.02313232, 357.28491211,
                           0., 0., 1]

        return color_img, depth_img

    def get_data(self):
        import pyrealsense2 as rs
        import numpy as np

        # 初始化管道和配置
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, 30)

        # 启动并等待对齐帧
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        # 等待几帧稳定
        for _ in range(5):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

        # 获取对齐后的帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("对齐失败，未获取到帧")

        # 转换为numpy图像
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        # 获取深度相机内参
        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = depth_intrin  # 这将是 pyrealsense2.intrinsics 类型

        # 停止相机（如果只想采一帧）
        pipeline.stop()

        return color_img, depth_img, depth_frame

    def stop(self):
        self.pipeline.stop()


class OAKCamera:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)

        # https://docs.oakchina.cn/projects/api/samples/Camera/camera_undistort.html 使用video节点，用于拍照
        xoutRgb.setStreamName("still")

        # Properties
        camRgb.setPreviewSize(width, height)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        # Linking
        camRgb.preview.link(xoutRgb.input)

        self.device = dai.Device(pipeline)

        qRgb = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        color_image = qRgb.get().getCvFrame()

    def get_color_image(self):
        qRgb = self.device.getOutputQueue(name="still", maxSize=4, blocking=False)
        color_image = qRgb.get()
        color_image = color_image.getCvFrame()
        return color_image

    def get_intrinsics(self):
        calibData = self.device.readCalibration()
        camera_matrix = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.width, self.height))
        print(f"RGB Camera resized intrinsics... {self.width} x {self.height}:\n {camera_matrix}")
        coeffs = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        print("Coefficients...")
        [print(name + ": " + value) for (name, value) in
         zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
             [str(data) for data in coeffs])]
        return camera_matrix, coeffs


def create_camera(cam_type, height, width):
    if cam_type == "realsense":
        return RealsenseCamera(height, width)
    elif cam_type == "oak":
        return OAKCamera(height, width)


if __name__ == '__main__':
    # 创建 RealSense 相机实例
    camera = RealsenseCamera(width=1280, height=720)

    # 获取相机内参和畸变系数
    camera_matrix, distortion_coeffs = camera.get_intrinsics()
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", distortion_coeffs)

    # 捕获彩色图像
    color_image = camera.get_color_image()
    if color_image is not None:
        # 使用 OpenCV 显示图像
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()
    else:
        print("Failed to capture color image.")
    # cv2.imwrite('PIC/2.jpg', color_image)

    # 停止相机流
    camera.stop()
