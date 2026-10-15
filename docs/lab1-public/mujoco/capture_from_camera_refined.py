import cv2
import mujoco


def main():
    model = mujoco.MjModel.from_xml_path("world1.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="main_camera")
    image = renderer.render()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(image)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids, (255, 0, 0))

    cv2.imwrite("main_camera_markers_tracked.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
