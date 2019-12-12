"""
Script to generate a dataset and perform hand-eye calibration using a Universal Robot UR5e robot.
The script communicates with the robot through TCP/IP. The host is defined by the user.

The entire sample consist of two additional files:
    - universal_robots_hand_eye_script.urp: Robot program script that moves between different poses.
    - robot_communication_file.xml: communication set-up file.

Running the sample requires that you have universal_robots_hand_eye_script.urp on your UR5e robot,
and robot_communication_file.xml in the same repo as this sample.
Each robot pose must be modified to your scene. This is done in universal_robots_hand_eye_script.urp on the robot.

Further explanation of this sample is found in our knowledge base:
Further explaination of this sample is found in our knowledge base:
https://zivid.atlassian.net/wiki/spaces/ZividKB/pages/103481384/
"""
from pathlib import Path
import time
import datetime

import cv2
import numpy as np
import zivid
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


def quat_to_rotm(quat: np.array) -> np.array:
    """Convert from quaternion to rotation matrix

    Args:
        quat: Rotations in quaternions

    Returns:
        Rotations in degrees as 3x3 rotation matrix
    """

    q_w = quat[0]
    q_x = quat[1]
    q_y = quat[2]
    q_z = quat[3]

    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, 0] = 1 - 2 * np.power(q_y, 2) - 2 * np.power(q_z, 2)
    rot_matrix[0, 1] = 2 * q_x * q_y - 2 * q_z * q_w
    rot_matrix[0, 2] = 2 * q_x * q_z + 2 * q_y * q_w
    rot_matrix[1, 0] = 2 * q_x * q_y + 2 * q_z * q_w
    rot_matrix[1, 1] = 1 - 2 * np.power(q_x, 2) - 2 * np.power(q_z, 2)
    rot_matrix[1, 2] = 2 * q_y * q_z - 2 * q_x * q_w
    rot_matrix[2, 0] = 2 * q_x * q_z - 2 * q_y * q_w
    rot_matrix[2, 1] = 2 * q_y * q_z + 2 * q_x * q_w
    rot_matrix[2, 2] = 1 - 2 * np.power(q_x, 2) - 2 * np.power(q_y, 2)

    return rot_matrix


def rotvec_to_quat(rotvec: np.array, angle: float = None) -> np.array:
    """Convert from rotation vector to quaternions

    Args:
        rotvec: Rotation vector
        angle: Angle of rotation

    Returns:
        Rotation in quaternions
    """

    if angle is None:
        angle = np.linalg.norm(rotvec)

    quat = np.zeros((4))
    quat[0] = np.cos(0.5 * angle)
    q_r = np.sqrt((1 - np.power(quat[0], 2)) / np.power(np.linalg.norm(rotvec), 2))
    quat[1] = q_r * rotvec[0]
    quat[2] = q_r * rotvec[1]
    quat[3] = q_r * rotvec[2]
    quat = quat / np.sign(quat[0])

    return quat


def _write_robot_state(
    con: rtde, input_data, finish_capture: bool = False, camera_ready: bool = False
):
    """Write to robot I/O registrer

    Args:
        con: Connection between computer and robot
        input_data: Input package containing the specific input data registers
        finish_capture: Boolean value to robot_state that q_r scene capture is finished
        camera_ready: Boolean value to robot_state that camera is ready to capture images
    """

    if finish_capture:
        input_data.input_bit_register_64 = 1
    else:
        input_data.input_bit_register_64 = 0

    if camera_ready:
        input_data.input_bit_register_65 = 1
    else:
        input_data.input_bit_register_65 = 0

    con.send(input_data)


def _initialize_robot_sync(host: str, port: int):
    """Set up communication with UR robot

    Args:
        host: IP address to host
        port: Port number

    Returns:
        Connection to robot
        Input package containing the specific input data registers

    Raises:
        RuntimeError: If computer is not able to establish comminucation with robot
    """

    conf = rtde_config.ConfigFile(Path(Path.cwd() / "robot_communication_file.xml"))
    output_names, output_types = conf.get_recipe("out")
    input_names, input_types = conf.get_recipe("in")

    # Rtde communication is by default port 30004.
    con = rtde.RTDE(host, port)
    con.connect()

    # To ensure that the application and further versions of UR controller is compatible
    if not con.negotiate_protocol_version():
        raise RuntimeError(f"Protocol do not match")

    if not con.send_output_setup(output_names, output_types, frequency=200):
        raise RuntimeError(f"Unable to configure output")

    input_data = con.send_input_setup(input_names, input_types)

    if not con.send_start():
        raise RuntimeError(f"Unable to start synchronization")

    print("Communication initialization completed. \n")

    return con, input_data


def _save_zdf_and_pose(
    save_dir: Path, image_num: int, frame: zivid.Frame, transform: np.array
):
    """Save data to folder

    Args:
        save_dir: Directory to save data
        image_num: Image number
        frame: Point cloud stored as .zdf
        transform: 4x4 transformation matrix
    """

    frame.save(save_dir / f"img{image_num:02d}.zdf")

    file_storage = cv2.FileStorage(
        str(save_dir / f"pos{image_num:02d}.yaml"), cv2.FILE_STORAGE_WRITE
    )
    file_storage.write("PoseState", transform)
    file_storage.release()


def _generate_folder():
    """Generate folder where dataset weill be stored

    Returns:
        Directory to where data will be saved
    """

    location_dir = Path(
        Path.cwd() / "datasets" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    if not location_dir.is_dir():
        location_dir.mkdir(parents=True)

    return location_dir


def _get_frame_and_transform_matrix(con: rtde, cam: zivid.Camera):
    """Capture image with Zivid camera and read robot pose

    Args:
        con: Connection between computer and robot
        cam: Zivid camera

    Returns:
        Zivid frame
        4x4 tranformation matrix
    """

    robot_state = _read_robot_state(con)
    frame = cam.capture()
    robot_pose = np.array(con.receive().actual_TCP_pose)

    translation = robot_pose[:3] * 1000
    rotvec = robot_pose[3:]
    transform = np.eye(4)
    # transform[:3, :3] = R.from_quat(R.from_rotvec(rotvec).as_quat()).as_rotvec
    transform[:3, :3] = quat_to_rotm(rotvec_to_quat(rotvec))
    transform[:3, 3] = translation.T

    return frame, transform


def _set_camera_settings(cam: zivid.Camera):
    """Set camera settings

    Args:
        cam: Zivid camera
    """

    settings = cam.settings
    settings.iris = 20
    settings.exposure_time = datetime.timedelta(microseconds=10000)
    settings.gain = 1
    settings.brightness = 1.0
    settings.filters.gaussian.enabled = 1


def _read_robot_state(con: rtde):
    """Recieve robot output recipe

    Args:
        con: Connection between computer and robot

    Returns:
        Robot state
    """

    robot_state = con.receive()

    assert robot_state is not None, "Not able to recieve robot_state"

    return robot_state


def pose_from_datastring(datastring: str):
    """Extract pose from q_r -yaml file saved by openCV

    Args:
        datastring: String of text from .yaml file

    Returns:
        Robotic pose as q_r zivid Pose class
    """

    string = datastring.split("data:")[-1].strip().strip("[").strip("]")
    pose_matrix = np.fromstring(string, dtype=np.float, count=16, sep=",").reshape(
        (4, 4)
    )
    return zivid.handeye.Pose(pose_matrix)


def _save_hand_eye_results(save_dir: Path, transform: np.array, residuals: list):
    """Save transformation and residuals to folder

    Args:
        save_dir: Path to where data will be saved
        transform: 4x4 transformation matrix
        residuals: List of residuals
    """

    file_storage_transform = cv2.FileStorage(
        str(save_dir / f"transformation.yaml"), cv2.FILE_STORAGE_WRITE
    )
    file_storage_transform.write("PoseState", transform)
    file_storage_transform.release()

    file_storage_residuals = cv2.FileStorage(
        str(save_dir / f"residuals.yaml"), cv2.FILE_STORAGE_WRITE
    )
    residual_list = []
    for res in residuals:
        tmp = list([res.translation, res.translation])
        residual_list.append(tmp)

    file_storage_residuals.write(
        "Per pose residuals for rotation in deg and translation in mm",
        np.array(residual_list),
    )
    file_storage_residuals.release()


def _image_count(robot_state):
    """Read robot output register 24

    Args:
        robot_state: Robot state

    Returns:
        Number of captured images
    """
    return robot_state.output_int_register_24


def _start_capture(robot_state):
    """Read robot output register 64

    Args:
        robot_state: Robot state

    Returns:
        Boolean value that states if camera is ready to capture
    """
    return robot_state.output_bit_register_64


def _finish_capture(robot_state):
    """Read robot input register 64

    Args:
        robot_state: Robot state

    Returns:
        Boolean value that states if camera is finished capturing
    """

    return robot_state.input_bit_register_64


def _ready_to_capture(robot_state):
    """Read robot input register 65

    Args:
        robot_state: Robot state

    Returns:
        Boolean value that states if camera is ready to capture
    """

    return robot_state.input_bit_register_65


def _verify_good_capture(frame: zivid.Frame):
    """Verify that checkeroard featurepoints are detected in the frame

    Args:
        frame: Zivid frame containing point cloud

    Raises:
        RuntimeError: If no feature points are detected in frame
    """

    point_cloud = frame.get_point_cloud()
    detected_features = zivid.handeye.detect_feature_points(point_cloud)

    if not detected_features:
        raise RuntimeError(f"Failed to detect feature points from captured frame.")


def _capture_one_frame_and_robot_pose(
    con: rtde,
    cam: zivid.Camera,
    save_dir: Path,
    input_data,
    image_num: int,
    ready_to_capture: bool,
):
    """Capture 3D image and robot pose for a given robot posture, then signals robot to move to next posture

    Args:
        con: Connection between computer and robot
        cam: Zivid camera
        save_dir: Path to where data will be saved
        input_data: Input package containing the specific input data registers
        image_num: Image number
        ready_to_capture: Boolean value to robot_state that camera is ready to capture images
    """

    frame, transform = _get_frame_and_transform_matrix(con, cam)
    _verify_good_capture(frame)

    # Signal robot to move to next position, then set signal to low again.
    _write_robot_state(
        con, input_data, finish_capture=True, camera_ready=ready_to_capture,
    )
    time.sleep(0.1)
    _write_robot_state(
        con, input_data, finish_capture=False, camera_ready=ready_to_capture,
    )
    _save_zdf_and_pose(save_dir, image_num, frame, transform)
    print("Image and pose saved")


def _generate_dataset(con: rtde, input_data):
    """ Generate dataset based on predefined robot poses

    Args:
        con: Connection between computer and robot
        input_data: Input package containing the specific input data registers

    Returns:
        Directory to where dataset is saved
    """

    with zivid.Application() as app:
        with app.connect_camera() as cam:

            _set_camera_settings(cam)
            ready_to_capture = True
            save_dir = _generate_folder()

            # Signal robot that camera is ready to capture
            _write_robot_state(
                con, input_data, finish_capture=False, camera_ready=ready_to_capture
            )

            robot_state = _read_robot_state(con)

            print(
                "Initial output robot_states: \n"
                f"Image count: {_image_count(robot_state)} \n"
                f"Start capture: {_start_capture(robot_state)}\n"
            )

            images_captured = 1
            while _image_count(robot_state) != -1:
                robot_state = _read_robot_state(con)

                if _start_capture(robot_state) and images_captured == _image_count(
                    robot_state
                ):
                    print(f"Capture image {_image_count(robot_state)}")
                    _capture_one_frame_and_robot_pose(
                        con,
                        cam,
                        save_dir,
                        input_data,
                        images_captured,
                        ready_to_capture,
                    )
                    images_captured += 1

                time.sleep(0.1)

    _write_robot_state(con, input_data, finish_capture=False, camera_ready=False)
    time.sleep(1.0)
    con.send_pause()
    con.disconnect()

    print(f"Data saved to: {save_dir}")

    return save_dir


def perform_hand_eye_calibration(mode: str, data_dir: Path):
    """ Perform had-eye calibration based on mode

    Args:
        mode: Calibration mode, eye-in-hand or eye-to-hand
        data_dir: Path to dataset

    Returns:
        4x4 transformation matrix
        List of residuals

    Raises:
        RuntimeError: If no feature points are detected
        ValueError: If calibration mode is invalid
    """
    # setup zivid
    app = zivid.Application()

    calibration_inputs = []
    idata = 1
    while True:
        frame_file = data_dir / f"img{idata:02d}.zdf"
        pose_file = data_dir / f"pos{idata:02d}.yaml"

        if frame_file.is_file() and pose_file.is_file():

            print(f"Detect feature points from img{idata:02d}.zdf")
            point_cloud = zivid.Frame(frame_file).get_point_cloud()
            detected_features = zivid.handeye.detect_feature_points(point_cloud)

            if not detected_features:
                raise RuntimeError(
                    f"Failed to detect feature points from frame {frame_file}"
                )

            print(f"Read robot pose from pos{idata:02d}.yaml")
            with open(pose_file) as file:
                pose = pose_from_datastring(file.read())

            detection_result = zivid.handeye.CalibrationInput(pose, detected_features)
            calibration_inputs.append(detection_result)
        else:
            break

        idata += 1

    print(f"\nPerform {mode} calibration")

    if mode == "eye-in-hand":
        calibration_result = zivid.handeye.calibrate_eye_in_hand(calibration_inputs)
    elif mode == "eye-to-hand":
        calibration_result = zivid.handeye.calibrate_eye_to_hand(calibration_inputs)
    else:
        raise ValueError(f"Invalid calibration mode: {mode}")

    transform = calibration_result.hand_eye_transform
    residuals = calibration_result.per_pose_calibration_residuals

    print("\n\nTransform: \n")
    np.set_printoptions(precision=5, suppress=True)
    print(transform)

    print("\n\nResiduals: \n")
    for num, res in enumerate(residuals, start=1):
        print(
            f"pose: {num:02d}   Rotation: {res.rotation:.6f}   Translation: {res.translation:.6f}"
        )

    return transform, residuals


def _main():

    host = "192.168.1.246"
    port = 30004
    con, input_data = _initialize_robot_sync(host, port)
    con.send_start()

    dataset_dir = _generate_dataset(con, input_data)

    print("\n Starting hand-eye calibration \n")
    # mode [eye-in-hand, eye-to-hand]
    mode = "eye-in-hand"
    transform, residuals = perform_hand_eye_calibration(mode, dataset_dir)
    _save_hand_eye_results(dataset_dir, transform, residuals)


if __name__ == "__main__":
    _main()
