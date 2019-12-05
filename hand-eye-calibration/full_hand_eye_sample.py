"""
Script to set up communication, generate dataset and perform hand-eye calibration.
"""
from pathlib import Path
# mport sys
import time
import datetime
import zivid

import cv2
import numpy as np
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config


def quat2rotm(quat: np.array) -> np.array:
    """Convert from quaternion to rotation matrix

    Args:
        quat: Rotations in quaternions

    Return:
        rotm: Rotations as 3x3 rotation matrix
    """
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    rotm = np.zeros((3, 3))
    rotm[0, 0] = 1 - 2 * np.power(qy, 2) - 2 * np.power(qz, 2)
    rotm[0, 1] = 2 * qx * qy - 2 * qz * qw
    rotm[0, 2] = 2 * qx * qz + 2 * qy * qw
    rotm[1, 0] = 2 * qx * qy + 2 * qz * qw
    rotm[1, 1] = 1 - 2 * np.power(qx, 2) - 2 * np.power(qz, 2)
    rotm[1, 2] = 2 * qy * qz - 2 * qx * qw
    rotm[2, 0] = 2 * qx * qz - 2 * qy * qw
    rotm[2, 1] = 2 * qy * qz + 2 * qx * qw
    rotm[2, 2] = 1 - 2 * np.power(qx, 2) - 2 * np.power(qy, 2)

    return rotm


def rotvec2quat(rotvec: np.array, angle: float = np.nan) -> np.array:
    if np.isnan(angle):
        angle = np.linalg.norm(rotvec)

    quat = np.zeros((4))
    quat[0] = np.cos(0.5 * angle)
    a = np.sqrt((1 - np.power(quat[0], 2)) / np.power(np.linalg.norm(rotvec), 2))
    quat[1] = a * rotvec[0]
    quat[2] = a * rotvec[1]
    quat[3] = a * rotvec[2]
    quat = quat / np.sign(quat[0])

    return quat


def _calculate_transform_matrix(robot_pose_before: np.array, robot_pose_after: np.array) -> np.array:
    """ Compute transform matrix based on poses

    Args:
        robot_pose_before: Pose of robot taken right before image was captured
        robot_pose_after: Pose of robot right after image was captured

    Returns:
        transform: 4x4 Transform matrix


    """
    pose = (robot_pose_before + robot_pose_after) / 2
    translation = pose[:3] * 1000
    rotvec = pose[3:]
    tansform = np.eye(4)
    tansform[:3, :3] = quat2rotm(rotvec2quat(rotvec))
    tansform[:3, 3] = translation.T
    return tansform


def _write_robot_state(con: rtde, input_data, finish_capture: bool = False, camera_ready: bool = False) -> None:
    """ Write to robot I/O registrer

    Args:
        con: connection between computer and robot
        input_data: input package containing the specific data registers
        finish_capture: Boolean value to robot_state that a scene capture is finished
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
    """Set up communication with UR robot.

    Args:
        host: IP address to host

    Return:
        con: connection to robot
        input_data: recipe with input data to robot
    """
    conf = rtde_config.ConfigFile(Path(Path.cwd() / 'robot_communication_file.xml'))
    output_names, output_types = conf.get_recipe('out')
    input_names, input_types = conf.get_recipe('in')

    # Rtde communication is by default port 30004.
    con = rtde.RTDE(host, port)
    con.connect()

    # To ensure that the application and further versions of UR controller is compatible
    if not con.negotiate_protocol_version():
        raise Exception(f"Protocol do not match.")

    if not con.send_output_setup(output_names, output_types, frequency=200):
        raise Exception(f"Unable to configure output")

    input_data = con.send_input_setup(input_names, input_types)

    if not con.send_start():
        raise Exception(f"Unable to start synchronization")

    print("Communication initialization completed. \n")

    return con, input_data


def _save_zdf_and_pose(save_dir: Path, image_num: int, frame: zivid.Frame, transform: np.array) -> None:
    """ Save data to folder

    Args:
        save_dir: Directory to save data
        image_num: Image number
        frame: Point cloud stored as .zdf
        transform: 4x4 transformation matrix
    """
    frame.save(save_dir / f"img{image_num:02d}.zdf")

    file_storage = cv2.FileStorage(str(save_dir / f"pos{image_num:02d}.yaml"), cv2.FILE_STORAGE_WRITE)
    file_storage.write("PoseState", transform)
    file_storage.release()


def _generate_folder() -> Path:
    """ Generate folder where dataset weill be stored

    Return:
        location_dir: The directory to save data
    """
    location_dir = Path(Path.cwd() / 'datasets' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not location_dir.is_dir():
        location_dir.mkdir(parents=True)

    return location_dir


def _capture_and_get_robot_pose(con: rtde, cam: zivid.Camera) -> (zivid.Frame, int, np.array, np.array):
    """ Capture image with Zivid camera and read robot pose.

    Args:
        con: Connection to robot arm
        cam: Zivid camera

    Return:
        frame: Point cloud stored as .zdf
        image_number: Image number
        robot_pose_before: Pose of robot taken right before image was captured
        robot_pose_after: Pose of robot right after image was captured
    """
    robot_state = con.receive()
    image_num = robot_state.output_int_register_24
    robot_pose_before = np.array(robot_state.actual_TCP_pose)
    frame = cam.capture()
    robot_pose_after = np.array(con.receive().actual_TCP_pose)

    return frame, image_num, robot_pose_before, robot_pose_after


def _set_camera_settings(cam: zivid.Camera) -> None:
    """ Set camera settings

    Args:
        cam: Zivid camera
    """
    settings = zivid.Settings()
    settings.iris = 20
    settings.exposure_time = datetime.timedelta(microseconds=10000)
    settings.gain = 1
    settings.brightness = 1.8
    settings.filters.gaussian.enabled = 1
    cam.settings = settings


def _read_robot_state(con: rtde) -> (int, np.array):
    """ Recieve robot output recipe

    Args:
        con: Connection to robot arm

    Return:
        image_number: image number
        robot_pose: pose of robot
    """
    robot_state = con.receive()

    image_number = robot_state.output_int_register_24
    robot_pose = robot_state.output_int_register_24

    return image_number, robot_pose


def pose_from_datastring(datastring: str) -> zivid.handeye.Pose:
    """Extract pose from a -yaml file saved by openCV

    Args:
        datastring: String of text from .yaml file

    Return:
        pose_matrix: Robotic pose as a zivid Pose class
    """
    s = datastring.split("data:")[-1].strip().strip("[").strip("]")
    pose_matrix = np.fromstring(
        s, dtype=np.float, count=16, sep=",").reshape((4, 4))
    return zivid.handeye.Pose(pose_matrix)


def _save_hand_eye_results(save_dir: Path, transform: np.array, residuals: list) -> None:
    file_storage_transform = cv2.FileStorage(str(save_dir / f"transformation.yaml"), cv2.FILE_STORAGE_WRITE)
    file_storage_transform.write("PoseState", transform)
    file_storage_transform.release()

    file_storage_residuals = cv2.FileStorage(str(save_dir / f"residuals.yaml"), cv2.FILE_STORAGE_WRITE)
    residual_list = []
    for residual in residuals:
        tmp = list([residual.translation, residual.translation])
        residual_list.append(tmp)

    file_storage_residuals.write('Per pose residuals for rotation in deg and translation in mm', np.array(residual_list))
    file_storage_residuals.release()


def perform_hand_eye_calibration(mode: str, data_dir: Path) -> (np.array, list):
    """ Perform had-eye calibration based on mode

    Args:
        mode: Calibration mode, eye-in-hand or eye-to-hand
        data_dir: Path to dataset

    Return:
        transform: 4x4 transformation matrix
        residual: list of residuals
    """
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
                raise RuntimeError(f"Failed to detect feature-points from frame {frame_file}")

            print(f"Read robot pose from pos{idata:02d}.yaml")
            with open(pose_file) as f:
                pose = pose_from_datastring(f.read())

            detection_result = zivid.handeye.CalibrationInput(pose, detected_features)
            calibration_inputs.append(detection_result)
        else:
            break

        idata += 1

    print(f"\nPerform {mode} calibration")
    if mode == "eye-in-hand":
        calibration_result = zivid.handeye.calibrate_eye_in_hand(
            calibration_inputs)
    elif mode == "eye-to-hand":
        calibration_result = zivid.handeye.calibrate_eye_to_hand(
            calibration_inputs)
    else:
        raise ValueError("Invalid calibration mode: " + mode)

    transform = calibration_result.hand_eye_transform
    residuals = calibration_result.per_pose_calibration_residuals

    print("\n\nTransform: \n")
    np.set_printoptions(precision=5, suppress=True)
    print(transform)

    print("\n\nResiduals: \n")
    for num, residual in enumerate(residuals, start=1):
        print(
            f"pose: {num:02d}     Rotation[deg]: {residual.rotation:.6f}     Translation[mm]: {residual.translation:.6f}")

    return transform, residuals


def _generate_dataset(con: rtde, input_data) -> Path:
    """ Generate dataset based on predefined robot poses

    Args:
        host: IP address of robot

    Signals from UR robot:
    output_int_register_24 = image_count: Counter from robot side of number of images and poses taken.
    output_bit_register_64 = start_capture: Bool that trigger camera to capture an image

    Signals to UR robot:
    input_bit_register_64 = finish_capture: Bool that trigger UR robot to move to next position.
    input_bit_register_65 = ready_to_capture: bool that tells UR robot that the camera is ready to be used.
    """
    with zivid.Application() as app:
        with app.connect_camera() as cam:

            _set_camera_settings(cam)
            ready_to_capture = True

            save_dir = _generate_folder()

            # Signal robot that camera is ready to capture
            _write_robot_state(con, input_data, finish_capture=False, camera_ready=ready_to_capture)
            robot_state = con.receive()

            print(
                "Initial output robot_states: \n"
                f"Image count: {robot_state.output_int_register_24} \n"
                f"Start capture: {robot_state.output_bit_register_64}"
            )

            images_captured = 1
            while robot_state.output_int_register_24 != -1:

                robot_state = con.receive()

                if robot_state is None:
                    print("Not able to recieve robot_state")
                    con.disconnect()
                    break

                if robot_state.output_bit_register_64 and images_captured == robot_state.output_int_register_24:
                    images_captured += 1

                    print(f"[CAMERA] Capture image {robot_state.output_int_register_24}")
                    frame, image_num, robot_pose_before, robot_pose_after = _capture_and_get_robot_pose(con, cam)
                    transform = _calculate_transform_matrix(robot_pose_before, robot_pose_after)

                    # Signal robot to move to next position, then set to low again.
                    _write_robot_state(con, input_data, finish_capture=True, camera_ready=ready_to_capture)
                    time.sleep(0.1)
                    _write_robot_state(con, input_data, finish_capture=False, camera_ready=ready_to_capture)

                    # Using patlib to save images
                    _save_zdf_and_pose(save_dir, image_num, frame, transform)
                    print("Image and pose saved")

                time.sleep(0.1)

    _write_robot_state(con, input_data, finish_capture=False, camera_ready=False)
    time.sleep(1.0)
    con.send_pause()
    con.disconnect()

    print(f"Data saved to: {save_dir}")

    return save_dir


def main():
    host = '192.168.1.246'
    port = 30004
    con, input_data = _initialize_robot_sync(host, port)
    con.send_start()

    dataset_dir = _generate_dataset(con, input_data)

    print("\n Starting hand-eye calibration \n")
    mode = 'eye-in-hand'
    save_dir = Path(Path().cwd() / 'datasets' / mode)
    transform, residuals = perform_hand_eye_calibration(mode, save_dir)
    _save_hand_eye_results(save_dir, transform, residuals)


if __name__ == "__main__":
    main()
