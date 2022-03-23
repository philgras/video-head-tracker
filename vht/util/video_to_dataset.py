from vht.util.log import get_logger
from vht.data.video import frame2id

from pathlib import Path
from PIL import Image

import cv2
import face_alignment
import numpy as np
import json
import matplotlib.path as mpltPath

from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks,
)

# setup logger
logger = get_logger(__name__)


class Video2DatasetConverter:

    IMAGE_FILE_NAME = "image_0000.png"
    LMK_FILE_NAME = "keypoints_static_0000.json"

    def __init__(
        self,
        video_path,
        dataset_path,
    ):
        """
        Creates dataset_path where all results are stored
        :param video_path: path to video file
        :param dataset_path: path to results directory
        """
        self._video_path = Path(video_path)
        self._data_path = Path(dataset_path)
        self._no_iris_landmarks = [-1] * 6

        assert self._video_path.exists()
        self._data_path.mkdir(parents=True, exist_ok=True)

    def extract_frames(self):
        """
        Unpacks every frame of the video into a separate folder dataset_path/frame_xxx/image_0.png
        :return:
        """
        cap = cv2.VideoCapture(str(self._video_path))
        count = 1

        logger.info("Extracting all frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.info(f"Extracting frame {count:04d}")

                frame_dir = self._data_path / f"frame_{count:04d}"
                frame_dir.mkdir(exist_ok=True)

                img_file = frame_dir / Video2DatasetConverter.IMAGE_FILE_NAME
                cv2.imwrite(str(img_file), frame)
                count = count + 1

                # if count == 10:
                #   break
            else:
                break
        cap.release()

    def _get_frame_list(self):
        """
        Creates sorted list of paths to frames
        :return: list of frame file paths
        """
        frame_paths = []
        for frame_dir in self._data_path.iterdir():
            if "frame" in frame_dir.name and frame_dir.is_dir():
                for file in frame_dir.iterdir():
                    if (
                        Video2DatasetConverter.IMAGE_FILE_NAME == file.name
                        and file.is_file()
                    ):
                        frame_paths.append(file)
                        break

        frame_paths = sorted(frame_paths, key=lambda k: frame2id(k.parent.name))
        return frame_paths

    def _get_frame_gen(self):
        """
        Creates a Python generator to ease iteration over all frame paths
        :return: generator
        """

        # generator function that iterates over all frames
        def frame_generator():
            for frame_dir in self._data_path.iterdir():
                if "frame" in frame_dir.name and frame_dir.is_dir():
                    for file in frame_dir.iterdir():
                        if (
                            Video2DatasetConverter.IMAGE_FILE_NAME == file.name
                            and file.is_file()
                        ):
                            yield file
                            break

        return frame_generator

    def _annotate_facial_landmarks(self):
        """
        Annotates each frame with 68 facial landmarks
        :return: dict mapping frame number to landmarks numpy array and the same thing for bboxes
        """
        # 68 facial landmark detector
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, flip_input=True, device="cuda"
        )
        frames = self._get_frame_list()
        landmarks = {}
        bboxes = {}

        for frame in frames:
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Annotate facial landmarks for frame: {frame_id}")
            img = np.array(Image.open(frame))
            bbox = fa.face_detector.detect_from_image(img)

            if len(bbox) == 0:
                # if no faces detected, something is weird and
                # one shouldnt use the image
                raise RuntimeError(f"Error: No bounding box found for {frame}!")

            else:
                if len(bbox) > 1:
                    # if multiple boxes detected, use the one with highest confidence
                    bbox = [bbox[np.argmax(np.array(bbox)[:, -1])]]

                lmks = fa.get_landmarks_from_image(img, detected_faces=bbox)[0]

            landmarks[frame_id] = lmks
            bboxes[frame_id] = bbox[0]

        return landmarks, bboxes

    def _annotate_iris_landmarks(self):
        """
        Annotates each frame with 2 iris landmarks
        :return: dict mapping frame number to landmarks numpy array
        """

        # iris detector
        detect_faces = FaceDetection()
        detect_face_landmarks = FaceLandmark()
        detect_iris_landmarks = IrisLandmark()

        frames = self._get_frame_list()
        landmarks = {}

        for frame in frames:
            frame_id = int(frame.parent.name.split("_")[-1])
            logger.info(f"Annotate iris landmarks for frame: {frame_id}")

            img = Image.open(frame)

            width, height = img.size
            img_size = (width, height)
            lmks = self._no_iris_landmarks

            face_detections = detect_faces(img)
            if len(face_detections) != 1:
                logger.error("Empty iris landmarks")
            else:
                for face_detection in face_detections:
                    try:
                        face_roi = face_detection_to_roi(face_detection, img_size)
                    except ValueError:
                        logger.error("Empty iris landmarks")
                        break

                    face_landmarks = detect_face_landmarks(img, face_roi)
                    if len(face_landmarks) == 0:
                        logger.error("Empty iris landmarks")
                        break

                    iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)

                    if len(iris_rois) != 2:
                        logger.error("Empty iris landmarks")
                        break

                    lmks = []
                    for iris_roi in iris_rois[::-1]:
                        try:
                            iris_landmarks = detect_iris_landmarks(img, iris_roi).iris[
                                0:1
                            ]
                        except np.linalg.LinAlgError:
                            logger.error("Failed to get iris landmarks")
                            break

                        for landmark in iris_landmarks:
                            lmks.append(landmark.x * width)
                            lmks.append(landmark.y * height)
                            lmks.append(1.0)

                landmarks[frame_id] = np.array(lmks)

        return landmarks

    def _iris_consistency(self, lm_iris, lm_eye):
        """
        Checks if landmarks for eye and iris are consistent
        :param lm_iris:
        :param lm_eye:
        :return:
        """
        lm_iris = np.array(lm_iris).reshape(1, 3)[:, :2]
        lm_eye = np.array(lm_eye).reshape((-1, 3))[:, :2]

        polygon_eye = mpltPath.Path(lm_eye)
        valid = polygon_eye.contains_points(lm_iris)

        return valid[0]

    def annotate_landmarks(self, add_iris=True):
        """
        Annotates each frame with landmarks for face and iris. Assumes frames have been extracted
        :param add_iris:
        :return:
        """
        lmks_face, bboxes_faces = self._annotate_facial_landmarks()

        if add_iris:
            lmks_iris = self._annotate_iris_landmarks()

            # check conistency of iris landmarks and facial keypoints
            for k in lmks_face.keys():

                lmks_face_i = lmks_face[k].flatten().tolist()
                lmks_iris_i = lmks_iris[k]

                # validate iris landmarks
                left_face = lmks_face_i[36 * 3 : 42 * 3]
                right_face = lmks_face_i[42 * 3 : 48 * 3]

                right_iris = lmks_iris_i[:3]
                left_iris = lmks_iris_i[3:]

                if not (
                    self._iris_consistency(left_iris, left_face)
                    and self._iris_consistency(right_iris, right_face)
                ):
                    logger.warning(f"Inconsistent iris landmarks for frame {k}")
                    lmks_iris[k] = np.array(self._no_iris_landmarks)

        # construct final json
        for k in lmks_face.keys():
            lmk_dict = {}
            lmk_dict["bounding_box"] = bboxes_faces[k].tolist()
            lmk_dict["face_keypoints_2d"] = lmks_face[k].flatten().tolist()

            if add_iris:
                lmk_dict["iris_keypoints_2d"] = lmks_iris[k].flatten().tolist()

            json_dict = {"origin": "face-alignment", "people": [lmk_dict]}
            out_path = (
                self._data_path
                / f"frame_{k:04d}"
                / Video2DatasetConverter.LMK_FILE_NAME
            )

            with open(out_path, "w") as f:
                json.dump(json_dict, f)


def create_dataset(video, out_path):
    converter = Video2DatasetConverter(video, out_path)
    converter.extract_frames()
    converter.annotate_landmarks()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    create_dataset(args.video, args.out_path)
