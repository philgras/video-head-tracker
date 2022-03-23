from vht.model.tracking import FlameTracker as Tracker
from vht.data.video import VideoDataset
from vht.util.log import get_logger
from vht.util.video_to_dataset import Video2DatasetConverter

from argparse import ArgumentParser
from configargparse import ArgumentParser as ConfigArgumentParser
from pathlib import Path

logger = get_logger("vht", root=True)


def main():
    parser = ArgumentParser()
    parser = Tracker.add_argparse_args(parser)

    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument("--config", required=True, is_config_file=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video", required=False)
    parser.add_argument("--tracking_resolution", type=int, nargs=2, required=True)

    args = parser.parse_args()
    args_dict = vars(args)

    data_path = Path(args.data_path)
    if not data_path.exists() or args.video is not None:
        converter = Video2DatasetConverter(args.video, args.data_path)
        converter.extract_frames()
        converter.annotate_landmarks()

    logger.info(
        f"Start tracking with the following"
        f" configuration: \n {parser.format_values()}"
    )

    data = VideoDataset(args.data_path, args.tracking_resolution)
    tracker = Tracker(data, **args_dict)
    tracker.optimize()


if __name__ == "__main__":
    main()
