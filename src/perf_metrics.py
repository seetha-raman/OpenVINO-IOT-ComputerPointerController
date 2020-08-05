from dataclasses import dataclass
import pandas as pd
import pathlib


@dataclass
class Metrics:
    _load_time: float = 0.0
    infer_time: float = 0.0

    @property
    def load_time(self):
        return self._load_time * 1000

    @load_time.setter
    def load_time(self, load_time):
        self._load_time = load_time

    def add_infer_time(self, infer_time):
        self.infer_time += infer_time

    def avg_infer_time(self, total_frame):
        return (self.infer_time / total_frame) * 1000


class MetricsBuilder:
    FACE_DETECTION = 'face_detection'
    LANDMARKS_DETECTION = 'landmarks_detection'
    HEAD_POSE_ESTIMATION = 'head_pose_estimation'
    GAZE_ESTIMATION = 'gaze_estimation'

    def __init__(self, precision):
        self.precision = precision

        self.face_detection = Metrics()
        self.landmarks_detection = Metrics()
        self.head_pose_estimation = Metrics()
        self.gaze_estimation = Metrics()

    def save_metrics(self, total_frame):
        index = [MetricsBuilder.FACE_DETECTION, MetricsBuilder.LANDMARKS_DETECTION,
                 MetricsBuilder.HEAD_POSE_ESTIMATION, MetricsBuilder.GAZE_ESTIMATION]

        def write_df(df, metrics_type):
            output_file = f'{self.precision}.csv'
            output_dir = pathlib.Path(f'./results/{metrics_type}')
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / output_file)

        data = [{self.precision: self.face_detection.load_time},
                {self.precision: self.landmarks_detection.load_time},
                {self.precision: self.head_pose_estimation.load_time},
                {self.precision: self.gaze_estimation.load_time}]

        df = pd.DataFrame(data, index=index)
        write_df(df, 'loadtime')

        data = [{self.precision: self.face_detection.avg_infer_time(total_frame)},
                {self.precision: self.landmarks_detection.avg_infer_time(total_frame)},
                {self.precision: self.head_pose_estimation.avg_infer_time(total_frame)},
                {self.precision: self.gaze_estimation.avg_infer_time(total_frame)}]

        df = pd.DataFrame(data, index=index)
        write_df(df, 'infertime')
