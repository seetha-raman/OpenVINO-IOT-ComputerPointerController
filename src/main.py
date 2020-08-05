import click
import logging
from types import SimpleNamespace

from contextlib import contextmanager
from timeit import default_timer

import cv2
from PIL import Image
import numpy as np
import pathlib

from input_feeder import InputFeeder
from mouse_controller import MouseController
from perf_metrics import MetricsBuilder

from face_detection_model import FaceDetectionModel
from landmark_detection_model import LandmarkDetectionModel
from head_pose_estimation_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    # skipping exit logic


@click.command()
@click.option('-fdm', '--fdmodel', help='Face Detection Model')
@click.option('-ldm', '--ldmodel', help='Landmark Detection Model')
@click.option('-hpem', '--hpemodel', help='Head Pose Estimation Model')
@click.option('-gem', '--gemodel', help='Gaze Estimation Model')
@click.option('-d', '--device', default='CPU', required=False,
              type=click.Choice(['CPU', 'GPU', 'MYRIAD', 'FPGA'], case_sensitive=False), help='Device')
@click.option("-pr", "--precision", required=False, default='FP32', help="model precision")
@click.option('-ext', '--extension', required=False, help='Extensions')
@click.option('-p', '--prob_threshold', default=0.5, required=False, help='Probability Threshold')
@click.option('-c', '--is_cam', default=False, required=False, is_flag=True, help='Enable camera for input')
@click.option('-i', '--input', required=False, type=click.Path(exists=True), help='Input File path')
@click.option('-v', '--is_visual', default=False, required=False, is_flag=True,
              help='Draw intermediate result and make more visual or not')
@click.option('-mp', '--is_move_pointer', default=False, required=False, is_flag=True, help='Move mouse pointer or not')
@click.option('-sf', '--is_show_frame', default=False, required=False, is_flag=True, help='Show frame or not')
def cli(**kwargs):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("computer-pointer-controller.log")
        ]
    )

    options = SimpleNamespace(**kwargs)
    logging.debug(options)

    run_app(options)


def run_app(options):
    metrics_builder = MetricsBuilder(options.precision)

    with elapsed_timer() as et:
        fdmodel = FaceDetectionModel(options.fdmodel, options.device, options.prob_threshold, options.is_visual,
                                     options.extension)
        fdmodel.load_model()
        fdmodel_loadtime = et()
        metrics_builder.face_detection.load_time = fdmodel_loadtime
        logging.info(f'face detection loading time taken: {fdmodel_loadtime}')

    with elapsed_timer() as et:
        ldmodel = LandmarkDetectionModel(options.ldmodel, options.device, options.prob_threshold, options.is_visual,
                                         options.extension)
        ldmodel.load_model()
        ldmodel_loadtime = et()
        metrics_builder.landmarks_detection.load_time = ldmodel_loadtime
        logging.info(f'Landmark detection loading time taken: {ldmodel_loadtime}')

    with elapsed_timer() as et:
        hpemodel = HeadPoseEstimationModel(options.hpemodel, options.device, options.prob_threshold, options.is_visual,
                                           options.extension)
        hpemodel.load_model()
        hpemodel_loadtime = et()
        metrics_builder.head_pose_estimation.load_time = hpemodel_loadtime
        logging.info(f'Head Position Estimation loading time taken: {hpemodel_loadtime}')

    with elapsed_timer() as et:
        gemodel = GazeEstimationModel(options.gemodel, options.device, options.prob_threshold, options.is_visual,
                                      options.extension)
        gemodel.load_model()
        gemodel_loadtime = et()
        metrics_builder.gaze_estimation.load_time = gemodel_loadtime
        logging.info(f'Gazer Estimation loading time taken: {gemodel_loadtime}')

    try:

        # Get and open video capture
        if options.is_cam:
            feeder = InputFeeder('cam')
        else:
            feeder = InputFeeder('video', options.input)
        feeder.load_data()

        initial_w, initial_h = feeder.get_size()
        fps = feeder.get_fps()

        fdmodel.set_inputsize(initial_w, initial_h)
        ldmodel.set_inputsize(initial_w, initial_h)
        hpemodel.set_inputsize(initial_w, initial_h)
        gemodel.set_inputsize(initial_w, initial_h)

        frame_count = 0

        mouse_controller = MouseController("low", "fast")

        window_name = 'computer pointer controller'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, initial_w, initial_h)

        out_path = str(pathlib.Path('./results/output_video.mp4'))
        print(out_path)
        out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

        for frame in feeder.next_batch():

            if frame is None:
                break

            # exit video for escape key
            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break

            frame_count += 1

            # detect face
            p_frame = fdmodel.preprocess_input(frame)
            with elapsed_timer() as et:
                fdmodel_output = fdmodel.predict(p_frame)
                metrics_builder.face_detection.add_infer_time(et())
            out_frame, fboxes = fdmodel.preprocess_output(fdmodel_output, frame)

            # Take first face - (xmin,ymin,xmax,ymax)
            fbox = fboxes[0]

            # landmarks estimation
            # face = np.asarray(Image.fromarray(frame).crop(fbox))
            xmin, ymin, xmax, ymax = fbox
            face = frame[ymin:ymax, xmin:xmax]

            p_frame = ldmodel.preprocess_input(face)
            with elapsed_timer() as et:
                lmoutput = ldmodel.predict(p_frame)
                metrics_builder.landmarks_detection.add_infer_time(et())
            out_frame, left_eye_point, right_eye_point = ldmodel.preprocess_output(lmoutput, fbox, out_frame)

            # head pose estimation
            p_frame = hpemodel.preprocess_input(face)
            with elapsed_timer() as et:
                hpoutput = hpemodel.predict(p_frame)
                metrics_builder.head_pose_estimation.add_infer_time(et())
            out_frame, headpose_angels = hpemodel.preprocess_output(hpoutput, out_frame, face, fbox)
            #
            # # gaze  estimation
            out_frame, left_eye, right_eye = gemodel.preprocess_input(out_frame, face, left_eye_point,
                                                                      right_eye_point)
            with elapsed_timer() as et:
                geoutput = gemodel.predict(headpose_angels, left_eye, right_eye)
                metrics_builder.gaze_estimation.add_infer_time(et())
            out_frame, gazevector = gemodel.preprocess_output(geoutput, out_frame, fbox, left_eye_point,
                                                              right_eye_point)
            # show frame
            if options.is_show_frame:
                cv2.imshow(window_name, out_frame)

            # mouse controller
            if options.is_move_pointer:
                x, y, _ = gazevector
                mouse_controller.move(x, y)

            out_video.write(out_frame)

        # performance metrics
        metrics_builder.save_metrics(frame_count)

        feeder.close()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error("Fatal error in main loop", exc_info=True)


if __name__ == '__main__':
    cli()
