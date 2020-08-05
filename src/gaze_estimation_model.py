'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import Model
import cv2


class GazeEstimationModel(Model):
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_input(self, frame, face, left_eye_point, right_eye_point):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        https://docs.openvinotoolkit.org/2020.3/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

        Blob in the format [BxCxHxW]
        with the name left_eye_image and the shape [1x3x60x60] - Blob in the format [BxCxHxW]
        with the name right_eye_image and the shape [1x3x60x60] - Blob in the format [BxC]
        with the name head_pose_angles and the shape [1x3]

        B - batch size
        C - number of channels
        H - image height
        W - image width
        '''

        eye_input_shape = [1, 3, 60, 60]

        def get_eye_frame(eye_point):
            # crop left eye
            x_center, y_center = eye_point
            height, width = eye_input_shape[2], eye_input_shape[3]

            face_height_shape, face_width_edge = face.shape[0], face.shape[1]

            xmin = int(x_center - width // 2)
            xmin = xmin if xmin > 0 else 0
            xmax = int(x_center + width // 2)
            xmax = xmax if xmax < face_width_edge else face_width_edge

            ymin = int(y_center - height // 2)
            ymin = ymin if ymin > 0 else 0
            ymax = int(y_center + height // 2)
            ymax = ymax if ymax < face_height_shape else face_height_shape

            eye_image = face[ymin: ymax, xmin:xmax]
            eye_frame = cv2.resize(eye_image, (width, height))
            eye_frame = eye_frame.transpose((2, 0, 1))
            eye_frame = eye_frame.reshape(1, *eye_frame.shape)
            return eye_frame, eye_image

        left_eye_frame, left_eye_image = get_eye_frame(left_eye_point)
        right_eye_frame, right_eye_image = get_eye_frame(right_eye_point)

        if self.is_visual:
            frame[150:150 + left_eye_image.shape[0], 20:20 + left_eye_image.shape[1]] = left_eye_image
            frame[150:150 + right_eye_image.shape[0], 100:100 + right_eye_image.shape[1]] = right_eye_image

        return frame, left_eye_frame, right_eye_frame

    def preprocess_output(self, outputs, image, facebox, left_eye_point, right_eye_point):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        https://docs.openvinotoolkit.org/2020.3/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector.
        Please note that the output vector is not normalizes and has non-unit length
        '''

        outputs = outputs[self.output_name]

        x, y, z = outputs[0][0], outputs[0][1], outputs[0][2]
        if self.is_visual:
            self.draw_outputs(image, facebox, left_eye_point, right_eye_point, x, y, z)

        return image, [x, y, z]

    def draw_outputs(self, image, facebox, left_eye_point, right_eye_point, x, y, z):
        '''
        TODO: This method needs to be completed by you
        '''
        gaze_text = f'Gaze: (x,y,z): ({x:.2f}, {y:.2f}, {z:.2f})'
        cv2.putText(image, gaze_text, (20, 130), 0, 0.6, (255, 255, 0))

        xmin, ymin, _, _ = facebox

        def draw_gaze_line(eye_point):
            x_center, y_center = eye_point
            eye_center_x = int(xmin + x_center)
            eye_center_y = int(ymin + y_center)

            cv2.arrowedLine(image, (eye_center_x, eye_center_y),
                            (eye_center_x + int(x * 100), eye_center_y + int(-y * 100)), (255, 100, 100), 5)

        draw_gaze_line(left_eye_point)
        draw_gaze_line(right_eye_point)
