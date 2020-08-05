'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import Model
import cv2


class LandmarkDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_output(self, outputs, facebox, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_attributes_landmarks_regression_0009_onnx_desc_landmarks_regression_retail_0009.html
        outputs shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks
        coordinates (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1]
        '''

        outputs = outputs[self.output_name]

        xmin, ymin, xmax, ymax = facebox
        width, height = xmax - xmin, ymax - ymin

        landmarks_output = outputs[0]
        x_lefteye, y_lefteye = landmarks_output[0], landmarks_output[1]
        x_righteye, y_righteye = landmarks_output[2], landmarks_output[3]

        left_eye_point = [int(x_lefteye * width), int(y_lefteye * height)]
        right_eye_point = [int(x_righteye * width), int(y_righteye * height)]

        # Drawing Eye circle
        if self.is_visual:
            self.draw_outputs(image, (xmin, ymin), left_eye_point, right_eye_point)

        return image, left_eye_point, right_eye_point

    def draw_outputs(self, image, xymin, left_eye_point, right_eye_point):
        '''
        TODO: This method needs to be completed by you
        '''
        landmarks_text = f'FaceLandmarks: left-eye: {left_eye_point}, right-eye: {right_eye_point}'
        cv2.putText(image, landmarks_text, (20, 60), 0, 0.6, (255, 255, 0))

        cv2.circle(image, (xymin[0] + left_eye_point[0], xymin[1] + left_eye_point[1]), 30, (0, 255, 255), 2)
        cv2.circle(image, (xymin[0] + right_eye_point[0], xymin[1] + right_eye_point[1]), 30, (0, 255, 255), 2)
