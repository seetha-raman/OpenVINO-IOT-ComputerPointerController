'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import Model
import cv2


class FaceDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        https://docs.openvinotoolkit.org/2020.3/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
        outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection,
        the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        '''

        outputs = outputs[self.output_name]

        coords = []
        probs = outputs[0, 0, :, 2]
        for i, p in enumerate(probs):
            if p > self.threshold:
                box = outputs[0, 0, i, 3:]
                xymin = (int(box[0] * self.initial_w), int(box[1] * self.initial_h))
                xymax = (int(box[2] * self.initial_w), int(box[3] * self.initial_h))
                coords.append((*xymin, *xymax))
                if self.is_visual:
                    self.draw_outputs(image, xymin, xymax)
        return image, coords

    def draw_outputs(self, image, xymin, xymax):
        '''
        TODO: This method needs to be completed by you
        '''
        face_text = f'Face: xymin: {xymin}, xymax: {xymax}'
        cv2.putText(image, face_text, (20, 20), 0, 0.6, (255, 255, 0))

        cv2.rectangle(image, xymin, xymax, (0, 255, 0), 2)
