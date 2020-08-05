'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import logging


class Model:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', threshold=0.5, is_visual=False, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'

        self.device = device
        self.threshold = threshold
        self.is_visual = is_visual
        self.extensions = extensions

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        try:
            # giving depcreated error
            # self.model = IENetwork(self.model_structure, self.model_weights)
            self.core = IECore()
            self.network = self.core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape

        self.output_name = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_name].shape

        layers_supported = self.core.query_network(self.network, device_name=self.device)
        layers = self.network.layers.keys()
        if not all(l in layers_supported for l in layers):
            self.core.add_extension(self.extensions, self.device)

        self.exec_net = self.core.load_network(self.network, self.device)

    def set_inputsize(self, initial_w, initial_h):
        self.initial_w = initial_w
        self.initial_h = initial_h

    def get_model_input_shape(self):
        self.n, self.c, self.model_h, self.model_w = self.input_shape
        return self.n, self.c, self.model_h, self.model_w

    def predict(self, *input_values):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        input_keys = self.network.inputs.keys()
        input = {key: value for key, value in zip(input_keys, input_values)}

        infer_request_handle = self.exec_net.start_async(0, inputs=input)
        status = infer_request_handle.wait()
        if status == 0:
            outputs = infer_request_handle.outputs

        return outputs

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        n, c, h, w = self.get_model_input_shape()

        preproc_image = cv2.resize(image, (w, h), cv2.INTER_AREA)
        preproc_image = preproc_image.transpose((2, 0, 1))
        preproc_image = preproc_image.reshape(n, c, h, w)
        return preproc_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
