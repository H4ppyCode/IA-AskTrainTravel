# Imports
from typing import Union
import types
import os
import json

class Model:
    def __init__(self, model_name: str = None):
        self.model_name: str = None
        self.dataset_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nlpDataSet')
        self.test_data_filename: str = 'test.json'
        self._test_data = None
        self.train_data_filename: str = 'train.json'
        self._train_data = None
        self.validation_data_filename: str = 'validation.json'
        self._validation_data = None

        if model_name:
            self.load_model(model_name)

    def refresh_methods(self):
        # Rebind all methods from the class to the instance
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name)
            if callable(attr) and not attr_name.startswith("__"):
                setattr(self, attr_name, types.MethodType(attr, self))

    @property
    def test_data_path(self):
        return os.path.join(self.dataset_path, self.test_data_filename)

    @property
    def test_data(self):
        if self._test_data is None:
            self._test_data = self.load_data(self.test_data_path)
        return self._test_data
    
    @property
    def train_data_path(self):
        return os.path.join(self.dataset_path, self.train_data_filename)

    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = self.load_data(self.train_data_path)
        return self._train_data
    
    @property
    def validation_data_path(self):
        return os.path.join(self.dataset_path, self.validation_data_filename)

    @property
    def validation_data(self):
        if self._validation_data is None:
            self._validation_data = self.load_data(self.validation_data_path)
        return self._validation_data

    def load_data(self, data_path: str):
        with open(data_path, "r") as file:
            data = json.load(file)
        prepared = self.prepare_data(data)
        return prepared

    def prepare_data(self, data):
        return data

    def get_sentence(self, data):
        return data[0]

    def __call__(self, *args, **kwds):
        if not hasattr(self, "model"):
            raise AttributeError("Model attribute not found")
        return self.model(*args, **kwds)

    def load_model(self, model_name: str):
        raise NotImplementedError("load_model method not implemented")

    def evaluate_model(self, dataset):
        raise NotImplementedError("evaluate_model method not implemented")
