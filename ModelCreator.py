from PIL import Image
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
import torch
from glob import glob
import pickle


class Model:
    def __init__(self):
        self.classy = Pipeline([('scaler', MinMaxScaler()),
                                ('pca', PCA(n_components=.9)),
                                ('SVC', SVC())])
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.model = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def _vectorize(self, images):
        images = [images] if not isinstance(images, list) else images
        images = [i.convert('RGB') for i in images]
        print(f'vectorizing {len(images)} images')
        inputs = self.image_processor(images, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        print(f'  vectorization finished')
        return outputs.pooler_output

    def train(self, images_train, labels_train):
        print('Starting to fit')
        X_train = self._vectorize(images_train)
        y_train = labels_train
        self.classy.fit(X_train, y_train)
        print('  Correctly fitted')

    def predict(self, images):
        X = self._vectorize(images)
        return self.classy.predict(X)

    def __call__(self, images):
        return self.predict(images)

    def export_model(self, model_name=f'model.pickle'):
        # Save
        with open(f'model.pickle', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(model_name=f'model.pickle'):
        with open(f'model.pickle', 'rb') as file2:
            m = pickle.load(file2)
        return m


"""
install requirements 

pip install -r requirements.txt

Example of use:

from ModelCreator import Model

model_path = 'model.pickle'
m2 = Model().load_model(model_path)

image_path = 'some_image_path/image_1.png'
result = m2(image_path)
print(result)

"""
