# tfidf_vectorizer_model.py
import mlflow.pyfunc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
# import cloudpickle

class TfidfVectorizerModel(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer
        self.data_features = None
        self.train_data = None

    def fit(self, data):
        self.train_data = data
        self.data_features = self.vectorizer.fit_transform(data)

    def predict(self, context, model_input, params: dict):
        # Transform the model_input
        print(model_input)
        print(type(model_input))
        print(params)
        input_features = self.vectorizer.transform(model_input)
        print(input_features)
        print(type(input_features))
        number_of_recommendations = params["number_of_recommendations"]
        # Compute cosine similarity
        cosine_similarities = linear_kernel(input_features, self.data_features)
        print(type(cosine_similarities))
        # Get the indices of the most similar rows and their similarities
        top_similar_indices = np.argsort(-cosine_similarities, axis=1)[:, :number_of_recommendations]
        top_similarities = np.sort(-cosine_similarities, axis=1)[:, :number_of_recommendations]
        
        return top_similar_indices, -top_similarities
    
    # # Serialize and deserialize methods for the model
    # def save(self, path):
    #     with open(path, 'wb') as f:
    #         cloudpickle.dump(self, f)

    # @classmethod
    # def load(cls, path):
    #     with open(path, 'rb') as f:
    #         return cloudpickle.load(f)