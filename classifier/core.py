from collections import defaultdict
from math import log, sqrt

from utils import timed

import numpy as np
import scipy.sparse
from sklearn.base import ClassifierMixin
import scipy.linalg
import sklearn.preprocessing


class TopicClassifier(ClassifierMixin):

    """
    The core classifier class. This is designed to be close to the
    scikit.learn paradigms for classifiers.

    Each class is represented by the centroid of its training samples;
    test samples are classified to the class with the nearest centroid.

    The class labels have been already remapped to integers to
    integers corresponding to rows in the internal model matrix.

    The features have been already remapped to integers corresponding
    to columns in the internal model matrix.

    Attributes:
       models_: (list) centroids for each class indice
    """

    @timed
    def fit(self, X, y):
        """
        Fit the nearest centroid model according to the given training data
        
        Args:
          X: (scipy.sparse.csr_matrix) training data matrix of shape = [n_samples, n_features]
          where each row corresponds to a training sample 
          n_samples = total number of training samples (=len(data))
          n_features = vocabulary size or total number of features
          y : (list) class indices to be fit. 
        
        """
        subrows = defaultdict(list)
        for i in range(len(y)):
            # Collect indices of exemplars for the given class label
            subrows[y[i]].append(i)

        models = None
        for index, label in enumerate(subrows.keys()):
            # subselect an array of exemplars associated with
            # this label
            exemplars = X[subrows[label]]
            # compute centroid for exemplars
            centroid = self.centroid(exemplars)
            # we stack arrays row wise into models
            if models == None:
                models = scipy.sparse.csr_matrix(centroid)
            else:
                models = scipy.sparse.vstack([models, centroid])
        self.models_ = models
        return self

    def centroid(self, X):
        """
        Compute the normalized centroid for a set of training vectors X
        
        Args: 
          X: (scipy.sparse.csr_matrix) matrix of exemplars
          
        Returns:
          numpy centroid matrix
        """
        centroid = X.mean(axis=0)
        sklearn.preprocessing.normalize(centroid, axis=1, norm='l2')
        return centroid

    @timed
    def predict(self, X):
        """
        Predict class labels on an array of test vectors X

        Args: 
          X: (scipy.sparse.csr_matrix) matrix of test data.
             shape = [n_samples, n_features]
          
        Returns:
          (list) predicted class label for each test sample 

        Raises:
          Exception: No prediction vector computed
        """
        predictions = self.predict_with_scores(X)
        if not predictions:
            raise Exception("No prediction vector")
        result = []
        for prediction in predictions:
            # pick closest predicted class. list is
            # already sorted by descending score
            (score, label_index) = prediction[0]
            result.append(label_index)
        return result

    def predict_with_scores(self, X):
        """
        Compute the scores for each class label for test data X
        
        Args: 
          X: (scipy.sparse.csr_matrix) matrix of test data
             shape = [n_samples, n_features]
   
        Returns:
          (list) A list of (score, topic_id) pairs for each 
          test sample sorted by descending score

        Raises:
          AttributeError: No model has not been trained yet
        """
        if not hasattr(self, "models_"):
            raise AttributeError("Model has not been trained yet.")
        result = self._compute_scores(X, self.models_)
        return self._extract_prediction(result)

    def _compute_scores(self, X, centroid):
        """
        Compute the scores for each class label for test data X
        
        Args: 
          X: (scipy.sparse.csr_matrix) matrix of test data
             shape = [n_samples, n_features]
          centroid: (numpy or scipy matrix) centroid matrix    
   
        Returns:
          (scipy.sparse.csr.csr_matrix) distances of X vectors 
          from centroid
        """
        # Normalize each row of X
        X = sklearn.preprocessing.normalize(X, axis=1)
        # cosine distance between exemplars and centroid vectors
        result = X * centroid.T
        return result

    def _extract_prediction(self, R):
        """Switch from internal Matrix representation to python
        representation of results

        Args :
          R : A scipy sparse matrix of dimensions (nsamples,ntopics) of
          classifier scores.

        Returns:
           (list) Each sample or row in list contains a sequence of 
           (score, topic_id) pairs sorted in descending order of score
        """
        result = []
        for row in range(R.shape[0]):
            result.append([(R[row, col], col) for col in range(R.shape[1])])
            result[row].sort(reverse=True)
        return result

    @timed
    def metrics(self, X):
        """
        Compute radii and spread of centroid models
        X here is a set of exemplars per class label

        Args :
          X: (scipy.sparse.csr_matrix) matrix of data with
             shape = [n_samples, n_features]

        Returns :
           (list) tuple of score, radii of data wrt centroid
           vector computed from data 
        """
        radii = self.radii(X)
        spread = self.spread(X)
        return (spread, radii)

    def radii(self, X):
        """
        Computes radii of data matrix from centroid
        X here is a set of exemplars per class label

        Args:
          X: (scipy.sparse.csr_matrix) matrix of data with
             shape = [n_samples, n_features]

        Returns:
           (tuple) A row matrix with the distance of each 
           exemplar from their collective centroid; mean of radii, 
           std of radii
        """
        centroid = self.centroid(X)
        radii = centroid - X
        return (radii, np.mean(radii), np.std(radii))

    def spread(self, X):
        """
        Computes spread of data matrix from centroid
        X here is a set of exemplars per class label

        Args:
          X: (scipy.sparse.csr_matrix) matrix of data with
             shape = [n_samples, n_features]

        Returns:
          (tuple) classifier score of each exemplar in X 
          relative to their collective centroid, mean of spread &
          std of spread 
        """
        centroid = self.centroid(X)
        scores = self._compute_scores(X, centroid)
        return (scores, np.mean(scores), np.std(scores))

    def stats(self):
        """ Returns statistics of the classifier """
        if not hasattr(self, "models_"):
            raise AttributeError("Model has not been trained yet.")
        shape = self.models_.shape
        return {'topics': shape[0], 'vocabulary': shape[1]}
