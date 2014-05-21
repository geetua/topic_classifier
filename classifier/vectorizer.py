from math import log
import scipy.sparse
import scipy.linalg
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict


def tfidf(tf, df, num_corpus_docs):
    """
    Computes tfidf of document

    Args:
      tf = (int) term_frequency for a doc
      df = (int) doc frequency (i.e. number of docs term appears in)
      num_corpus_docs = (int) total number of documents in corpus

    Returns:
      (float) tfidf score

    Raises:
      ValueError: Incorrect document frequency value
    """
    if df <= 0:
        raise ValueError('Incorrect doc frequency value')
    return tf * log(float(num_corpus_docs) / float(df))


def topic_exemplars_to_bunch(topic_exemplars):
    """
    Converts training data (topic id with exemplar docs from solr)
    to sklearn Bunch object

    Args:
      topic_examplars: (dict) Each topic id is associated with a list of 
      term dictionaries. Each term dictionary is a dict of terms in document with
      attributes - tf, df, and ndocs and having format as below:
              {chid : term_dcts, chid : term_dcts, ...}
              where term_dcts = list of term_dct objects 
              term_dct = {term :{'tf' : tf, 'df' : df, 'ndocs' : ndocs},
                term : {'tf' : tf, 'df' : df, 'ndocs' : ndocs}, ...}
    Returns:
      (Bunch) sklearn dict object with attributes data, target_names and targets
 
    """
    data = []
    target_names = []
    targets = []

    for index, (topic, exemplars) in enumerate(topic_exemplars.iteritems()):
        target_names.append(topic)
        for example in exemplars:
            data.append(example)
            targets.append(index)

    return Bunch(data=data,
                 target_names=target_names,
                 targets=targets)


class SolrVectorizer(BaseEstimator, TransformerMixin):

    """
    A Vectorizer in the sklearn paradigm where we consolidate the vocabulary
    and normalize the tfidf from the given solr term vector data.

    Attributes:
      vocabulary_: (dict) term to index mapping for feature vobulary
      lookup_:     (dict) index to term mapping for feature vobulary 
    """

    def fit(self, data):
        """
        Learn a vocabulary and lookup table for all the training data

        Args:
          data: (list) feature vectors or term dictionaries. Each term 
          dictionary is a dict of terms in document with attributes: tf, df, and ndocs
          and having format [term_dct, term_dct, ...]
                  where term_dct = { term :{'tf' : tf, 'df' : df, 'ndocs' : ndocs},
                                     term : {'tf' : tf, 'df' : df, 'ndocs' : ndocs}, ...}        
        """
        vocabulary = set()
        for term_dct in data:
            vocabulary.update(term_dct.keys())
        self.vocabulary_ = dict([(term, i)
                                for i, term in enumerate(vocabulary)])
        self.lookup_ = dict([(i, term) for i, term in enumerate(vocabulary)])
        return self

    def transform(self, data):
        """
        Transforms the feature vectors or term dictionaries by normalizing
        the term values to tfidf scores and returns scipy.sparse.matrices 
        for use with sklearn estimators

        Args:
          data: (list) feature vectors or term dictionaries. Each term 
          dictionary is a dict of terms in document with attributes: tf, df, and ndocs
          and having format [term_dct, term_dct, ...]
                  where term_dct = { term :{'tf' : tf, 'df' : df, 'ndocs' : ndocs},
                                     term : {'tf' : tf, 'df' : df, 'ndocs' : ndocs}, ...}
        Returns:
          (scipy.sparse.csr_matrix) training data matrix of shape = [n_samples, n_features]
          where each row corresponds to a training sample 
          n_samples = total number of training samples (=len(data))
          n_features = vocabulary size or total number of features

        Raises:
          AttributeError: if vocabulary has not been fitted first
        """
        if not getattr(self, 'vocabulary_', None):
            raise AttributeError("No Vocabulary has been yet fitted")

        row = []
        col = []
        values = []
        for idx, tdct in enumerate(data):
            for term, tfdf in tdct.iteritems():
                try:
                    vindex = self.vocabulary_[term]
                except KeyError:
                    continue
                col.append(vindex)
                row.append(idx)
                values.append(tfidf(tfdf['tf'], tfdf['df'], tfdf['ndocs']))
        return scipy.sparse.csr_matrix((values, (row, col)),
                                       shape=(len(data), len(self.vocabulary_)))
