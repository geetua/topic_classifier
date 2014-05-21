import logging
from collections import defaultdict
from pprint import pprint as pp

from classifier.core import TopicClassifier
from classifier.vectorizer import topic_exemplars_to_bunch, SolrVectorizer

docs = {'ones': [" ".join(map(str, range(0, 10, 2))),
                 " ".join(map(str, range(1, 10, 2))),
                 " ".join(map(str, range(0, 10, 4))),
                 " ".join(map(str, range(1, 10, 4)))],
        'tens': [" ".join(map(str, range(10, 100, 20))),
                 " ".join(map(str, range(11, 100, 20))),
                 " ".join(map(str, range(10, 100, 40))),
                 " ".join(map(str, range(11, 100, 40)))],
        'hundreds': [" ".join(map(str, range(100, 1000, 200))),
                     " ".join(map(str, range(101, 1000, 200))),
                     " ".join(map(str, range(100, 1000, 400))),
                     " ".join(map(str, range(101, 1000, 400)))],
        }

# pp(docs)

testdocs = {'hundreds': ["501 900 300 10 30"],
            'tens':  ["701 31 91 10 30"],
            'ones':  ["8 9 71 1 0 2000"]
            }


def termvecs(docdct):
    ndocs = 0
    df = defaultdict(int)

    result = defaultdict(list)
    for topic, docs in docdct.iteritems():
        for doc in docs:
            words = doc.split()
            tf = defaultdict(int)
            for word in words:
                tf[word] += 1
            result[topic].append(tf)
            for w in tf.keys():
                df[w] += 1
        ndocs += len(docs)

    tvs = {}
    for topic, tfs in result.iteritems():
        termvecs = []
        for tf in tfs:
            termvec = defaultdict()
            for w, cnt in tf.iteritems():
                d = dict(tf=cnt,
                         df=df[w],
                         ndocs=ndocs)
                termvec[w] = d
            termvecs.append(termvec)
        tvs[topic] = termvecs
    return tvs


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    sv = SolrVectorizer()
    tvs = termvecs(docs)
    bunch = topic_exemplars_to_bunch(tvs)

    # Fitting and transforming training exmplars using SolrVectorizer
    train_vector = sv.fit_transform(bunch.data)

    cc = TopicClassifier()
    model = cc.fit(train_vector, bunch.targets)

    # lets look at spread and radii of first 3 vectors
    # (spread,radii) = cc.metrics(train_vector[:3])

    tvs = termvecs(testdocs)
    test_bunch = topic_exemplars_to_bunch(tvs)
    test_vector = sv.transform(test_bunch.data)

    predictions = cc.predict_with_scores(test_vector)
    for index, result in enumerate(predictions):
        print "\nTest Doc:", index
        for s, l in result:
            print "Score, Class Label:", s, bunch.target_names[l]

    correct_Y = test_bunch.targets
    # Note: this calls score inherited from ClassifierMixin
    score = cc.score(test_vector, correct_Y)
    print "\nAccuracy:", score
