
import numpy as np
import time
from nutrition.structure.data_set import DataSet
from nutrition.structure.counter import Counter
from nutrition.feature import stanford_feature

class FeatureExtractor(object):
    
    def save(self):
        self.data_set.save_feature_matrix(self.feature_matrix)
    
    def __init__(self, data_set):
        self.data_set = data_set
        
        # load features we have extracted so far
        # this would not be needed if we use append file instead of np.savetxt to store the feature matrix
        self.feature_matrix = self.data_set.load_feature_matrix()
        
        # load count, i.e. how many documents have been parsed successfully
        counter = Counter(data_set.feature_path,
            commit_interval=50,
            on_commit=self.save)
        
        start = time.time()
        while counter.count < data_set.data['count']:
            doc_start = time.time()
            
            # read raw text and parse tree
            text = data_set.get_text(counter.count)
            label = data_set.data['labels'][counter.count]
            annotation = data_set.load_stanford_annotation(counter.count)
            
            # insert row to matrix
            # also, initialize feature matrix if it is None
            row = stanford_feature.get_features(annotation)
            row.append(label)

            if self.feature_matrix is None:
                self.feature_matrix = np.zeros([data_set.data['count'], len(row)])
            self.feature_matrix[counter.count,:] = row
            
            # count annd print
            counter.increment()
            print('%i, %i%% %.2f seconds (%.0f total))' % (counter.count-1, 100*counter.count/data_set.data['count'], time.time() - doc_start, time.time() - start))
            
        counter.commit()

if __name__ == '__main__':
    data_set = DataSet('cepp')
    FeatureExtractor(data_set)
    