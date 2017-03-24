import numpy as np
#import matplotlib.pyplot as plt
# load MNIST dataset
import cPickle, gzip
# For the elastic distorsion transformation
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#import  matplotlib.pyplot as plt 


class BatchDispenser:

    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        self.X_test = []
        self.y_test  = []
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = 0
        self.transform_data = False

    def load_dataset(self):
        # Load the dataset
        f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        # Inicialize
        self.X_train = train_set[0]
        self.y_train = train_set[1]
        self.X_val = valid_set[0]
        self.y_val = valid_set[1]
        self.X_test = test_set[0]
        self.y_test = test_set[1]
        # Number of examples
        self.num_examples = self.X_train.shape[0]
        # Print Shapes
        print "Train set:", self.X_train.shape, self.y_train.shape
        print "Val set:", self.X_val.shape, self.y_val.shape
        print "Test set:", self.X_test.shape, self.y_test.shape

    # serve data by batches
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        # when all trainig data have been already used, it is reorder randomly
        if self.index_in_epoch > self.num_examples:
            # finished epoch
            self.epochs_completed += 1
            # After epoch 1 we transform the data
            self.transformData = True
            # shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.X_train = self.X_train[perm]
            self.y_train = self.y_train[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        X_bach = self.X_train[start:end]
        Y_bach = self.y_train[start:end]
        if self.transform_data:
            X_bach = transform_data(X_bach)
        return X_bach, Y_bach

     # serve data by batches
    def next_batch_matrix(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        
        # when all trainig data have been already used, it is reorder randomly
        if self.index_in_epoch > self.num_examples:
            # finished epoch
            self.epochs_completed += 1
            # After epoch 1 we transform the data
            self.transformData = True
            # shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.X_train = self.X_train[perm]
            self.y_train = self.y_train[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        X_bach = self.X_train[start:end]
        Y_bach = self.y_train[start:end]
        if self.transform_data:
            X_bach = transform_data(X_bach)

        image_width = image_height = np.ceil(  np.sqrt( X_bach.shape[1] ) ).astype(np.uint8)
        X_bach = X_bach.reshape( (X_bach.shape[0], image_width, image_height) )

        return X_bach, Y_bach


# UTILITES
def show_images_tensor(X, y):  
    numsubplots = np.ceil( np.sqrt( X.shape[0] ) ).astype(np.int8)
    # 8x8 only for represenatation
    fig, axes = plt.subplots(numsubplots, numsubplots, figsize=(8, 8))  
    for i, ax in enumerate(axes.flat):
        ax.imshow( 
            X[i,:,:], 
            cmap='binary', 
            interpolation='nearest'
            )
        ax.text(
            0.05, 0.05, 
            str(y[i]), 
            transform=ax.transAxes, 
            color='green'
            )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



# UTILITES
def show_images(X, y):
    numsubplots = np.ceil( np.sqrt( X.shape[0] ) ).astype(np.int8)
    
    image_width = image_height = np.ceil( 
        np.sqrt( X.shape[1] )
        ).astype(np.uint8)

    # 8x8 only for represenatation
    fig, axes = plt.subplots(numsubplots, numsubplots, figsize=(8, 8))  
    for i, ax in enumerate(axes.flat):
        ax.imshow( 
            X[i,:].reshape( (image_width, image_height) ), 
            cmap='binary', 
            interpolation='nearest'
            )
        ax.text(
            0.05, 0.05, 
            str(y[i]), 
            transform=ax.transAxes, 
            color='green'
            )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def transform_data(X):
    return np.apply_along_axis(elastic_transform, 1, X)

def elastic_transform(image, alpha = 35.0, sigma=5.0, random_state=None):
    """ Elastic distrosions from a image

    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    At the beginning of every epoch the entire MNIST training set gets 
    deformed. Initial experiments with small networks suggested the following 
    deformation parameters: sigma = 5.0 - 6.0, alfa = 36.0 -38.0

    Args:
        image: any size image.
        alpha:
        sigma:
        random_state:

    Retruns:
        The elastic trasformed image.

    Raises:
        assert: If image is not 2 dimensional   
    """

    sigma = float(np.random.uniform(5, 6, size=1))
    alpha = float(np.random.uniform(36, 38, size=1))

    image_size = np.ceil( np.sqrt( image.shape[0] )).astype(np.uint8)
    image  = image.reshape( (image_size, image_size ) )
    assert len(image.shape)==2
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), 
        sigma, 
        mode="constant", 
        cval=0
        ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0
        ) * alpha
    x, y = np.meshgrid(np.arange(
        shape[0]), 
        np.arange(shape[1]), 
        indexing='ij'
        )
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    final =  map_coordinates(image, indices, order=1).reshape(shape)
    final  = final.reshape( (final.shape[0]*final.shape[1], ) )
    return final
