from diploma import *
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, itemsPaths, batch_size=32, input_dim=(224,224), n_channels=3, n_classes=2, shuffle=True,equal_classes=True, augmentation=None):
        'Initialization'
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.itemsPaths = itemsPaths
        self.shuffle = shuffle
        self.on_epoch_end()
        self.shuffleExamples()
        self.augmentation = augmentation

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.itemsPaths) / self.batch_size))

    def __getitem__(self, index):
        # Generate data
        X, y = self.__data_generation(index)
        return  X, y

    def shuffleExamples(self):
        np.random.shuffle(self.itemsPaths)
        
    def get_labels(self):
        return np.array(self.itemsPaths.map(lambda example: example["class_num"]))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.shuffleExamples()
    def __data_generation(self,index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        for i, array_index in enumerate(range(self.batch_size*index,  self.batch_size*index + self.batch_size)):
            item =  self.itemsPaths[array_index]
            class_name = item["class_num"]
            img = self.getImage(item["frame_path"],self.input_dim)
            if self.augmentation:
                for function in self.augmentation:
                    img = function(img)
            X[i] = img 
            y[i] = class_name 
        return X, y
    def getImage(self,image_path, input_image_size):
    # Read and normalize an image
        img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, input_image_size,  interpolation=cv2.INTER_LINEAR )
        img = img/255.0
        img = img.reshape(img.shape +(1,))
        return img




#@title Dataset Generator for single frame input TODO
class DataGenerator_OLD(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, itemsPaths, batch_size=32, input_dim=(224,224), n_channels=3, n_classes=2, shuffle=True,equal_classes=True, augmentation=None):
        'Initialization'
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.itemsPaths = itemsPaths
        self.shuffle = shuffle
        self.on_epoch_end()
        self.shuffleExamples()
        self.augmentation = augmentation

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.itemsPaths) / self.batch_size))

    def __getitem__(self, index):
        # Generate data
        X, y = self.__data_generation(index)
        return  X, y

    def shuffleExamples(self):
        np.random.shuffle(self.itemsPaths)

    def get_labels(self):
        return np.array(self.itemsPaths.map(lambda example: example["class_num"]))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.shuffleExamples()
    def __data_generation(self,index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        for i, array_index in enumerate(range(self.batch_size*index,  self.batch_size*index + self.batch_size)):
            item =  self.itemsPaths[array_index]
            class_name = item["class_num"]
            imgs_res = self.getImage(item["frame_path"],self.input_dim)
            X[i] = imgs_res 
            y[i] = class_name 
        if self.augmentation:
            return self.augmentation(X, training=True),y
            augmentation(x, training=True)
        else:
            return X, y
    def getImage(self,image_path, input_image_size):
    # Read and normalize an image
        img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, input_image_size,  interpolation=cv2.INTER_LINEAR )
        img = img/255.0
        img = img.reshape(img.shape +(1,))
        return img

