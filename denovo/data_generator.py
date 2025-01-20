import numpy as np
import keras
from encoding import DataEncoding


def split_input_target(sequence, sequence_label=None):
    """ 
    Sequence is separated into input and target for one-hot encoded datasets. 
    arguments 
    sequence: one-hot encoded data to be separated into input and target
    sequence_label: label data that does not contain dummy atom (for masking)
    returns
    input_seq, label_seq
    """

    # Input of sequence is retrieved until penultimate sequence-entry
    input_seq = sequence[:, 0 : -1, :] 

    # If sequence_label is given (ergo masking), then following case
    # The label_sequence is always retrieved the same: The second to last sequence-entry are retrieved
    if sequence_label is not None:
        if sequence_label.any(): 
            label_seq = sequence_label[:, 1:, :]   # from sequence_label
        else:
            label_seq = sequence[:, 1:, :]     # from original sequence
    else:
        label_seq = sequence[:, 1:, :]     
        
    return input_seq, label_seq


class DataGenerator(keras.utils.Sequence):

    """ 
    Generates data for large datasets.
    """

    def __init__(self, 
                 data, 
                 vocab,
                 batch_size = 1024, 
                 shuffle=True):
        
        self.data = data
        self.vocab = vocab   

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))    # array of indexes for data

        # Flag check for batch size, cannot be bigger than dataset size
        if batch_size > len(self.data):
            raise ValueError("Batch size has to be less than the number of sequences.")
        
        # Check for vocabulary
        if not vocab or not isinstance(vocab, dict):
            raise ValueError("Invalid vocabulary provided.")

        self.shuffle = shuffle  # Flag to shuffle
        self.on_epoch_end() # shuffle data initially or at the end of each epoch

        # Initialize of data encoder
        self.encoder = DataEncoding()

    def __len__(self):
        """ 
        Calculates the number of batches per epoch.
        arguments
        None
        returns
        number of batches in one epoch
        """

        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        """ 
        Generate one batch of data for encoding. 
        arguments
        index: index of batch to retrieve
        returns
        X,y (X input data, y target data) as a tuple
        """

        # Calculate the indexes for the current batch based on the batch size
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Retrieve the data sequences corresponding to the batch indexes
        train_sequences = [self.data[i] for i in batch_indexes]

        # Calls the function to one-hot encode the molecule and divide it into input and target data
        X, y = self.__data_generation(train_sequences)

        return X, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch, shuffing them if necessary. 
        arguements
        None
        returns
        None
        """

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_list):
        """ Generates data containing batch_size samples. It encodes and separates the data
        into input and target set. 
        arguments
        data_list: list of data to be encoded
        returns
        inputs, label
        """

        if not data_list:
            raise ValueError("Data list is empty. Cannot generate batch.")
        
        # Encoding of data with pre-defined vocabulary
        one_hot_encoded, _ = self.encoder.one_hot_encoding(data_list, self.vocab)
        # Splitting into input and label
        inputs, labels = split_input_target(one_hot_encoded)

        return inputs, labels