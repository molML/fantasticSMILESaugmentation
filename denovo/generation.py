import json
import keras
import pickle
from data_generator import DataGenerator

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

class CLM:
    
    """
    
    Chemical Language Models for Training, using Data Generator for large dataset sizes.
    
    """

    def __init__(
            self,
            model_parameters: dict,     
            mode: str,      # Mode either Train, Finetune or Predict
            path: str,      # Saving path of the model
            pre_trained_model_path: str = None,     # If pre-trained model is necessary
            freeze_layers: bool = False,
            use_data_generator: bool = False   # New flag to toggle data generator
            ):
        
        super(CLM, self).__init__()

        self.model_parameters = model_parameters  # Parameters of the model, saved in a dictionary        
        self.mode = mode        # Mode of the model, either Train, Finetune, Predict
        if self.mode not in ['Train', 'Predict', 'Finetune']:
            raise ValueError(f"Invalid mode '{self.mode}'. Allowed modes are: 'Train', 'Predict', 'Finetune'.")
        
        self.path = path    # Saving path of the new trained model
        self.pre_trained_model_path = pre_trained_model_path    # Path of pre-traing model (if mode == Finetune)        
        self.use_data_generator = use_data_generator      # if using data generator or not (for large datasets)

        self.info_size = self.model_parameters['info_size'] # Size of axis, it is the vocab size

        # If model on train mode, then batch size adaptable, stateful model needs to be False so information is
        # not remebered between the batches
        if self.mode == 'Train':
            stateful = False
            self.batch_size = self.model_parameters['batch_size']
        # If model on fine-tuning mode, then batch size to two (small batch sizes, as small dataset), stateful is False
        # as still in training mode
        if self.mode == 'Finetune':
            stateful = False
            self.batch_size = 2
            if self.pre_trained_model_path is None:
                raise ValueError("pre_trained_model_path cannot be None in 'Finetune' mode.")
        # If model on predict mode, then batch size to one, so that we can sampled one-by-one. Stateful is True,
        # so information is remebered between batches.
        elif self.mode == 'Predict':     
            stateful = True
            self.batch_size = 1

        # Creation of the LSTM layers of the model. 
        # Different number of layers and sizes are possible.
        self.layers_lstm = []
        for layer_ix, hidden_units in enumerate(self.model_parameters['size_layers']):
            self.lstm = keras.layers.LSTM(
                hidden_units,
                return_sequences = True,
                activation = self.model_parameters['lstm_activation'], 
                recurrent_activation = self.model_parameters['lstm_recurrentactivation'],
                dropout = self.model_parameters['dropout_rate'],
                stateful = stateful,
                name=f'lstm{layer_ix}_layer')
            self.layers_lstm.append(self.lstm)

        # Creation of Dense layer
        self.dense = keras.layers.TimeDistributed(keras.layers.Dense(
            self.model_parameters['info_size'],
            activation = self.model_parameters["dense_activation"],
            name = 'output_layer'))

        self.epochs = self.model_parameters['n_epochs']     # Epochs for training

        # Calling of parameters from the self.model_parameters
        self.optimizer = keras.optimizers.get(self.model_parameters['optimizer_name'])
        self.optimizer.learning_rate.assign(self.model_parameters['learning_rate'])
        self.loss = self.model_parameters['loss']
        self.metrics = self.model_parameters['metric'] 

        # Creating the settings for transfer learning and fine-tuning
        # Mostly for fine-tuning mode
        if freeze_layers:
            # Trying of loading of model weights from pre-training
            try:
                self.load_weights(pre_trained_model_path)
            except FileNotFoundError:
                print('File with trained model not found, please pre-train the model before.')
            for layer_lstm in self.layers_lstm:
                layer_lstm.trainable = False      # Setting trainable to False, so layers are frozen   

    def call(self, inputs, training=None):
        """
        Calls the model.
        arguments
        inputs: input of the model
        returns
        output of the model after training process
        """      

        x_s = inputs
        
        for layer_lstm in self.layers_lstm:
            x_s = layer_lstm(x_s,  training=training)
        
        x_out = self.dense(x_s)

        return x_out

    def train_model(self, 
                    training_data=None, 
                    validation_data=None, 
                    xTrain=None, 
                    yTrain=None, 
                    xVal=None, 
                    yVal=None,
                    vocab=None):

        """
        Training of the model.
        arguments
        training_data: training data
        validation_data: validation data
        vocab: vocabulary to perform data encoding on the go
        returns
        model, history of training process
        """
        
        if self.mode == 'Predict' or self.mode == 'Finetune':
            raise ValueError('Mode of the Model is  "Predict". Call "Train" mode.')
        
        my_callbacks=[
            keras.callbacks.ModelCheckpoint(filepath=self.path+'/m.keras', monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, patience=10)
            ]
        
        inputs = keras.layers.Input((None, len(vocab)))     # HELENA

        model = keras.models.Model(
            inputs=[inputs], 
            outputs=self.call(inputs)
            )
        
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics]) # Compile model

        # if we need to use data generator
        if self.use_data_generator:
            # Vocab, training_data and validation_data cannot be None and needs to be entered
            if not vocab or training_data is None or validation_data is None:
                raise FileNotFoundError('Vocab, training_data, and validation_data must be provided when using a data generator.')
            # Generate training and validation data
            train_generator = DataGenerator(training_data, vocab, self.batch_size)
            val_generator = DataGenerator(validation_data, vocab, self.batch_size)
            history = model.fit(train_generator,
                                epochs = self.epochs,
                                validation_data = val_generator,
                                callbacks=my_callbacks,
                                verbose=1)
            
        # if data generator not necessary    
        else:
            # To ensure xTrain, yTrain, xVal, and yVal are provided
            if xTrain is None or yTrain is None or xVal is None or yVal is None:
                raise ValueError('xTrain, yTrain, xVal, and yVal must be provided when not using a data generator.')
        
            validation = (xVal, yVal) # Create tupel
            # Training of the model
            history = model.fit(
                xTrain, yTrain, 
                batch_size=self.batch_size, 
                epochs=self.epochs, 
                validation_data=validation,
                callbacks=my_callbacks)
        
        # Saving various components of the model after training process:
        model.save_weights(f'{self.path}/.weights.h5')
        model.save(f'{self.path}/model.h5')
        training_config = model.get_config()

        # Saving model configuration
        with open(f"{self.path}/training_config.json", "w") as json_file:
            json.dump(training_config, json_file)

        # Saving training history
        with open(f"{self.path}/training_history.pkl", "wb") as history_file:
            pickle.dump(history.history, history_file)

        return model, history
    
    def fine_tune_model(self, xTrain, yTrain, xVal=None, yVal=None):
        """
        Fine-Tuning of the model.
        arguments
        training_data: training data
        validation_data: validation data
        vocab: vocabulary to perform data encoding on the go
        returns
        model, history of training process
        """

        if self.mode == 'Predict' or self.mode == 'Train':
            raise ValueError(f'Mode of the Model is  {self.mode}. Call "Finetune" mode.')
           
        if not xVal.any() and not yVal.any():
            validation_split=0.2
            validation=None
        else:
            validation_split=0.0
            validation=(xVal, yVal)

        my_callbacks=[
            keras.callbacks.ModelCheckpoint(filepath=self.path+'/m.keras', monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, patience=10)
            ]
        
        inputs = keras.layers.Input((None, xTrain.shape[2]))  
        model = keras.models.Model(
            inputs=[inputs], 
            outputs=self.call(inputs)
            )

        # Update the learning rate for fine-tuning
        fine_tuning_learning_rate = self.model_parameters['learning_rate'] * 0.001
        # Define the optimizer with gradient clipping
        updated_optimizer = keras.optimizers.Adam(learning_rate=fine_tuning_learning_rate, clipnorm=1.0)

        # Unfreeze specific LSTM layer, so it can adjust its weights.
        for layer in self.layers_lstm:
            layer.trainable = True

        # Calling of the weights of the (pre-)trained model
        trained_model = keras.models.load_model(f'{self.pre_trained_model_path}/model.h5')
        model.set_weights(trained_model.get_weights())

        model.compile(optimizer=updated_optimizer, loss=self.loss, metrics=[self.metrics])

        # Training of the model
        history = model.fit(
            xTrain, yTrain, 
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            validation_data=validation,
            validation_split=validation_split,
            callbacks=my_callbacks)
        
        # Saving various components of the model after training process:
        model.save_weights(f'{self.path}/.weights.h5')
        model.save(f'{self.path}/model.h5')
        training_config = model.get_config()

        with open(f"{self.path}/training_config.json", "w") as json_file:
            json.dump(training_config, json_file)

        with open(f"{self.path}/training_history.pkl", "wb") as history_file:
            pickle.dump(history.history, history_file)

        return model, history

    def predict_model(self):
        """
        Prediction mode of the model. If mode is Train, then error message.
        Model is build and weights are based on the saved training model.
        arguments: 
        None
        returns
        model with weights set for prediction
        """

        if self.mode == 'Train' or self.mode == 'Finetune':
            raise ValueError(f'Mode of the Model is still in {self.mode}. Call "Predict" mode.')
        
        inputs = keras.layers.Input(batch_shape=(self.batch_size, None, self.info_size))
        model = keras.models.Model(
            inputs=[inputs], 
            outputs=self.call(inputs)
            )
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])
        
        # Calling of the weights of the (in the past) trained model
        trained_model = keras.models.load_model(f'{self.path}/model.h5')
        model.set_weights(trained_model.get_weights())

        return model 
