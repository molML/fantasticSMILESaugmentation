import os
file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

import numpy as np
import json
from generation import CLM


class SamplingMolecules:
    
    def __init__(self, sampling_parameter, mode: str = None):

        super(SamplingMolecules, self).__init__()
        self.sampling_parameter = sampling_parameter
        # Call in the hyperparameters of the 
        self.hp_space = self.sampling_parameter["hps"]

        if mode != 'Finetune':
            self.model_dir = self.sampling_parameter["hps"]["combination_saving_dir"]
        else:
            # Change depending on what the task of the model is!
            self.model_dir = self.sampling_parameter["fine_tuning"]["saving_dir"]
        
        # Loading of the vocabulary 
        with open(f"{self.hp_space['main_saving_dir']}/segment2label.json", 'r') as json_file:
            self.encoding_dict = json.load(json_file)

        # initialization of model. if more than one model, then adaptable
        self.model = CLM(self.hp_space, 'Predict', self.model_dir)
        self.model = self.model.predict_model() # Model is in prediction mode

        self.max_seq_len = self.hp_space['maxlen']
        self.max_info_len = self.hp_space['info_size']

        self.decoding_dict = {x: y for y, x in self.encoding_dict.items()}
    
    def temperature_sampling(self, prediction, temperature=1.0):
        """
        Temperature sampling of the predicitions. If not mentioned otherwise, the temperature is 1.0.
        Arguments:
        prediction: What was predicted on the model.
        temperature: temperature for sampling
        Returns:
        Array with the predicted value based on the multinomial distrubution
        """

        prediction = np.asarray(prediction).astype('float64')
        preds = np.log(prediction) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        probability = np.random.multinomial(1, preds, 1)
        next_value = np.argmax(probability)     # takes the position of the max value
        return next_value

    def sample_one(self, temperature=1.0, start_char = 'G'):
        """
        Sampling of one molecule. Depending on the temperature, more or less conservative molecules are created 
        (see sample_multiple).
        arguments
        temperature: T > 1.0 leads to more divers, T < 1.0 to more conservative molecules
        starting character: depends on the encoding procedure
        returns:
        molecule string in SMILES form
        """
        
        # Creating of the starting vector
        next_char = np.zeros((1, 1, self.max_info_len))
        next_char[0, 0, self.encoding_dict[start_char]] = 1

        # Start of the molecular string
        molecule_string = start_char
            
        for _ in range(self.max_seq_len - 1):   # 
        
            pred = self.model.predict(np.expand_dims(next_char[:,-1,:],axis = 1), verbose=0).flatten()  # Prediction of the next character of the sequence
            next_token = self.temperature_sampling(pred, temperature)     # Temperature sampling
            next_token = self.decoding_dict[next_token]     # Decoding of the predicted value
            
            molecule_string += next_token   # Adding to the molecular string
            
            # Creating of the next atom of the molecular vector
            x_next = np.zeros((1, 1, self.max_info_len))    
            x_next[0,0, self.encoding_dict[next_token]] = 1
            next_char = np.append(next_char, x_next, axis=1)

            # If we have a end / padding character, then break
            if next_token == 'E':
                break

        # Reset states of the model   
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        
        return molecule_string

    def sample_multiple(self, num_sample, temperature = 1.0, starting_char='G'):
        """
        Samples n_samples molecules at a specific temperature T. T < 1.0: conversative sampling,
        T > 1.0: diverse sampling. T = 1.0: output of the model is linear.
        arguments
        num_sample: number of molecules to be sampled
        starting_char: starting character in sampling procedure
        returns
        list of sampled molecules
        """

        molecules = list()
        for _ in range(num_sample):
            # Sample one molecules
            molecule_string = self.sample_one(temperature, starting_char)
            molecule_string = molecule_string[1 : -1]     # Eliminating the starting and end character
            molecule = "".join([str(item) for item in molecule_string])     # joing string together
            molecules.append(molecule)
        return molecules