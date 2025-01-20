
import numpy as np
import regex as re

class DataEncoding:


    def __init__(self):
        super(DataEncoding, self).__init__()
    
    def tokenizer(self, smiles_list):
        """
        Tokenization of the SMILES and creation of the vocabulary (dictionary) of the tokenized SMILES.
        arguments:
        smiles_list: list of SMILES to be tokenized
        returns
        tokenized smiles list
        vocabulary in form of a dictionary from the tokenized SMILES
        """

        # Breakdown of regular expression:
        # "Cl" or "Br": Matches the halogen elements Chlor and Brom
        # "\[.*?\]": Matches any bracketed group (special groups)
        # "[0-9A-Za-z)=(-+#]": Matches individual alphanumeric characters, specific punctuation (e.g., parentheses, equals sign), or the "#" symbol
        # "\-": Matches a standalone dash, often used to represent single bonds or as part of a chemical formula.
        regular_expression = r"Cl|Br|\[.*?\]|[0-9A-Za-z)=(-+#]|\-"
        
        tokenized_list = []
        for ix, smiles in enumerate(smiles_list):
            char_smiles_list = re.findall(regular_expression, smiles)   # Find all regular expressions in one SMILES strings
            tokenized_list.append(char_smiles_list)
        
        return tokenized_list
    
    def add_tokens(self, smiles_list, starting_token = ['G'], end_token = ['E']):
        """ 
        Adding a starting and ending token to the SMILES list. G ('go') in the beginning, 'E' ('end') at the end.
        Molecule is padded with 'E'. SMILES strings need to be tokenized first
        arguments
        smiles list: SMILES list
        starting_token: token that signalizes the start of the molecule.
        end_token: token that signalizes the end of the molecule.
        returns
        padded smiles list
        """

        max_length = max(len(seq) for seq in smiles_list)   # determine longest SMILES sequence
        # Padding of SMILES strings
        padded_sequences = [starting_token + seq + end_token * (max_length - len(seq)) for seq in smiles_list]

        return padded_sequences
    
    def create_vocab(self, smiles_list, vocab=None):
        """ 
        Creation of the vocabulary for further use. All separate tokens need to be contained in the vocabulary.
        arguments
        smiles list: SMILES list
        vocab: eventually already available vocabulary
        returns
        padded smiles list
        """

        tokenized_smiles = self.tokenizer(smiles_list)
        tokenized_smiles = self.add_tokens(tokenized_smiles)
        
        flat_list = [item for sublist in tokenized_smiles for item in sublist]  # flatten list of tokenized SMILES
        unique_char = list(set([x for x in flat_list]))     # eliminate duplicated
        if vocab == None:
            vocab = dict((char, i) for i, char in enumerate(unique_char))   # if no vocab, create one from the unique characters
        else:   # updating the already available dictionary
            starting_id = len(vocab)    
            for i in range(0, len(unique_char)):
                if unique_char[i] not in vocab.keys():  # should not be contained in the vocab already
                    vocab.update({unique_char[i] : starting_id})
                    starting_id += 1
                    
        return vocab, tokenized_smiles
    
    def one_hot_encoding(self, smiles_list, encoding_dict):
        """
        Encoding of the SMILES: The characters are transformed into integers.
        Padding does already take place by making the dim of the matrix to max_length_seq + 1.
        For 
        arguments
        tokenized_list: List of SMILES in its tokenized form
        returns
        encoded SMILES, encoding vocabulary
        """

        smiles_list = np.asarray(smiles_list)
        
        # Creation of one hot 
        one_hot = np.zeros((smiles_list.shape[0], smiles_list.shape[1] + 1, len(encoding_dict)), dtype=np.int8)
        # Parse through every SMILES
        for i, smile in enumerate(smiles_list):
            # Parse through every character
            for j, char in enumerate(smile):
                one_hot[i, j, encoding_dict[char]] = 1  # substitute where the SMILES has the char
            one_hot[i, len(smile):, encoding_dict["E"]] = 1     # padded
        
        return one_hot, encoding_dict