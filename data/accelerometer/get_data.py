import pickle
import pandas as pd
from pathlib import Path

class DataLoader:
    '''
    Loads data and stores relevant meta data, one file at a time.
    '''
    def __init__(self, file_path, meta_data=None):
        '''
        Args:
            file_path: file path either as str or Path object (will accept None but no data will be loaded)
            meta_data: associated meta data, defaults to empty dictionary
        '''
        
        if file_path is not None and not isinstance(file_path, (str, Path)):
            raise TypeError('file_path should be either str or Path type') 
        
        if meta_data is None:
            meta_data = {} # empty dict assigned here since 'mutable defaults' create shared a mutable if the class/function is called again (not what we want)

        self.file_path = file_path
        self.meta_data = meta_data
        self.timesteps = None
        self.data = None
        
        if file_path is not None:
            self._load_data()
            

    def _load_data(self):
        '''
        Loads data from the accepted file types into class attribute, data, and stores
        the number of values in attribute, timesteps.

        Currently accepted file types: pkl
        '''

        if self._is_file('pkl'):
            with open(self.file_path,"rb") as f:
                self.data = pickle.load(f)
            self.timesteps = self.data.shape[1] # expecting 2D array from self.data, .shape[0] yields number of channel (expecting 3)
        else:
            raise ValueError('Unsupported file format, expecting .pkl') # currently only takes .pkl, can add more if necessary
    

    def _is_file(self, file_type):
        '''
        Checks if attribute, file_path matches the expected file_type. Accommodates
        file_path as a string or Path object.

        Args:
            file_type: expected file type, with or without the dot (e.g., '.pkl', 'pkl', 'tsv')
        Returns:
            bool: True if file_path is of the specified type
        '''

        file_as_path = self.file_path if isinstance(self.file_path, Path) else Path(self.file_path) # make the file a Path object
        file_type_without_dot = file_type.lstrip('.') # remove dot if present
        return file_as_path.suffix == '.' + file_type_without_dot


    def get_data(self):
        '''
        Gets the current data. This is standard for readbility purposes, implies that data has
        been transformed or manipulated while generally attributes imply stored/static values.
        '''
        return self.data
    

    # will be helpful to get info in a "multiloaded setting" (with MultiDataLoader) since tedious to individually check attributes 
    def get_info(self): 
        '''
        Gets relevant info about the data in a dictionary.

        Currently included info: data shape and meta data
        '''
        info_dict = {
            
            'Data shape': self.data.shape,
            'Timesteps': self.timesteps,
            'File Path': self.file_path, 
            'Meta data': self.meta_data
        }
        return info_dict


class MultiDataLoader:
    '''
    Loads a list of data files with their associated meta data at once.
    '''
    def __init__(self, file_paths, meta_data_list=None):
        if meta_data_list is None: # if meta data not provided, create empty dict for each file
            meta_data_list = [{}] * len(file_paths)
        elif len(meta_data_list) != len(file_paths):
            raise ValueError('meta_data_list length must match file_paths length')
        
        self.file_paths = file_paths
        self.meta_data_list = meta_data_list
        
        self._multiload()
    
    # utilize DataLoader on each file sequentially -> attribute, loaders will hold list of DataLoader objects
    def _multiload(self):
        self.loaders = [DataLoader(file, meta) for file, meta in zip(self.file_paths, self.meta_data_list)] # note: zip makes [(file1,meta1),(file2,meta2),...]
        self.multi_data = [loader.get_data() for loader in self.loaders] # use get_data method from DataLoader to get data from the DataLoader object

    # _multiload is meant to work internally but if data changes, it can also just reinitialize the loaders and raw data
    # reskinned as reload to make it more intuitive (not recommended to reload if file_paths or meta_data_list changes <- since validation logic in __init__)
    def reload(self):
        self._multiload()

    def get_raw_loaders(self): # the list of DataLoader objects
        return self.loaders

    def get_multi_data(self): # not called all_data since only shows what was "multiloaded"
        # Note: data is put in an attribute since it will be processed and stored there (current working copy),
        # while allowing loaders to retain raw data
        return self.multi_data

    def get_multi_info(self): # same process as multi_data
        # fetches info dynamically since values are mostly static (e.g. don't need a processed copy)
        return [loader.get_info() for loader in self.loaders] 