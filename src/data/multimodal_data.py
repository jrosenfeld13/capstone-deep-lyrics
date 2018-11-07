from fastai import *
from fastai.text import *

class AudioDataset(torch.utils.data.Dataset):
    '''
    PyTorch Dataset for Aggregate Audio features from MSD
    '''
    def __init__(self, data:np.ndarray, text_link:TextDataset):
        '''
        Args
        ----
          data (2-darray) : a 2-d array of (n, features). n must be the same
                            size as the length of `text_link`
          text_link       : an instance of `TextDataset` from fastai to link
                            with. This is necessary to create a dataloader
                            of language and corresponding audio features
        '''
        # audio features
        self.data = data
        
        # linked song lyrics
        self.text_link = text_link
        assert len(self.text_link) == len(self.data),\
        "Number of examples must be the same."
        
        # return the length of each song
        self.text_length = self._get_text_length()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx:int):
        return self.data[idx]
    
    def _get_text_length(self):
        '''
        An list of size n containing the length of lyrics for each song in `text_link`
        '''
        return [len(song[0]) for song in self.text_link]
    
    @property
    def feature_size(self):
        return self.data.shape[1]

class MultimodalDataLoader(LanguageModelLoader):
    """
    A data loader for language model with bptt with augmented audio features
    """
    def __init__(self, audio_dataset, *args, **kwargs):
        super(MultimodalDataLoader, self).__init__(*args, **kwargs)
        self.audio_dataset = audio_dataset
        self.audio_data = self.batchify_audio(audio_dataset)
        
    def batchify_audio(self, audio_data):
        '''
        Repeats audio features for each word in song and batchifies
        '''
        nb = np.sum(audio_data.text_length) // self.bs # words per batch
        
        # repeat audio features
        repeated_audio = np.empty((0, self.audio_dataset.feature_size))
        for song in zip(audio_data.data, audio_data.text_length):
            features, song_length = song
            repeated_audio = np.append(repeated_audio,
                                       np.tile(features, (song_length, 1)),
                                       axis=0)
        
        # reshape to (nb * bs)
        audio = repeated_audio[:nb*self.bs]\
            .reshape(-1, self.bs, self.audio_dataset.feature_size)
        return Tensor(audio)
    
    def get_batch(self, i:int, seq_len:int) -> Tuple[LongTensor, Tensor, LongTensor]:
        "Create a batch at `i` of a given `seq_len`."
        seq_len = min(seq_len, len(self.data) - 1 - i)
        return ((self.data[i:i+seq_len],
                self.audio_data[i:i+seq_len]),
                self.data[i+1:i+1+seq_len].contiguous().view(-1))
                
class MultimodalRNNLearner():
    """
    RNN Learner extended to multimodality with aggregate audio data.
    """
    pass
