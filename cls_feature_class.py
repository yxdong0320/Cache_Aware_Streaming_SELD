import os
import numpy as np
# from IPython import embed
import matplotlib.pyplot as plot
#import librosa
plot.switch_backend('agg')
import math
import librosa
import pdb

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        # Input directories

        # Local parameters

        # self._fs = params['fs']
        self._fs = 24000
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)

        self._dataset = 'foa'
        self._eps = 1e-8
        self._nb_channels = 4

        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
        # Sound event classes dictionary

    def _load_audio(self, audio):
        fs = self._fs
        audio_transposed = audio.T
        # audio_transposed = audio_transposed[:, :self._nb_channels] / 32768.0 + self._eps
        # pdb.set_trace()
        return audio_transposed, fs

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()
    
    def _spectrogram(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T
    
    def _get_spectrogram_for_file(self, audio):
        audio_in, fs = self._load_audio(audio)
         
        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        audio_spec = self._spectrogram(audio_in, nb_feat_frames)
        return audio_spec
    
    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat
    
    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
        
        I_norm = I/E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), self._mel_wts), (0,2,1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def extract_audio_feature(self, _audio_in):
        # _file_cnt, _wav_path, _feat_path = _arg_in

        spect = self._get_spectrogram_for_file(_audio_in)

        #extract mel
        # if not self._use_salsalite:
        mel_spect = self._get_mel_spectrogram(spect)

        feat = None
        if self._dataset == 'foa':
            # extract intensity vectors
            foa_iv = self._get_foa_intensity_vectors(spect)
            feat = np.concatenate((mel_spect, foa_iv), axis=-1)

        return feat

