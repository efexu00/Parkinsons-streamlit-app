import librosa
import numpy as np

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

    target_frames = 64  # 12 × 64 = 768
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]

    return mfcc.flatten().reshape(1, -1)
