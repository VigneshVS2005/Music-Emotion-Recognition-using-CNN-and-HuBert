import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import re
import json
import datetime
import time
from collections import Counter
import psutil
import gc
from pathlib import Path
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

print("=" * 70)
print("ENHANCED MUSIC EMOTION RECOGNITION - SONG-LEVEL PIPELINE")
print("=" * 70)

class Config:
    """Configuration parameters"""
    
    # BASE PATH
    BASE_DIR = '/dist_home/suryansh/v/s6/Speech/DEAM_datasets'
    
    # AUDIO PATHS
    AUDIO_DIR = BASE_DIR + '/MEMD_audio'
    
    # AVERAGED STATIC ANNOTATIONS
    SONG_ANNOTATIONS_DIR = BASE_DIR + '/DEAM_Annotations/annotations averaged per song/song_level'
    ANNOTATIONS_FILE_1 = SONG_ANNOTATIONS_DIR + '/static_annotations_averaged_songs_1_2000.csv'
    ANNOTATIONS_FILE_2 = SONG_ANNOTATIONS_DIR + '/static_annotations_averaged_songs_2000_2058.csv'
    
    # AUDIO PARAMETERS
    SAMPLE_RATE = 16000
    MAX_AUDIO_DURATION = 45  # seconds
    
    # FEATURE EXTRACTION
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MFCC = 13  # Number of MFCC coefficients
    
    # FIXED TIME STEPS FOR MEL-SPECTROGRAM
    EXPECTED_TIME_STEPS = int((MAX_AUDIO_DURATION * SAMPLE_RATE) / HOP_LENGTH) + 1
    
    # MODEL PARAMETERS
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.3
    NUM_WORKERS = 4
    
    # DATA SIZING
    USE_ALL_AUDIO = True
    USE_PRECOMPUTED_FEATURES = False
    AUGMENT_DATA = True
    USE_SEMI_SUPERVISED = True
    
    # TRAIN/VAL/TEST SPLIT
    TEST_SIZE = 0.25
    VAL_SIZE = 0.15
    
    # OUTPUT
    OUTPUT_BASE_DIR = '/dist_home/suryansh/v/s6/Speech'
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = OUTPUT_BASE_DIR + f'/enhanced_output_{TIMESTAMP}'
    MODEL_SAVE_PATH = OUTPUT_DIR + '/best_enhanced_song_model.pth'
    LOG_FILE = OUTPUT_DIR + '/training_log.txt'
    RESULTS_JSON = OUTPUT_DIR + '/results_summary.json'
    
    # EMOTION CLASSES (8-Class Octant Model)
    EMOTION_MAP = {
        0: 'Happy/Pleased',
        1: 'Excited/Alert',
        2: 'Tense/Nervous',
        3: 'Angry/Annoyed',
        4: 'Sad/Depressed',
        5: 'Bored/Lethargic',
        6: 'Calm/Relaxed',
        7: 'Content/Serene'
    }
    
    # EMOTION COLORS
    EMOTION_COLORS = {
        0: '#FF9F1C',  # Orange
        1: '#FF4D4D',  # Bright Red
        2: '#D90429',  # Dark Red
        3: '#8338EC',  # Purple
        4: '#3A0CA3',  # Dark Blue
        5: '#4361EE',  # Blue
        6: '#4CC9F0',  # Light Blue
        7: '#2EC4B6'   # Teal
    }

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOGGING UTILITY
# ============================================================================

class Logger:
    """Enhanced logger to write to both console and file"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Redirect stdout
sys.stdout = Logger(Config.LOG_FILE)

print(f"Output directory: {Config.OUTPUT_DIR}")
print(f"Log file: {Config.LOG_FILE}")

# ============================================================================
# GPU UTILITY FUNCTIONS
# ============================================================================

def print_gpu_info():
    """Print GPU information"""
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA Available: No - Running on CPU")
    
    print(f"\nSystem RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")

print_gpu_info()

# ============================================================================
# 2. ENHANCED DATA LOADER
# ============================================================================

class EnhancedSongLevelDataLoader:
    """Enhanced data loader using averaged static annotations only"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("Initializing Enhanced Song-Level Data Loader...")
        print("="*50)
        
        self.audio_dir = Config.AUDIO_DIR
        self.annotations_file_1 = Config.ANNOTATIONS_FILE_1
        self.annotations_file_2 = Config.ANNOTATIONS_FILE_2
        self.sample_rate = Config.SAMPLE_RATE
        
        self._check_paths()
        self.song_metadata = self._load_and_process_annotations()
        self.all_audio_files = self._get_all_audio_files()
        
        print(f"\n✅ Found {len(self.all_audio_files)} total audio files")
        if len(self.song_metadata) > 0:
            print(f"✅ Loaded {len(self.song_metadata)} songs with averaged annotations")
    
    def _check_paths(self):
        """Check if all required paths exist"""
        print("\nChecking paths...")
        
        paths_to_check = [
            ("Audio directory", Config.AUDIO_DIR),
            ("Annotations file 1", Config.ANNOTATIONS_FILE_1),
            ("Annotations file 2", Config.ANNOTATIONS_FILE_2),
        ]
        
        for name, path in paths_to_check:
            exists = os.path.exists(path)
            print(f"  {name:25} {path}")
            print(f"  {'Exists:':27} {exists}")
    
    def _get_all_audio_files(self):
        """Get all MP3 files with their song IDs"""
        if not os.path.exists(self.audio_dir):
            return []
        
        audio_files = []
        for f in os.listdir(self.audio_dir):
            if f.lower().endswith('.mp3'):
                numbers = re.findall(r'\d+', f)
                if numbers:
                    song_id = int(numbers[0])
                    audio_files.append({
                        'filename': f,
                        'song_id': song_id,
                        'path': os.path.join(self.audio_dir, f)
                    })
        
        return audio_files
    
    def _load_and_process_annotations(self):
        """Load averaged static annotations from both files"""
        print("\nLoading averaged static annotations...")
        
        all_data = []
        
        # Process first file
        if os.path.exists(self.annotations_file_1):
            try:
                df1 = pd.read_csv(self.annotations_file_1)
                print(f"  ✓ Loaded file 1 with columns: {list(df1.columns)}")
                
                # Clean column names
                df1.columns = df1.columns.str.strip()
                
                # Keep only the columns we need
                required_cols = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']
                available_cols = [col for col in required_cols if col in df1.columns]
                df1 = df1[available_cols]
                
                all_data.append(df1)
                print(f"     Added {len(df1)} songs from file 1")
            except Exception as e:
                print(f"  ⚠️ Error loading file 1: {e}")
        
        # Process second file
        if os.path.exists(self.annotations_file_2):
            try:
                df2 = pd.read_csv(self.annotations_file_2)
                print(f"  ✓ Loaded file 2 with columns: {list(df2.columns)}")
                
                # Clean column names
                df2.columns = df2.columns.str.strip()
                
                # Keep only the columns we need
                required_cols = ['song_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']
                available_cols = [col for col in required_cols if col in df2.columns]
                df2 = df2[available_cols]
                
                all_data.append(df2)
                print(f"     Added {len(df2)} songs from file 2")
            except Exception as e:
                print(f"  ⚠️ Error loading file 2: {e}")
        
        if not all_data:
            print("⚠️ No annotation files found.")
            return pd.DataFrame()
        
        # Combine all data
        song_df = pd.concat(all_data, ignore_index=True)
        
        # Normalize values if they're in 1-9 range
        if song_df['valence_mean'].max() > 1.5:
            song_df['valence_mean'] = (song_df['valence_mean'] - 5) / 4
            song_df['arousal_mean'] = (song_df['arousal_mean'] - 5) / 4
            if 'valence_std' in song_df.columns:
                song_df['valence_std'] = song_df['valence_std'] / 4
                song_df['arousal_std'] = song_df['arousal_std'] / 4
        
        # Convert valence-arousal to emotion class
        song_df['emotion'] = song_df.apply(
            lambda row: self._vad_to_emotion(row['valence_mean'], row['arousal_mean']),
            axis=1
        )
        
        print(f"\n✅ Loaded {len(song_df)} songs with averaged annotations")
        print(f"   Columns: {list(song_df.columns)}")
        
        return song_df
    
    def _vad_to_emotion(self, valence, arousal):
        """Convert valence-arousal to 8 emotion classes"""
        angle_rad = np.arctan2(arousal, valence)
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360
            
        octant = int(angle_deg // 45)
        
        if octant == 8:
            octant = 7
            
        return octant
    
    def augment_audio(self, audio):
        """Create augmented versions of audio"""
        augmented = [audio]
        
        # Add Gaussian noise
        noise = np.random.randn(len(audio)) * 0.005
        augmented.append(audio + noise)
        
        try:
            # Time stretch (slower)
            stretched_slow = librosa.effects.time_stretch(audio, rate=0.9)
            if len(stretched_slow) >= self.sample_rate * 30:
                if len(stretched_slow) > len(audio):
                    stretched_slow = stretched_slow[:len(audio)]
                else:
                    stretched_slow = np.pad(stretched_slow, (0, len(audio) - len(stretched_slow)))
                augmented.append(stretched_slow)
            
            # Time stretch (faster)
            stretched_fast = librosa.effects.time_stretch(audio, rate=1.1)
            if len(stretched_fast) >= self.sample_rate * 30:
                if len(stretched_fast) > len(audio):
                    stretched_fast = stretched_fast[:len(audio)]
                else:
                    stretched_fast = np.pad(stretched_fast, (0, len(audio) - len(stretched_fast)))
                augmented.append(stretched_fast)
            
            # Pitch shift
            pitch_up = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2)
            augmented.append(pitch_up)
            
            pitch_down = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=-2)
            augmented.append(pitch_down)
        except:
            pass
        
        return augmented
    
    def prepare_dataset(self):
        """Prepare complete dataset"""
        print("\n" + "="*80)
        print("PREPARING ENHANCED DATASET")
        print("="*80)
        
        metadata_dict = {}
        if len(self.song_metadata) > 0:
            for _, row in self.song_metadata.iterrows():
                metadata_dict[row['song_id']] = row
        
        all_audio = []
        all_emotions = []
        all_valence = []
        all_arousal = []
        all_song_ids = []
        all_has_label = []
        
        print(f"Processing {len(self.all_audio_files)} audio files...")
        start_time = time.time()
        
        for i, audio_info in enumerate(self.all_audio_files):
            song_id = audio_info['song_id']
            
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i}/{len(self.all_audio_files)} files... ({elapsed:.1f}s elapsed)")
            
            try:
                audio, sr = librosa.load(
                    audio_info['path'],
                    sr=self.sample_rate,
                    mono=True,
                    duration=Config.MAX_AUDIO_DURATION
                )
            except Exception as e:
                print(f"  Error loading {audio_info['filename']}: {e}")
                continue
            
            has_label = song_id in metadata_dict
            
            if has_label:
                meta = metadata_dict[song_id]
                
                all_audio.append(audio)
                all_emotions.append(meta['emotion'])
                all_valence.append(meta['valence_mean'])
                all_arousal.append(meta['arousal_mean'])
                all_song_ids.append(song_id)
                all_has_label.append(True)
                
                # Add augmented versions
                if Config.AUGMENT_DATA:
                    augmented_audios = self.augment_audio(audio)
                    for aug_audio in augmented_audios[1:]:
                        all_audio.append(aug_audio)
                        all_emotions.append(meta['emotion'])
                        all_valence.append(meta['valence_mean'])
                        all_arousal.append(meta['arousal_mean'])
                        all_song_ids.append(f"{song_id}_aug")
                        all_has_label.append(True)
            
            elif Config.USE_ALL_AUDIO and Config.USE_SEMI_SUPERVISED:
                all_audio.append(audio)
                all_emotions.append(-1)
                all_valence.append(0)
                all_arousal.append(0)
                all_song_ids.append(song_id)
                all_has_label.append(False)
        
        if len(all_audio) == 0:
            print("❌ ERROR: No audio files loaded!")
            sys.exit(1)
        
        # Convert to numpy arrays
        all_emotions = np.array(all_emotions, dtype=np.int64)
        all_valence = np.array(all_valence, dtype=np.float32)
        all_arousal = np.array(all_arousal, dtype=np.float32)
        all_song_ids = np.array(all_song_ids)
        all_has_label = np.array(all_has_label, dtype=bool)
        
        print(f"\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(f"✅ Total samples: {len(all_audio)}")
        print(f"   - Labeled: {np.sum(all_has_label)}")
        print(f"   - Unlabeled: {len(all_audio) - np.sum(all_has_label)}")
        
        if np.sum(all_has_label) > 0:
            labeled_emotions = all_emotions[all_has_label]
            unique_labels, counts = np.unique(labeled_emotions, return_counts=True)
            print(f"\n📊 LABELED CLASS DISTRIBUTION:")
            for label, count in zip(unique_labels, counts):
                emotion_name = Config.EMOTION_MAP[label]
                percentage = 100 * count / len(labeled_emotions)
                print(f"   {emotion_name:20} {count:4d} samples ({percentage:5.1f}%)")
            
            # Save class distribution
            class_dist_df = pd.DataFrame({
                'emotion_id': unique_labels,
                'emotion_name': [Config.EMOTION_MAP[l] for l in unique_labels],
                'count': counts,
                'percentage': [100 * c / len(labeled_emotions) for c in counts]
            })
            class_dist_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'class_distribution.csv'), index=False)
        
        return (all_audio, all_emotions, all_valence, all_arousal, 
                all_song_ids, all_has_label)

# ============================================================================
# 3. ENHANCED FEATURE EXTRACTOR - WITH MFCC
# ============================================================================

class EnhancedFeatureExtractor:
    """Extracts enhanced features from audio including MFCC"""
    
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.n_mels = Config.N_MELS
        self.n_fft = Config.N_FFT
        self.hop_length = Config.HOP_LENGTH
        self.n_mfcc = Config.N_MFCC
        self.expected_time_steps = Config.EXPECTED_TIME_STEPS
        
        self.hubert_processor = None
        self.hubert_model = None
        
        print("\nEnhanced Feature Extractor initialized")
        print(f"Expected time steps for mel-spectrogram: {self.expected_time_steps}")
        print(f"Number of MFCC coefficients: {self.n_mfcc}")
        self._init_hubert()
    
    def _init_hubert(self):
        """Initialize HuBERT model"""
        if self.hubert_processor is None:
            try:
                print("Loading HuBERT model...")
                from transformers import Wav2Vec2FeatureExtractor, HubertModel
                
                self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/hubert-base-ls960"
                )
                self.hubert_model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960"
                )
                
                if torch.cuda.is_available():
                    self.hubert_model = self.hubert_model.cuda()
                
                self.hubert_model.eval()
                print("✅ HuBERT model loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Could not load HuBERT: {e}")
    
    def extract_mel_spectrogram(self, audio):
        """Extract Mel-spectrogram with fixed time dimension"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0
        )
        
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        
        # Ensure fixed time dimension
        current_time_steps = log_mel.shape[1]
        
        if current_time_steps > self.expected_time_steps:
            log_mel = log_mel[:, :self.expected_time_steps]
        elif current_time_steps < self.expected_time_steps:
            pad_width = self.expected_time_steps - current_time_steps
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        log_mel = np.expand_dims(log_mel, axis=0)
        
        return log_mel
    
    def extract_mfcc(self, audio):
        """Extract MFCC features with fixed time dimension"""
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Normalize MFCCs
        mfccs = (mfccs - mfccs.mean(axis=1, keepdims=True)) / (mfccs.std(axis=1, keepdims=True) + 1e-8)
        
        # Ensure fixed time dimension
        current_time_steps = mfccs.shape[1]
        
        if current_time_steps > self.expected_time_steps:
            mfccs = mfccs[:, :self.expected_time_steps]
        elif current_time_steps < self.expected_time_steps:
            pad_width = self.expected_time_steps - current_time_steps
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        # Add channel dimension
        mfccs = np.expand_dims(mfccs, axis=0)
        
        return mfccs
    
    def extract_hubert_features(self, audio):
        """Extract HuBERT embeddings"""
        if self.hubert_model is None:
            return np.zeros(768, dtype=np.float32)
        
        audio_input = audio.astype(np.float32)
        
        try:
            inputs = self.hubert_processor(
                audio_input,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.hubert_model(**inputs)
                features = torch.mean(outputs.last_hidden_state, dim=1)
            
            return features.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting HuBERT features: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def extract_all_features(self, audio):
        """Extract all features (Mel-spectrogram, MFCC, HuBERT)"""
        spectral = self.extract_mel_spectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        hubert = self.extract_hubert_features(audio)
        
        return spectral, mfcc, hubert

# ============================================================================
# 4. ENHANCED MODEL ARCHITECTURES - WITH MFCC BRANCH
# ============================================================================

class EnhancedCNNBranch(nn.Module):
    """Enhanced CNN branch for Mel-spectrograms"""
    
    def __init__(self, input_channels=1):
        super(EnhancedCNNBranch, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(Config.DROPOUT_RATE)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(Config.DROPOUT_RATE)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        return x


class EnhancedMFCCBranch(nn.Module):
    """CNN branch for MFCC features"""
    
    def __init__(self, input_channels=1, n_mfcc=13):
        super(EnhancedMFCCBranch, self).__init__()
        
        # 1D CNN for MFCC temporal features
        self.conv1d_1 = nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2)
        self.bn1d_1 = nn.BatchNorm1d(64)
        self.pool1d_1 = nn.MaxPool1d(2)
        
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn1d_2 = nn.BatchNorm1d(128)
        self.pool1d_2 = nn.MaxPool1d(2)
        
        self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1d_3 = nn.BatchNorm1d(256)
        self.pool1d_3 = nn.MaxPool1d(2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(Config.DROPOUT_RATE)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # x shape: (batch, 1, n_mfcc, time)
        x = x.squeeze(1)  # Remove channel dim: (batch, n_mfcc, time)
        
        x = self.pool1d_1(F.relu(self.bn1d_1(self.conv1d_1(x))))
        x = self.pool1d_2(F.relu(self.bn1d_2(self.conv1d_2(x))))
        x = self.pool1d_3(F.relu(self.bn1d_3(self.conv1d_3(x))))
        
        x = self.global_pool(x).squeeze(-1)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        
        return x


class EnhancedHuBERTBranch(nn.Module):
    """MLP branch for HuBERT embeddings"""
    
    def __init__(self, input_dim=768):
        super(EnhancedHuBERTBranch, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(Config.DROPOUT_RATE)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(Config.DROPOUT_RATE)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        
        return x


class EnhancedFusionModel(nn.Module):
    """Fusion model with Mel-spectrogram CNN + MFCC CNN + HuBERT MLP"""
    
    def __init__(self, num_classes=8):
        super(EnhancedFusionModel, self).__init__()
        
        self.mel_branch = EnhancedCNNBranch(input_channels=1)      # Output: 256
        self.mfcc_branch = EnhancedMFCCBranch(n_mfcc=Config.N_MFCC)  # Output: 64
        self.hubert_branch = EnhancedHuBERTBranch()                # Output: 128
        
        fusion_dim = 256 + 64 + 128  # 448
        
        self.fusion_fc1 = nn.Linear(fusion_dim, 256)
        self.bn_fusion1 = nn.BatchNorm1d(256)
        self.dropout_fusion1 = nn.Dropout(Config.DROPOUT_RATE)
        
        self.fusion_fc2 = nn.Linear(256, 128)
        self.bn_fusion2 = nn.BatchNorm1d(128)
        self.dropout_fusion2 = nn.Dropout(Config.DROPOUT_RATE)
        
        self.fusion_fc3 = nn.Linear(128, 64)
        self.bn_fusion3 = nn.BatchNorm1d(64)
        
        self.classifier = nn.Linear(64, num_classes)
        
        print(f"\nEnhanced Fusion Model Initialized (with MFCC)")
        print(f"  Mel-spectrogram CNN: 256 dim")
        print(f"  MFCC CNN: 64 dim")
        print(f"  HuBERT MLP: 128 dim")
        print(f"  Total fusion dim: {fusion_dim}")
    
    def forward(self, mel_spectral, mfcc, hubert):
        mel_out = self.mel_branch(mel_spectral)
        mfcc_out = self.mfcc_branch(mfcc)
        hubert_out = self.hubert_branch(hubert)
        
        fused = torch.cat([mel_out, mfcc_out, hubert_out], dim=1)
        
        x = F.relu(self.bn_fusion1(self.fusion_fc1(fused)))
        x = self.dropout_fusion1(x)
        x = F.relu(self.bn_fusion2(self.fusion_fc2(x)))
        x = self.dropout_fusion2(x)
        x = F.relu(self.bn_fusion3(self.fusion_fc3(x)))
        
        output = self.classifier(x)
        
        return output
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# ============================================================================
# 5. ENHANCED DATASET CLASS - WITH MFCC
# ============================================================================

class EnhancedSongDataset(Dataset):
    """Enhanced dataset with Mel-spectrogram, MFCC, and HuBERT features"""
    
    def __init__(self, audio_list, labels, feature_extractor,
                 has_label=None, extract_features=True):
        
        self.audio_list = audio_list
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.has_label = has_label if has_label is not None else np.ones(len(labels), dtype=bool)
        
        if extract_features:
            print(f"\nExtracting features for {len(audio_list)} samples...")
            start_time = time.time()
            
            self.mel_features = []
            self.mfcc_features = []
            self.hubert_features = []
            
            for i, audio in enumerate(audio_list):
                mel, mfcc, hubert = feature_extractor.extract_all_features(audio)
                self.mel_features.append(mel)
                self.mfcc_features.append(mfcc)
                self.hubert_features.append(hubert)
                
                if i % 50 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    progress = (i / len(audio_list)) * 100
                    eta = (elapsed / i) * (len(audio_list) - i)
                    print(f"  Extracted {i}/{len(audio_list)} ({progress:.1f}%) - "
                          f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            
            self.mel_features = np.array(self.mel_features, dtype=np.float32)
            self.mfcc_features = np.array(self.mfcc_features, dtype=np.float32)
            self.hubert_features = np.array(self.hubert_features, dtype=np.float32)
            
            total_time = time.time() - start_time
            print(f"✅ Feature extraction complete in {total_time:.1f}s")
            print(f"   Mel-spectrograms: {self.mel_features.shape}")
            print(f"   MFCC features: {self.mfcc_features.shape}")
            print(f"   HuBERT features: {self.hubert_features.shape}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mel = self.mel_features[idx]
        mfcc = self.mfcc_features[idx]
        hubert = self.hubert_features[idx]
        label = self.labels[idx]
        has_label = self.has_label[idx]
        
        mel_tensor = torch.FloatTensor(mel)
        mfcc_tensor = torch.FloatTensor(mfcc)
        hubert_tensor = torch.FloatTensor(hubert)
        
        return mel_tensor, mfcc_tensor, hubert_tensor, label, has_label

# ============================================================================
# 6. ENHANCED TRAINER - WITH MFCC
# ============================================================================

class EnhancedTrainer:
    """Trainer with MFCC features"""
    
    def __init__(self, model, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        print(f"\nTraining on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def train_epoch(self, dataloader, optimizer, epoch_idx, use_consistency=True):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_sup_loss = 0
        total_consistency_loss = 0
        correct = 0
        total = 0
        
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            mel, mfcc, hubert, labels, has_label = batch
            mel = mel.to(self.device)
            mfcc = mfcc.to(self.device)
            hubert = hubert.to(self.device)
            labels = labels.to(self.device)
            has_label = has_label.to(self.device)
            
            outputs = self.model(mel, mfcc, hubert)
            
            if has_label.any():
                sup_loss = self.criterion(outputs[has_label], labels[has_label]).mean()
            else:
                sup_loss = torch.tensor(0.0).to(self.device)
            
            if use_consistency and (~has_label).any():
                noise_mel = torch.randn_like(mel[~has_label]) * 0.01
                noise_mfcc = torch.randn_like(mfcc[~has_label]) * 0.01
                mel_noisy = mel[~has_label] + noise_mel
                mfcc_noisy = mfcc[~has_label] + noise_mfcc
                
                outputs_noisy = self.model(mel_noisy, mfcc_noisy, hubert[~has_label])
                
                with torch.no_grad():
                    outputs_clean = outputs[~has_label]
                
                consistency_loss = F.mse_loss(outputs_noisy, outputs_clean)
            else:
                consistency_loss = torch.tensor(0.0).to(self.device)
            
            loss = sup_loss + 0.1 * consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            if has_label.any():
                total_sup_loss += sup_loss.item()
                _, predicted = outputs[has_label].max(1)
                total += has_label.sum().item()
                correct += predicted.eq(labels[has_label]).sum().item()
            
            if use_consistency:
                total_consistency_loss += consistency_loss.item()
            
            if batch_idx % 10 == 0:
                batch_time = time.time() - batch_start_time
                print(f"    Batch {batch_idx}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f} - "
                      f"Time: {batch_time:.2f}s")
                batch_start_time = time.time()
        
        stats = {
            'total_loss': total_loss / len(dataloader),
            'sup_loss': total_sup_loss / len(dataloader) if total_sup_loss > 0 else 0,
            'consistency_loss': total_consistency_loss / len(dataloader) if total_consistency_loss > 0 else 0,
            'accuracy': 100. * correct / total if total > 0 else 0
        }
        
        return stats
    
    def evaluate(self, dataloader, return_probs=False):
        """Evaluate on labeled data"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                mel, mfcc, hubert, labels, has_label = batch
                mel = mel.to(self.device)
                mfcc = mfcc.to(self.device)
                hubert = hubert.to(self.device)
                labels = labels.to(self.device)
                has_label = has_label.to(self.device)
                
                if not has_label.any():
                    continue
                
                outputs = self.model(mel[has_label], mfcc[has_label], hubert[has_label])
                loss = self.criterion(outputs, labels[has_label]).mean()
                
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = probs.max(1)
                total += labels[has_label].size(0)
                correct += predicted.eq(labels[has_label]).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels[has_label].cpu().numpy())
                if return_probs:
                    all_probs.extend(probs.cpu().numpy())
        
        if total == 0:
            return 0, 0, [], [], [] if return_probs else (0, 0, [], [])
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        if return_probs:
            return avg_loss, accuracy, all_preds, all_labels, all_probs
        else:
            return avg_loss, accuracy, all_preds, all_labels

# ============================================================================
# 7. ENHANCED VISUALIZATION - WITH MFCC PLOTS
# ============================================================================

class EnhancedVisualizer:
    """Enhanced visualization utilities with MFCC plots"""
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training')
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation')
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, history['learning_rate'], 'm-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time
        if 'epoch_time' in history:
            axes[1, 1].plot(epochs, history['epoch_time'], 'c-', linewidth=2)
            axes[1, 1].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Training history plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix_absolute(y_true, y_pred, save_path=None):
        """Plot confusion matrix (absolute counts)"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(Config.EMOTION_MAP.values()),
                    yticklabels=list(Config.EMOTION_MAP.values()))
        plt.title('Confusion Matrix (Absolute Counts)', fontsize=16, fontweight='bold')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion matrix (absolute) saved")
        
        plt.show()
        plt.close()
        
        return cm
    
    @staticmethod
    def plot_confusion_matrix_normalized(y_true, y_pred, save_path=None):
        """Plot confusion matrix (normalized)"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=list(Config.EMOTION_MAP.values()),
                    yticklabels=list(Config.EMOTION_MAP.values()))
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion matrix (normalized) saved")
        
        plt.show()
        plt.close()
        
        return cm_normalized
    
    @staticmethod
    def plot_class_performance_bar(y_true, y_pred, save_path=None):
        """Plot class-wise performance bar chart"""
        report = classification_report(y_true, y_pred, output_dict=True)
        
        classes = list(Config.EMOTION_MAP.values())
        precision = [report[str(i)]['precision'] for i in range(len(classes))]
        recall = [report[str(i)]['recall'] for i in range(len(classes))]
        f1_score = [report[str(i)]['f1-score'] for i in range(len(classes))]
        
        plt.figure(figsize=(14, 8))
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='navy', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', color='lightcoral', edgecolor='darkred', alpha=0.8)
        plt.bar(x + width, f1_score, width, label='F1-Score', color='lightgreen', edgecolor='darkgreen', alpha=0.8)
        
        plt.xlabel('Emotion Classes', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Class-wise Performance Metrics', fontsize=16, fontweight='bold')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Class performance bar chart saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_class_performance_heatmap(y_true, y_pred, save_path=None):
        """Plot class-wise performance heatmap"""
        report = classification_report(y_true, y_pred, output_dict=True)
        
        classes = list(Config.EMOTION_MAP.values())
        precision = [report[str(i)]['precision'] for i in range(len(classes))]
        recall = [report[str(i)]['recall'] for i in range(len(classes))]
        f1_score = [report[str(i)]['f1-score'] for i in range(len(classes))]
        
        metrics_matrix = np.array([precision, recall, f1_score])
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=classes, yticklabels=['Precision', 'Recall', 'F1-Score'],
                    cbar_kws={'label': 'Score'})
        plt.title('Performance Metrics Heatmap', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Class performance heatmap saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_class_support(y_true, save_path=None):
        """Plot class support distribution"""
        class_counts = Counter(y_true)
        classes = [Config.EMOTION_MAP[i] for i in sorted(class_counts.keys())]
        counts = [class_counts[i] for i in sorted(class_counts.keys())]
        colors = [Config.EMOTION_COLORS[i] for i in sorted(class_counts.keys())]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(classes, counts, color=colors, edgecolor='black', alpha=0.8)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Emotion Classes', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Class support chart saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_va_scatter(valence, arousal, emotions, save_path=None):
        """Plot valence-arousal scatter plot"""
        plt.figure(figsize=(14, 10))
        
        colors = [Config.EMOTION_COLORS[e] for e in emotions]
        scatter = plt.scatter(valence, arousal, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('Valence', fontsize=14)
        plt.ylabel('Arousal', fontsize=14)
        plt.title('Valence-Arousal Distribution by Emotion', fontsize=16, fontweight='bold')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.grid(True, alpha=0.3)
        
        # Add quadrant labels
        plt.text(0.6, 0.6, 'Q1: High Arousal\nHigh Valence', ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.text(-0.6, 0.6, 'Q2: High Arousal\nLow Valence', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.text(-0.6, -0.6, 'Q3: Low Arousal\nLow Valence', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.text(0.6, -0.6, 'Q4: Low Arousal\nHigh Valence', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Create legend
        legend_elements = []
        for emotion_id, emotion_name in Config.EMOTION_MAP.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=Config.EMOTION_COLORS[emotion_id],
                                             markersize=10, label=emotion_name))
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ VA scatter plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_va_density(valence, arousal, save_path=None):
        """Plot valence-arousal density map"""
        plt.figure(figsize=(12, 10))
        
        h = plt.hist2d(valence, arousal, bins=40, cmap='viridis', range=[[-1, 1], [-1, 1]])
        plt.colorbar(h[3], label='Number of Samples')
        
        plt.xlabel('Valence', fontsize=14)
        plt.ylabel('Arousal', fontsize=14)
        plt.title('Valence-Arousal Density Map', fontsize=16, fontweight='bold')
        plt.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        plt.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        plt.grid(False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ VA density map saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_model_architecture(model, save_path=None):
        """Plot model architecture summary"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Parameter distribution
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        param_data = {
            'Total': total_params,
            'Trainable': trainable_params,
            'Non-trainable': total_params - trainable_params
        }
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = axes[0].bar(param_data.keys(), param_data.values(), color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, (k, v) in zip(bars, param_data.items()):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                        f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        axes[0].set_title('Model Parameter Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Parameters')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Branch contribution
        branches = ['Mel-Spectrogram CNN', 'MFCC CNN', 'HuBERT MLP']
        branch_params = [
            sum(p.numel() for p in model.mel_branch.parameters()),
            sum(p.numel() for p in model.mfcc_branch.parameters()),
            sum(p.numel() for p in model.hubert_branch.parameters())
        ]
        
        wedges, texts, autotexts = axes[1].pie(branch_params, labels=branches, autopct='%1.1f%%',
                                               colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90,
                                               textprops={'fontsize': 11, 'fontweight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        axes[1].set_title('Parameter Distribution by Branch', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Model Architecture Summary\nTotal Parameters: {total_params:,}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Model architecture plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_prediction_distribution(y_true, y_pred, save_path=None):
        """Plot distribution of predictions vs true labels"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        all_classes = sorted(set(y_true) | set(y_pred))
        class_names = [Config.EMOTION_MAP[i] for i in all_classes]
        
        # True distribution
        true_values = [true_counts.get(i, 0) for i in all_classes]
        true_colors = [Config.EMOTION_COLORS[i] for i in all_classes]
        
        bars1 = axes[0].bar(class_names, true_values, color=true_colors, edgecolor='black', alpha=0.8)
        axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Emotion')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars1, true_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val}', ha='center', va='bottom', fontweight='bold')
        
        # Predicted distribution
        pred_values = [pred_counts.get(i, 0) for i in all_classes]
        pred_colors = [Config.EMOTION_COLORS[i] for i in all_classes]
        
        bars2 = axes[1].bar(class_names, pred_values, color=pred_colors, edgecolor='black', alpha=0.8)
        axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Emotion')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars2, pred_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val}', ha='center', va='bottom', fontweight='bold')
        
        # Difference
        diff = [pred_counts.get(i, 0) - true_counts.get(i, 0) for i in all_classes]
        diff_colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]
        
        bars3 = axes[2].bar(class_names, diff, color=diff_colors, edgecolor='black', alpha=0.8)
        axes[2].set_title('Prediction Difference (Pred - True)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Emotion')
        axes[2].set_ylabel('Difference')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, d in zip(bars3, diff):
            y_pos = bar.get_height() + 2 if d >= 0 else bar.get_height() - 8
            va = 'bottom' if d >= 0 else 'top'
            axes[2].text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{d:+d}', ha='center', va=va, fontweight='bold')
        
        plt.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Prediction distribution plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_mfcc_sample(mfcc_features, save_path=None):
        """Plot a sample MFCC heatmap"""
        if len(mfcc_features) == 0:
            return
        
        # Take a random sample
        idx = np.random.randint(0, len(mfcc_features))
        mfcc_sample = mfcc_features[idx].squeeze()  # Remove channel dimension
        
        plt.figure(figsize=(14, 8))
        librosa.display.specshow(mfcc_sample, x_axis='time', sr=Config.SAMPLE_RATE,
                                 hop_length=Config.HOP_LENGTH, cmap='viridis')
        plt.colorbar(format='%+2.0f')
        plt.title('MFCC Features - Sample Audio', fontsize=16, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ MFCC sample plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_mfcc_statistics(mfcc_features, save_path=None):
        """Plot MFCC statistics (mean and std per coefficient)"""
        if len(mfcc_features) == 0:
            return
        
        # Compute statistics across all samples
        mfcc_flat = mfcc_features.reshape(mfcc_features.shape[0], -1)
        mfcc_mean = np.mean(mfcc_flat, axis=0)
        mfcc_std = np.std(mfcc_flat, axis=0)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Mean plot
        axes[0].plot(mfcc_mean, 'b-', linewidth=2)
        axes[0].fill_between(range(len(mfcc_mean)), 
                             mfcc_mean - mfcc_std, 
                             mfcc_mean + mfcc_std, 
                             alpha=0.3, color='blue')
        axes[0].set_title('MFCC Mean Values Across All Samples', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('MFCC Coefficient Index')
        axes[0].set_ylabel('Mean Value')
        axes[0].grid(True, alpha=0.3)
        
        # Heatmap of MFCC coefficients across samples
        # Take a subset if too many samples
        if len(mfcc_features) > 100:
            indices = np.random.choice(len(mfcc_features), 100, replace=False)
            mfcc_subset = mfcc_features[indices]
        else:
            mfcc_subset = mfcc_features
        
        mfcc_reshaped = mfcc_subset.reshape(mfcc_subset.shape[0], -1)
        
        im = axes[1].imshow(mfcc_reshaped, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1].set_title('MFCC Feature Matrix (Samples × Coefficients)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('MFCC Coefficient Index')
        axes[1].set_ylabel('Sample Index')
        plt.colorbar(im, ax=axes[1], label='Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ MFCC statistics plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_mfcc_by_emotion(mfcc_features, labels, save_path=None):
        """Plot MFCC patterns for each emotion class"""
        if len(mfcc_features) == 0:
            return
        
        unique_emotions = np.unique(labels)
        n_emotions = len(unique_emotions)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, emotion in enumerate(sorted(unique_emotions)):
            if i >= 8:  # Max 8 emotions
                break
                
            # Get samples for this emotion
            emotion_mask = labels == emotion
            emotion_mfcc = mfcc_features[emotion_mask]
            
            if len(emotion_mfcc) == 0:
                continue
            
            # Compute mean MFCC for this emotion
            mfcc_mean = np.mean(emotion_mfcc, axis=0).squeeze()
            
            # Plot
            im = axes[i].imshow(mfcc_mean, aspect='auto', cmap='viridis', 
                                interpolation='nearest')
            axes[i].set_title(f'{Config.EMOTION_MAP[emotion]}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Time Frames')
            axes[i].set_ylabel('MFCC Coeff')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_emotions, 8):
            axes[i].set_visible(False)
        
        plt.suptitle('Average MFCC Patterns by Emotion', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ MFCC by emotion plot saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_tsne_visualization(features, labels, save_path=None):
        """Plot t-SNE visualization"""
        if len(features) > 1000:
            indices = np.random.choice(len(features), 1000, replace=False)
            features_viz = features[indices]
            labels_viz = labels[indices]
        else:
            features_viz = features
            labels_viz = labels
        
        plt.figure(figsize=(14, 10))
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_tsne = tsne.fit_transform(features_viz)
        
        colors = [Config.EMOTION_COLORS[l] for l in labels_viz]
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        plt.title('t-SNE Visualization of Learned Features', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = []
        for emotion_id, emotion_name in Config.EMOTION_MAP.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=Config.EMOTION_COLORS[emotion_id],
                                             markersize=10, label=emotion_name))
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ t-SNE visualization saved")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_pca_visualization(features, labels, save_path=None):
        """Plot PCA visualization"""
        if len(features) > 1000:
            indices = np.random.choice(len(features), 1000, replace=False)
            features_viz = features[indices]
            labels_viz = labels[indices]
        else:
            features_viz = features
            labels_viz = labels
        
        plt.figure(figsize=(14, 10))
        
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_viz)
        
        colors = [Config.EMOTION_COLORS[l] for l in labels_viz]
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        explained_var = pca.explained_variance_ratio_.sum()
        plt.title(f'PCA Visualization of Learned Features\n(Explained Variance: {explained_var:.2%})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = []
        for emotion_id, emotion_name in Config.EMOTION_MAP.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=Config.EMOTION_COLORS[emotion_id],
                                             markersize=10, label=emotion_name))
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ PCA visualization saved")
        
        plt.show()
        plt.close()

# ============================================================================
# 8. MAIN ENHANCED PIPELINE - WITH MFCC
# ============================================================================

def run_enhanced_pipeline():
    """Run the enhanced pipeline with MFCC features"""
    
    print("\n" + "="*80)
    print("🎵 STARTING ENHANCED SONG-LEVEL PIPELINE (WITH MFCC)")
    print("="*80)
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Timestamp: {Config.TIMESTAMP}")
    
    pipeline_start_time = time.time()
    
    # ========================================================================
    # STEP 1: LOAD ALL DATA
    # ========================================================================
    print("\n[STEP 1] LOADING ALL AVAILABLE DATA")
    print("-" * 40)
    
    data_loader = EnhancedSongLevelDataLoader()
    
    result = data_loader.prepare_dataset()
    
    if result[0] is None:
        print("❌ ERROR: No data loaded. Exiting.")
        return None
    
    (audio_list, emotions, valence, arousal, song_ids, has_label) = result
    
    print(f"\n✅ Loaded {len(audio_list)} total samples")
    print(f"   - Labeled: {np.sum(has_label)}")
    print(f"   - Unlabeled: {len(audio_list) - np.sum(has_label)}")
    
    # ========================================================================
    # STEP 2: SPLIT DATA
    # ========================================================================
    print("\n[STEP 2] SPLITTING LABELED DATA")
    print("-" * 40)
    
    labeled_indices = np.where(has_label)[0]
    labeled_emotions = emotions[labeled_indices]
    
    train_val_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=Config.TEST_SIZE,
        random_state=42,
        stratify=labeled_emotions
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=Config.VAL_SIZE/(1-Config.TEST_SIZE),
        random_state=42,
        stratify=emotions[train_val_idx]
    )
    
    if Config.USE_SEMI_SUPERVISED:
        unlabeled_indices = np.where(~has_label)[0]
        train_idx = np.concatenate([train_idx, unlabeled_indices])
        print(f"Added {len(unlabeled_indices)} unlabeled samples to training")
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    # ========================================================================
    # STEP 3: INITIALIZE FEATURE EXTRACTOR
    # ========================================================================
    print("\n[STEP 3] INITIALIZING FEATURE EXTRACTOR")
    print("-" * 40)
    
    feature_extractor = EnhancedFeatureExtractor()
    
    # ========================================================================
    # STEP 4: CREATE DATASETS
    # ========================================================================
    print("\n[STEP 4] CREATING DATASETS")
    print("-" * 40)
    
    train_audio = [audio_list[i] for i in train_idx]
    train_labels = emotions[train_idx]
    train_has_label = has_label[train_idx]
    
    val_audio = [audio_list[i] for i in val_idx]
    val_labels = emotions[val_idx]
    val_has_label = has_label[val_idx]
    
    test_audio = [audio_list[i] for i in test_idx]
    test_labels = emotions[test_idx]
    test_has_label = has_label[test_idx]
    
    print("Creating training dataset...")
    train_dataset = EnhancedSongDataset(
        train_audio, train_labels, feature_extractor,
        has_label=train_has_label,
        extract_features=True
    )
    
    print("\nCreating validation dataset...")
    val_dataset = EnhancedSongDataset(
        val_audio, val_labels, feature_extractor,
        has_label=val_has_label,
        extract_features=True
    )
    
    print("\nCreating test dataset...")
    test_dataset = EnhancedSongDataset(
        test_audio, test_labels, feature_extractor,
        has_label=test_has_label,
        extract_features=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    
    # ========================================================================
    # STEP 5: INITIALIZE MODEL
    # ========================================================================
    print("\n[STEP 5] INITIALIZING ENHANCED MODEL")
    print("-" * 40)
    
    num_classes = len(Config.EMOTION_MAP)
    model = EnhancedFusionModel(num_classes=num_classes)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # STEP 6: TRAIN MODEL
    # ========================================================================
    print("\n[STEP 6] TRAINING ENHANCED MODEL")
    print("-" * 40)
    
    trainer = EnhancedTrainer(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    history = {
        'train_loss': [], 'train_acc': [], 'sup_loss': [],
        'val_loss': [], 'val_acc': [], 'learning_rate': [],
        'epoch_time': [], 'gpu_memory': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    
    for epoch in range(Config.EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print('='*50)
        
        train_stats = trainer.train_epoch(train_loader, optimizer, epoch, use_consistency=True)
        history['train_loss'].append(train_stats['total_loss'])
        history['train_acc'].append(train_stats['accuracy'])
        history['sup_loss'].append(train_stats['sup_loss'])
        
        val_loss, val_acc, _, _ = trainer.evaluate(val_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_time)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            history['gpu_memory'].append(gpu_memory)
        
        print(f"\nTrain Loss: {train_stats['total_loss']:.4f} (Sup: {train_stats['sup_loss']:.4f})")
        print(f"Train Acc: {train_stats['accuracy']:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {gpu_memory:.2f} GB")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, Config.MODEL_SAVE_PATH)
            print(f"\n  🎯 New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (Val Acc: {best_val_acc:.2f}%)")
    
    # ========================================================================
    # STEP 7: EVALUATE MODEL
    # ========================================================================
    print("\n[STEP 7] EVALUATING MODEL")
    print("-" * 40)
    
    test_loss, test_acc, test_preds, test_labels, test_probs = trainer.evaluate(test_loader, return_probs=True)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    
    print("\n📈 CLASSIFICATION REPORT:")
    report = classification_report(
        test_labels,
        test_preds,
        target_names=list(Config.EMOTION_MAP.values()),
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        test_labels,
        test_preds,
        target_names=list(Config.EMOTION_MAP.values()),
        zero_division=0
    ))
    
    # ========================================================================
    # STEP 8: VISUALIZE RESULTS - Each plot saved separately (including MFCC)
    # ========================================================================
    print("\n[STEP 8] VISUALIZING RESULTS")
    print("-" * 40)
    
    # 1. Training history plot
    EnhancedVisualizer.plot_training_history(
        history,
        save_path=os.path.join(Config.OUTPUT_DIR, '01_training_history.png')
    )
    
    # 2. Confusion matrix absolute
    EnhancedVisualizer.plot_confusion_matrix_absolute(
        test_labels,
        test_preds,
        save_path=os.path.join(Config.OUTPUT_DIR, '02_confusion_matrix_absolute.png')
    )
    
    # 3. Confusion matrix normalized
    EnhancedVisualizer.plot_confusion_matrix_normalized(
        test_labels,
        test_preds,
        save_path=os.path.join(Config.OUTPUT_DIR, '03_confusion_matrix_normalized.png')
    )
    
    # 4. Class performance bar chart
    EnhancedVisualizer.plot_class_performance_bar(
        test_labels,
        test_preds,
        save_path=os.path.join(Config.OUTPUT_DIR, '04_class_performance_bar.png')
    )
    
    # 5. Class performance heatmap
    EnhancedVisualizer.plot_class_performance_heatmap(
        test_labels,
        test_preds,
        save_path=os.path.join(Config.OUTPUT_DIR, '05_class_performance_heatmap.png')
    )
    
    # 6. Class support distribution
    EnhancedVisualizer.plot_class_support(
        test_labels,
        save_path=os.path.join(Config.OUTPUT_DIR, '06_class_support.png')
    )
    
    # 7. VA scatter plot
    EnhancedVisualizer.plot_va_scatter(
        valence[labeled_indices],
        arousal[labeled_indices],
        emotions[labeled_indices],
        save_path=os.path.join(Config.OUTPUT_DIR, '07_va_scatter.png')
    )
    
    # 8. VA density map
    EnhancedVisualizer.plot_va_density(
        valence[labeled_indices],
        arousal[labeled_indices],
        save_path=os.path.join(Config.OUTPUT_DIR, '08_va_density.png')
    )
    
    # 9. Model architecture plot
    EnhancedVisualizer.plot_model_architecture(
        model,
        save_path=os.path.join(Config.OUTPUT_DIR, '09_model_architecture.png')
    )
    
    # 10. Prediction distribution
    EnhancedVisualizer.plot_prediction_distribution(
        test_labels,
        test_preds,
        save_path=os.path.join(Config.OUTPUT_DIR, '10_prediction_distribution.png')
    )
    
    # 11. MFCC sample plot
    EnhancedVisualizer.plot_mfcc_sample(
        test_dataset.mfcc_features,
        save_path=os.path.join(Config.OUTPUT_DIR, '11_mfcc_sample.png')
    )
    
    # 12. MFCC statistics plot
    EnhancedVisualizer.plot_mfcc_statistics(
        test_dataset.mfcc_features,
        save_path=os.path.join(Config.OUTPUT_DIR, '12_mfcc_statistics.png')
    )
    
    # 13. MFCC by emotion plot
    EnhancedVisualizer.plot_mfcc_by_emotion(
        test_dataset.mfcc_features,
        test_labels,
        save_path=os.path.join(Config.OUTPUT_DIR, '13_mfcc_by_emotion.png')
    )
    
    # 14-15. Feature embedding visualizations
    # Collect features for visualization
    all_features = []
    all_labels_viz = []
    
    with torch.no_grad():
        for batch in test_loader:
            mel, mfcc, hubert, labels, _ = batch
            mel = mel.to(trainer.device)
            mfcc = mfcc.to(trainer.device)
            hubert = hubert.to(trainer.device)
            
            mel_out = model.mel_branch(mel)
            mfcc_out = model.mfcc_branch(mfcc)
            hubert_out = model.hubert_branch(hubert)
            fused = torch.cat([mel_out, mfcc_out, hubert_out], dim=1)
            
            all_features.append(fused.cpu().numpy())
            all_labels_viz.extend(labels.numpy())
    
    if all_features:
        all_features = np.vstack(all_features)
        all_labels_viz = np.array(all_labels_viz)
        
        # 14. t-SNE visualization
        EnhancedVisualizer.plot_tsne_visualization(
            all_features,
            all_labels_viz,
            save_path=os.path.join(Config.OUTPUT_DIR, '14_tsne_visualization.png')
        )
        
        # 15. PCA visualization
        EnhancedVisualizer.plot_pca_visualization(
            all_features,
            all_labels_viz,
            save_path=os.path.join(Config.OUTPUT_DIR, '15_pca_visualization.png')
        )
    
    print(f"\n✅ All 15 visualization plots saved to {Config.OUTPUT_DIR}")
    
    # ========================================================================
    # STEP 9: SAVE RESULTS
    # ========================================================================
    print("\n[STEP 9] SAVING RESULTS")
    print("-" * 40)
    
    # Save test predictions
    test_results = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': test_preds,
        'true_emotion': [Config.EMOTION_MAP[l] for l in test_labels],
        'predicted_emotion': [Config.EMOTION_MAP[p] for p in test_preds]
    })
    
    for i in range(num_classes):
        test_results[f'prob_class_{i}'] = [p[i] for p in test_probs]
    
    test_results_path = os.path.join(Config.OUTPUT_DIR, 'test_predictions.csv')
    test_results.to_csv(test_results_path, index=False)
    print(f"✓ Test predictions saved")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(Config.OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved")
    
    # Save final model
    final_model_path = os.path.join(Config.OUTPUT_DIR, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'sample_rate': Config.SAMPLE_RATE,
            'n_mels': Config.N_MELS,
            'n_mfcc': Config.N_MFCC,
            'emotion_map': Config.EMOTION_MAP,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
        },
        'history': history,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc
    }, final_model_path)
    print(f"✓ Final model saved")
    
    # Save summary results
    total_time = time.time() - pipeline_start_time
    summary = {
        'timestamp': Config.TIMESTAMP,
        'test_accuracy': float(test_acc),
        'best_val_accuracy': float(best_val_acc),
        'total_samples': len(audio_list),
        'labeled_samples': int(np.sum(has_label)),
        'unlabeled_samples': int(len(audio_list) - np.sum(has_label)),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'batch_size': Config.BATCH_SIZE,
        'epochs': Config.EPOCHS,
        'total_training_time_seconds': total_time,
        'classification_report': report,
        'gpu_used': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    with open(Config.RESULTS_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Results summary saved")
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Total Samples: {len(audio_list)}")
    print(f"   Labeled Samples: {np.sum(has_label)}")
    print(f"   Model Parameters: {total_params:,}")
    
    print(f"\n📁 OUTPUT FILES (15 Visualization Plots):")
    print(f"   01_training_history.png - Training curves")
    print(f"   02_confusion_matrix_absolute.png - Confusion matrix (counts)")
    print(f"   03_confusion_matrix_normalized.png - Confusion matrix (normalized)")
    print(f"   04_class_performance_bar.png - Precision/Recall/F1 bar chart")
    print(f"   05_class_performance_heatmap.png - Performance metrics heatmap")
    print(f"   06_class_support.png - Class distribution")
    print(f"   07_va_scatter.png - Valence-Arousal scatter plot")
    print(f"   08_va_density.png - Valence-Arousal density map")
    print(f"   09_model_architecture.png - Model architecture summary")
    print(f"   10_prediction_distribution.png - Prediction analysis")
    print(f"   11_mfcc_sample.png - Sample MFCC heatmap")
    print(f"   12_mfcc_statistics.png - MFCC statistics")
    print(f"   13_mfcc_by_emotion.png - MFCC patterns by emotion")
    print(f"   14_tsne_visualization.png - t-SNE feature visualization")
    print(f"   15_pca_visualization.png - PCA feature visualization")
    print(f"   test_predictions.csv - Detailed predictions with probabilities")
    print(f"   training_history.csv - Numerical training metrics")
    print(f"   final_model.pth - Saved model weights")
    print(f"   results_summary.json - Complete results summary")
    print(f"   class_distribution.csv - Class distribution statistics")
    print(f"   training_log.txt - Complete console output")
    
    return model, feature_extractor, test_acc, summary


# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎵 ENHANCED MUSIC EMOTION RECOGNITION - SONG-LEVEL")
    print("Using Averaged Static Annotations + MFCC Features")
    print("="*80)
    
    try:
        print("\n🚀 Starting enhanced pipeline...")
        result = run_enhanced_pipeline()
        
        if result:
            model, feature_extractor, accuracy, summary = result
            print(f"\n✅ Enhanced pipeline completed with accuracy: {accuracy:.2f}%")
            print(f"📁 All outputs saved to: {Config.OUTPUT_DIR}")
            
            if isinstance(sys.stdout, Logger):
                sys.stdout.close()
                sys.stdout = sys.stdout.terminal
            
            print(f"\n✅ Pipeline completed successfully! Check {Config.OUTPUT_DIR} for all outputs.")
        else:
            print("\n⚠️ Pipeline completed with issues")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Pipeline interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if isinstance(sys.stdout, Logger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal