import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import math

# Note: TorchAudio operations require the torchaudio package
# Install with: pip install torchaudio

try:
    import torchaudio
    import torchaudio.transforms as T
    import torchaudio.functional as F_audio
    from torchaudio.datasets import SPEECHCOMMANDS, LIBRISPEECH
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: TorchAudio not available. Install with: pip install torchaudio")

# Audio Preprocessing Pipeline
class AudioPreprocessor:
    """Comprehensive audio preprocessing for speech tasks"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.transforms = {}
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup common audio transforms"""
        if not TORCHAUDIO_AVAILABLE:
            print("TorchAudio not available - using fallback implementations")
            return
        
        # Resampling
        self.transforms['resample_8k'] = T.Resample(
            orig_freq=self.sample_rate, 
            new_freq=8000
        )
        
        # Spectral transforms
        self.transforms['spectrogram'] = T.Spectrogram(
            n_fft=1024,
            win_length=None,
            hop_length=512,
            power=2.0
        )
        
        self.transforms['mel_spectrogram'] = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0
        )
        
        self.transforms['mfcc'] = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': 1024, 'hop_length': 512, 'n_mels': 80}
        )
        
        # Augmentations
        self.transforms['time_stretch'] = T.TimeStretch(
            hop_length=512,
            n_freq=513
        )
        
        self.transforms['pitch_shift'] = T.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=0
        )
        
        # Noise reduction
        self.transforms['spectral_centroid'] = T.SpectralCentroid(
            sample_rate=self.sample_rate
        )
        
        # Voice activity detection features
        self.transforms['zero_crossing_rate'] = lambda x: self._compute_zcr(x)
        
        print("✓ Audio transforms initialized")
    
    def load_audio(self, filepath: str, normalize: bool = True) -> Tuple[torch.Tensor, int]:
        """Load audio file"""
        if TORCHAUDIO_AVAILABLE:
            waveform, sample_rate = torchaudio.load(filepath)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = T.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize
            if normalize:
                waveform = waveform / torch.max(torch.abs(waveform))
            
            return waveform, self.sample_rate
        else:
            # Fallback: create dummy audio
            duration = 3.0  # 3 seconds
            t = torch.linspace(0, duration, int(self.sample_rate * duration))
            waveform = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
            return waveform, self.sample_rate
    
    def apply_noise_reduction(self, waveform: torch.Tensor, 
                            noise_factor: float = 0.1) -> torch.Tensor:
        """Apply simple noise reduction"""
        # Simple spectral subtraction approach
        if TORCHAUDIO_AVAILABLE:
            spec = self.transforms['spectrogram'](waveform)
            
            # Estimate noise (first 0.5 seconds)
            noise_frames = int(0.5 * self.sample_rate / 512)  # hop_length = 512
            noise_spec = torch.mean(spec[:, :, :noise_frames], dim=2, keepdim=True)
            
            # Spectral subtraction
            enhanced_spec = spec - noise_factor * noise_spec
            enhanced_spec = torch.clamp(enhanced_spec, min=0.1 * spec)
            
            # Convert back to waveform (simplified - in practice use Griffin-Lim)
            return self._spec_to_waveform(enhanced_spec)
        else:
            # Simple noise reduction fallback
            return waveform * (1 - noise_factor)
    
    def _spec_to_waveform(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram back to waveform (simplified)"""
        # In practice, use Griffin-Lim algorithm or learned inversion
        # This is a simplified version
        return torch.randn_like(spec[0, 0, :]).unsqueeze(0)
    
    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive audio features"""
        features = {}
        
        if TORCHAUDIO_AVAILABLE:
            # Spectral features
            features['mel_spectrogram'] = self.transforms['mel_spectrogram'](waveform)
            features['mfcc'] = self.transforms['mfcc'](waveform)
            features['spectrogram'] = self.transforms['spectrogram'](waveform)
            
            # Temporal features
            features['zero_crossing_rate'] = self._compute_zcr(waveform)
            features['energy'] = torch.sum(waveform ** 2, dim=-1, keepdim=True)
            
            # Spectral statistics
            features['spectral_centroid'] = self.transforms['spectral_centroid'](waveform)
            features['spectral_rolloff'] = self._compute_spectral_rolloff(waveform)
            
        else:
            # Fallback features
            features['waveform'] = waveform
            features['energy'] = torch.sum(waveform ** 2, dim=-1, keepdim=True)
        
        return features
    
    def _compute_zcr(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute zero crossing rate"""
        diff = torch.diff(torch.sign(waveform), dim=-1)
        zcr = torch.sum(torch.abs(diff), dim=-1, keepdim=True) / (2 * waveform.shape[-1])
        return zcr
    
    def _compute_spectral_rolloff(self, waveform: torch.Tensor, 
                                 rolloff_percent: float = 0.85) -> torch.Tensor:
        """Compute spectral rolloff"""
        if TORCHAUDIO_AVAILABLE:
            spec = self.transforms['spectrogram'](waveform)
            cumsum_spec = torch.cumsum(spec, dim=1)
            total_energy = cumsum_spec[:, -1:, :]
            rolloff_thresh = rolloff_percent * total_energy
            
            # Find rolloff frequency
            rolloff_idx = torch.argmax((cumsum_spec > rolloff_thresh).float(), dim=1)
            return rolloff_idx.float()
        else:
            return torch.zeros(waveform.shape[0], 1, dtype=torch.float32)
    
    def augment_audio(self, waveform: torch.Tensor, 
                     augmentation_type: str = 'time_stretch') -> torch.Tensor:
        """Apply audio augmentations"""
        if not TORCHAUDIO_AVAILABLE:
            return waveform
        
        if augmentation_type == 'time_stretch':
            # Random time stretch
            rate = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            spec = self.transforms['spectrogram'](waveform)
            stretched = self.transforms['time_stretch'](spec, rate)
            return self._spec_to_waveform(stretched)
        
        elif augmentation_type == 'pitch_shift':
            # Random pitch shift
            n_steps = torch.randint(-2, 3, (1,)).item()  # -2 to +2 semitones
            transform = T.PitchShift(self.sample_rate, n_steps)
            return transform(waveform)
        
        elif augmentation_type == 'noise':
            # Add white noise
            noise_level = torch.rand(1) * 0.1  # Up to 10% noise
            noise = torch.randn_like(waveform) * noise_level
            return waveform + noise
        
        elif augmentation_type == 'speed':
            # Change speed (affects both tempo and pitch)
            speed_factor = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            effects = [["speed", f"{speed_factor:.2f}"]]
            return F_audio.apply_effects_tensor(waveform, self.sample_rate, effects)[0]
        
        return waveform

# Speech Recognition Models
class SpeechRecognitionCNN(nn.Module):
    """CNN-based speech recognition model"""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

class SpeechRecognitionRNN(nn.Module):
    """RNN-based speech recognition model"""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int, num_classes: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Apply self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        output = self.classifier(pooled)
        return output

class SpeechTransformer(nn.Module):
    """Transformer-based speech recognition model"""
    
    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, num_classes: int):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        seq_len = x.size(1)
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Speaker Recognition
class SpeakerEmbeddingModel(nn.Module):
    """Model for speaker embedding extraction"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, embedding_dim)
        )
        
        self.classifier = nn.Linear(embedding_dim, 100)  # 100 speakers
    
    def forward(self, x, return_embedding=False):
        # x shape: (batch, features)
        embedding = self.feature_extractor(x)
        
        if return_embedding:
            return embedding
        
        classification = self.classifier(embedding)
        return classification, embedding

# Voice Activity Detection
class VADModel(nn.Module):
    """Voice Activity Detection model"""
    
    def __init__(self, input_size: int):
        super().__init__()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.feature_layers(x)

# Audio Data Pipeline
class AudioDataset(torch.utils.data.Dataset):
    """Custom audio dataset"""
    
    def __init__(self, audio_paths: List[str], labels: List[int],
                 preprocessor: AudioPreprocessor, 
                 transform_type: str = 'mel_spectrogram'):
        self.audio_paths = audio_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.transform_type = transform_type
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio
        waveform, _ = self.preprocessor.load_audio(self.audio_paths[idx])
        
        # Extract features
        features = self.preprocessor.extract_features(waveform)
        
        # Get specified feature type
        if self.transform_type in features:
            feature = features[self.transform_type]
        else:
            feature = waveform
        
        return feature, self.labels[idx]

# Speech Processing Pipeline
class SpeechProcessor:
    """Complete speech processing pipeline"""
    
    def __init__(self, task: str = 'classification', model_type: str = 'cnn'):
        self.task = task
        self.model_type = model_type
        self.preprocessor = AudioPreprocessor()
        self.model = None
        self.trained = False
    
    def build_model(self, num_classes: int, input_shape: Tuple[int, ...]):
        """Build model based on task and type"""
        
        if self.task == 'classification':
            if self.model_type == 'cnn':
                self.model = SpeechRecognitionCNN(num_classes)
            elif self.model_type == 'rnn':
                input_size = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
                self.model = SpeechRecognitionRNN(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=2,
                    num_classes=num_classes
                )
            elif self.model_type == 'transformer':
                input_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
                self.model = SpeechTransformer(
                    input_dim=input_dim,
                    d_model=256,
                    nhead=8,
                    num_layers=4,
                    num_classes=num_classes
                )
        
        elif self.task == 'speaker_recognition':
            input_dim = np.prod(input_shape)
            self.model = SpeakerEmbeddingModel(input_dim)
        
        elif self.task == 'vad':
            input_size = np.prod(input_shape)
            self.model = VADModel(input_size)
        
        print(f"✓ {self.model_type.upper()} model built for {self.task}")
    
    def train(self, train_loader, val_loader, epochs: int = 10):
        """Train the speech model"""
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss() if self.task != 'vad' else nn.BCELoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Reshape data for different model types
                if self.model_type == 'cnn' and len(data.shape) == 3:
                    data = data.unsqueeze(1)  # Add channel dimension
                elif self.model_type in ['rnn', 'transformer'] and len(data.shape) == 4:
                    # Flatten spatial dimensions for sequence models
                    batch_size, channels, height, width = data.shape
                    data = data.view(batch_size, height, width * channels)
                
                outputs = self.model(data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # For models that return multiple outputs
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx >= 5:  # Limit for demo
                    break
            
            # Validation
            val_accuracy = self._evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss = {epoch_loss/num_batches:.4f}, "
                  f"Val Acc = {val_accuracy:.4f}")
        
        self.trained = True
        print("✓ Training completed")
    
    def _evaluate(self, data_loader):
        """Evaluate model accuracy"""
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                # Reshape data
                if self.model_type == 'cnn' and len(data.shape) == 3:
                    data = data.unsqueeze(1)
                elif self.model_type in ['rnn', 'transformer'] and len(data.shape) == 4:
                    batch_size, channels, height, width = data.shape
                    data = data.view(batch_size, height, width * channels)
                
                outputs = self.model(data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                if batch_idx >= 3:  # Limit for demo
                    break
        
        self.model.train()
        return correct / total if total > 0 else 0.0

# Audio Visualization
class AudioVisualizer:
    """Visualization utilities for audio data"""
    
    @staticmethod
    def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform"):
        """Plot audio waveform"""
        time_axis = torch.linspace(0, waveform.shape[-1] / sample_rate, waveform.shape[-1])
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, waveform[0].cpu())
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_spectrogram(spectrogram: torch.Tensor, sample_rate: int, 
                        title: str = "Spectrogram", log_scale: bool = True):
        """Plot spectrogram"""
        if log_scale:
            spectrogram = torch.log(spectrogram + 1e-7)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(spectrogram[0].cpu(), aspect='auto', origin='lower')
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        plt.colorbar()
        plt.show()
    
    @staticmethod
    def plot_mel_spectrogram(mel_spec: torch.Tensor, title: str = "Mel Spectrogram"):
        """Plot mel spectrogram"""
        mel_spec_db = torch.log(mel_spec + 1e-7)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spec_db[0].cpu(), aspect='auto', origin='lower')
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.colorbar()
        plt.show()
    
    @staticmethod
    def plot_mfcc(mfcc: torch.Tensor, title: str = "MFCC"):
        """Plot MFCC features"""
        plt.figure(figsize=(12, 6))
        plt.imshow(mfcc[0].cpu(), aspect='auto', origin='lower')
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("MFCC Coefficients")
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    print("Comprehensive TorchAudio Speech Processing")
    print("=" * 45)
    
    if not TORCHAUDIO_AVAILABLE:
        print("TorchAudio not available. Demonstrating with fallback implementations.")
    
    print("\n1. Audio Preprocessing")
    print("-" * 25)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sample_rate=16000)
    
    # Create sample audio (sine wave)
    duration = 2.0
    sample_rate = 16000
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    waveform = torch.sin(2 * math.pi * frequency * t).unsqueeze(0)
    
    print(f"Sample audio shape: {waveform.shape}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Extract features
    features = preprocessor.extract_features(waveform)
    
    print("\nExtracted features:")
    for feature_name, feature_tensor in features.items():
        print(f"  {feature_name}: {feature_tensor.shape}")
    
    print("\n2. Audio Transformations")
    print("-" * 27)
    
    if TORCHAUDIO_AVAILABLE:
        # Apply various transformations
        transformations = ['noise', 'time_stretch', 'pitch_shift']
        
        for transform_type in transformations:
            try:
                augmented = preprocessor.augment_audio(waveform, transform_type)
                print(f"✓ {transform_type}: {augmented.shape}")
            except Exception as e:
                print(f"✗ {transform_type}: {e}")
    
    print("\n3. Speech Recognition Models")
    print("-" * 32)
    
    num_classes = 10  # 10 speech commands
    
    # Create different model architectures
    models = {}
    
    # CNN model
    models['CNN'] = SpeechRecognitionCNN(num_classes=num_classes)
    
    # RNN model
    models['RNN'] = SpeechRecognitionRNN(
        input_size=80,  # MFCC features
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    )
    
    # Transformer model
    models['Transformer'] = SpeechTransformer(
        input_dim=80,
        d_model=256,
        nhead=8,
        num_layers=4,
        num_classes=num_classes
    )
    
    # Test forward pass with different input shapes
    test_inputs = {
        'CNN': torch.randn(2, 1, 80, 100),  # (batch, channels, mel_bins, time)
        'RNN': torch.randn(2, 100, 80),     # (batch, time, features)
        'Transformer': torch.randn(2, 100, 80)  # (batch, time, features)
    }
    
    for name, model in models.items():
        try:
            output = model(test_inputs[name])
            if isinstance(output, tuple):
                output = output[0]
            print(f"{name} output shape: {output.shape}")
        except Exception as e:
            print(f"{name} error: {e}")
    
    print("\n4. Speaker Recognition")
    print("-" * 23)
    
    # Speaker embedding model
    speaker_model = SpeakerEmbeddingModel(input_dim=1024, embedding_dim=256)
    
    # Test speaker recognition
    test_speaker_input = torch.randn(3, 1024)  # 3 speakers
    classification_output, embeddings = speaker_model(test_speaker_input)
    
    print(f"Speaker classification output: {classification_output.shape}")
    print(f"Speaker embeddings: {embeddings.shape}")
    
    # Compute speaker similarity
    speaker_similarity = torch.cosine_similarity(
        embeddings[0:1], embeddings[1:2], dim=1
    )
    print(f"Speaker similarity (0 vs 1): {speaker_similarity.item():.3f}")
    
    print("\n5. Voice Activity Detection")
    print("-" * 30)
    
    # VAD model
    vad_model = VADModel(input_size=13)  # MFCC features
    
    # Test VAD
    test_vad_input = torch.randn(5, 13)  # 5 frames
    vad_output = vad_model(test_vad_input)
    
    print(f"VAD output shape: {vad_output.shape}")
    print(f"VAD predictions: {vad_output.squeeze().tolist()}")
    
    print("\n6. Complete Speech Processing Pipeline")
    print("-" * 43)
    
    # Create speech processor
    processor = SpeechProcessor(task='classification', model_type='cnn')
    
    # Build model
    processor.build_model(num_classes=10, input_shape=(1, 80, 100))
    
    # Create dummy dataset
    dummy_data = [torch.randn(80, 100) for _ in range(20)]
    dummy_labels = [i % 10 for i in range(20)]
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = DummyDataset(dummy_data, dummy_labels)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Train model
    processor.train(train_loader, val_loader, epochs=3)
    
    print("\n7. Audio Visualization")
    print("-" * 23)
    
    visualizer = AudioVisualizer()
    
    print("Audio visualization methods available:")
    print("  - plot_waveform(): Display time-domain waveform")
    print("  - plot_spectrogram(): Display frequency-time representation")
    print("  - plot_mel_spectrogram(): Display mel-scale spectrogram")
    print("  - plot_mfcc(): Display MFCC coefficients")
    
    # Note: Actual plotting commented out to avoid display issues
    # visualizer.plot_waveform(waveform, sample_rate)
    
    print("\n8. TorchAudio Best Practices")
    print("-" * 33)
    
    best_practices = [
        "Use appropriate sample rates for your application (16kHz for speech)",
        "Apply proper normalization to audio signals",
        "Use mel spectrograms for most speech recognition tasks",
        "Apply data augmentation to improve model robustness",
        "Use MFCC features for traditional speech processing",
        "Implement proper VAD for real-world applications",
        "Use attention mechanisms for variable-length sequences",
        "Apply noise reduction preprocessing when needed",
        "Monitor for data leakage in speaker recognition tasks",
        "Use appropriate evaluation metrics (WER for ASR)",
        "Implement proper padding for batch processing",
        "Consider computational efficiency for real-time applications"
    ]
    
    print("TorchAudio Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n9. Common Speech Tasks")
    print("-" * 24)
    
    speech_tasks = {
        "Speech Recognition": "Convert spoken words to text (ASR)",
        "Speaker Recognition": "Identify who is speaking",
        "Speech Synthesis": "Generate speech from text (TTS)",
        "Voice Activity Detection": "Detect presence of speech",
        "Emotion Recognition": "Classify emotional state from speech",
        "Language Identification": "Identify spoken language",
        "Keyword Spotting": "Detect specific wake words",
        "Speech Enhancement": "Remove noise from speech signals",
        "Diarization": "Segment audio by speaker",
        "Pronunciation Assessment": "Evaluate pronunciation quality"
    }
    
    print("Common Speech Processing Tasks:")
    for task, description in speech_tasks.items():
        print(f"  {task}: {description}")
    
    print("\n10. Integration and Deployment")
    print("-" * 33)
    
    integration_tips = [
        "Real-time processing: Use streaming inference with buffering",
        "Mobile deployment: Quantize models and use efficient architectures",
        "Cloud deployment: Implement proper scaling and load balancing",
        "Edge deployment: Optimize for low power consumption",
        "Multi-language: Train language-specific models or use multilingual approaches",
        "Noise robustness: Train with diverse acoustic conditions",
        "Privacy: Implement on-device processing when possible",
        "Latency optimization: Use smaller models and efficient inference"
    ]
    
    print("Integration and Deployment Tips:")
    for i, tip in enumerate(integration_tips, 1):
        print(f"{i}. {tip}")
    
    print("\nTorchAudio speech processing demonstration completed!")
    print("Key components covered:")
    print("  - Audio preprocessing and feature extraction")
    print("  - Multiple model architectures (CNN, RNN, Transformer)")
    print("  - Speaker recognition and voice activity detection")
    print("  - Complete training pipeline")
    print("  - Audio augmentation techniques")
    print("  - Visualization utilities")
    
    print("\nTorchAudio enables:")
    print("  - Efficient audio data loading and preprocessing")
    print("  - Rich set of audio transformations")
    print("  - Integration with PyTorch models")
    print("  - End-to-end speech processing pipelines")
    print("  - Support for various audio formats and codecs")