"""
Demo script: Test RT60 extraction with sample audio
"""
import numpy as np
from scipy.io import wavfile
import tempfile
from pathlib import Path
from rt60_extractor import RT60Extractor
import matplotlib.pyplot as plt

def create_test_audio(duration=3, sr=16000):
    """
    Create a synthetic reverberant audio signal for testing.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        
    Returns:
        Audio signal, sample rate
    """
    # Generate impulse response with exponential decay (simulating RT60)
    t = np.linspace(0, duration, int(duration * sr))
    
    # Decaying sinusoid (simulating reverberant speech)
    freq = 500  # Hz
    decay_rate = 3.0  # Higher = faster decay = lower RT60
    
    audio = np.sin(2 * np.pi * freq * t) * np.exp(-decay_rate * t)
    
    # Add noise
    audio += 0.05 * np.random.randn(len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio, sr


def test_rt60_extraction():
    """Test RT60 extraction on synthetic audio."""
    print("="*60)
    print("RT60 Extraction Test")
    print("="*60)
    
    # Create test audio
    print("\n1. Generating synthetic test audio...")
    audio, sr = create_test_audio(duration=3, sr=16000)
    print(f"   - Duration: {len(audio)/sr:.1f}s")
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Amplitude range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Save test audio
    temp_dir = tempfile.gettempdir()
    test_audio_path = Path(temp_dir) / "test_audio.wav"
    wavfile.write(test_audio_path, sr, (audio * 32767).astype(np.int16))
    print(f"   - Saved to: {test_audio_path}")
    
    # Extract RT60
    print("\n2. Running RT60 estimation...")
    extractor = RT60Extractor()
    rt60 = extractor.estimate_rt60(audio, sr)
    
    if rt60:
        print(f"   ✓ Estimated RT60: {rt60:.3f} seconds ({rt60*1000:.1f} ms)")
    else:
        print("   ✗ RT60 estimation failed")
        return
    
    # Test batch processing
    print("\n3. Testing batch processing...")
    video_paths = [str(test_audio_path)]  # Librosa can read WAV
    output_csv = Path(temp_dir) / "rt60_test_results.csv"
    
    extractor.process_batch(video_paths, str(output_csv))
    
    if output_csv.exists():
        print(f"   ✓ Results saved to {output_csv}")
        with open(output_csv, 'r') as f:
            print("\n" + "="*60)
            print("Output CSV Preview:")
            print("="*60)
            for i, line in enumerate(f):
                print(line.rstrip())
                if i > 5:  # Show first few lines
                    print("   ...")
                    break
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_rt60_extraction()
