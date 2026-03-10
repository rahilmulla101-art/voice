print("Checking ML libraries...\n")

try:
    import numpy
    print("✓ numpy OK")
except:
    print("✗ numpy NOT INSTALLED")

try:
    import librosa
    print("✓ librosa OK")
except:
    print("✗ librosa NOT INSTALLED")

try:
    import tensorflow as tf
    print("✓ tensorflow OK")
except:
    print("✗ tensorflow NOT INSTALLED")

try:
    import sklearn
    print("✓ scikit-learn OK")
except:
    print("✗ scikit-learn NOT INSTALLED")

try:
    import matplotlib
    print("✓ matplotlib OK")
except:
    print("✗ matplotlib NOT INSTALLED")

print("\nCheck complete!")
