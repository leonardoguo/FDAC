"""
FDAC-NET User Manual
Improved object detection and instance segmentation model based on Mask R-CNN
"""

# ==================== ENVIRONMENT CONFIGURATION ====================
"""
Required packages:
Package                 Version
----------------------- ---------
- tensorflow==2.4.0          # Deep learning framework
- keras==2.7.0              # High-level neural network API
- opencv-python==4.5.4.58   # Image processing
- Pillow==9.0.1             # Image handling
- numpy==1.19.5             # Numerical computing
- scipy==1.7.3              # Scientific computing
- matplotlib==3.5.1         # Plotting and visualization
- pycocotools-windows==2.0  # COCO dataset evaluation
- imgaug==0.2.6             # Image augmentation
- Cython==0.29.25           # Python C extensions
- h5py==2.10.0              # HDF5 file handling
- scikit-image==0.19.1      # Image processing algorithms
- Shapely==1.7.1            # Geometric operations
"""

# ==================== DATASET STRUCTURE ====================
"""
Download MiniCOCO dataset:
Link: https://pan.baidu.com/s/1SyYSGVuHoLfYFJz3n6Bb2A
Code: z1fj
minicoco/
├── train2017/           # Training set images
├── val2017/             # Validation set images
└── Annotations/         # Annotation files
"""

# ==================== MODEL TRAINING ====================
"""
# Basic training:
python coco.py train --dataset=minicoco --model=logs/FDAC-NET_coco.h5
"""

# ==================== MODEL EVALUATION ====================
"""
Download pre-trained weights (285 MB):FDAC-NET_coco.h5
Link: https://pan.baidu.com/s/1BLEP8O_liR-G3q53gU5rHA
Code: t899
# Evaluate model:
python coco.py evaluate --dataset=minicoco --model=logs/FDAC-NET_coco.h5

# Expected output format:
# Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.465
# Per category AP: [ 0.632  0.349  0.414]
"""

# ==================== TESTING & INFERENCE ====================
"""
# Run inference:
python test.py
"""

# ==================== QUICK START ====================
"""
# 1. Prepare dataset (minicoco)
# 2. Train model:
python coco.py train --dataset=minicoco --model=logs/FDAC-NET_coco.h5
# 3. Evaluate model:
python coco.py evaluate --dataset=minicoco --model=logs/FDAC-NET_coco.h5
# 4. Run inference:
python test.py


















