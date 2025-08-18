# Movement-Based Features for Interaction Design: Computational Prototype

This repository contains the computational prototype developed for the research paper **"Exploring a user-centered approach for movement-based features in interaction design"** submitted to the International Journal of Human-Computer Studies.

## Overview

This computational prototype implements a theoretical framework that bridges the gap between technical movement analysis capabilities and interaction design practice. The system operates in an **offline setting**, processing previously captured multi-view scenes to render motion-based features for analysis and visualization. Rather than presenting raw motion data, the system provides designer-interpretable features that support both technical reliability and creative exploration in movement-based interaction design.

## Research Context

The prototype was developed to address the accessibility gap in movement-based feature extraction tools for interaction designers. Many existing systems focus on technical capabilities without considering how to translate movement data into design opportunities. This work prioritizes designer interpretability and provides tools that accommodate different levels of technical expertise within interaction design practice.

## System Architecture

The prototype consists of three primary components:

### 1. 3D Pose Estimation and Tracking
- **Multi-camera approach** for large volume spaces and occlusion resolution
- **Camera calibration** using ChArUco board pattern with synchronized capture
- **AlphaPose body estimator** with YOLOV3 detector for human detection
- **Multi-view correspondence** and 3D pose reconstruction
- **Tracking-by-detection** with temporal filtering for smooth pose trajectories
- Detects **15 body landmark locations** for each person in 3D space

### 2. Feature Extraction System
The prototype implements **seven movement-based features** that progress from individual spatial analysis to group behavior detection:

#### Individual Movement Features
- **Bounding Box**: Real-time rectangular parallelepiped enclosing each person's 3D pose
- **Trajectory**: Short-time movement patterns using circular buffer of 3D points
- **Heading**: Directional intent analysis using velocity vector projection on ground plane

#### Group Behavior Features
- **Instantaneous Clustering**: Real-time group formation detection using mean shift algorithm
- **Hotspots**: Spatial usage pattern analysis using DBSCAN clustering over time windows
- **Trajectory Similarity**: Coordinated movement detection using Procrustes analysis
- **Correlations Across Movement Patterns**: Granular synchronized movement analysis between specific users

### 3. Visualization Engine
- **Multi-view video playback** with synchronized camera feeds
- **3D scene visualization** with skeletal representations
- **Feature-specific visualizations**:
  - Geometric representations for spatial analysis
  - Pattern-based visualizations for qualitative movement interpretation
  - Heatmaps for hotspot analysis
  - Event-based indicators for trajectory similarity

## Key Features

- **Designer-Interpretable Features**: Higher-level concepts that align with designers' mental models rather than raw technical data
- **Progressive Complexity**: Features build from basic spatial analysis to advanced group dynamics
- **Real-time Processing**: Live analysis and visualization of movement patterns
- **Multi-user Support**: Handles multiple people simultaneously in room-scale spaces
- **Scalable Deployment**: Camera-based approach suitable for museums, exhibitions, and public spaces

## Technical Specifications

### Data Processing Pipeline
The prototype processes pre-captured multi-view scenes that have been through:
- Multi-camera calibration using ChArUco board patterns
- AlphaPose body pose estimation with YOLOV3 detection
- 2D body pose estimation on each view

### Software Dependencies
- **Python 3.8** - Core programming language
- **PyTorch 1.7.1+cu101** - Deep learning framework with CUDA 10.1 and cuDNN 7.0 support
- **PyQt5 5.9.2** - GUI framework for the user interface
- **PyQtGraph 0.11.0** - Real-time plotting and 3D visualization library
- **PyOpenGL 3.1.5** - 3D graphics rendering engine
- **OpenCV-Python 4.5.4.58** - Computer vision operations and image processing
- **pymvg 2.0.0** - Multi-view geometry calculations
- **dvg-ringbuffer 1.0.3** - Circular buffer operations for real-time data handling

### Installation

#### Prerequisites
Ensure you have Python 3.8 installed and CUDA 10.1 with cuDNN 7.0 available on your system.

#### Install Dependencies
```bash
# Clone the repository
git clone [repository-url]
cd Exploring_movement_features_for_interaction_design

# Install PyTorch with CUDA 10.1 support
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining dependencies
pip install -r requirements.txt
```

**Note**: The PyTorch installation requires CUDA 10.1 and cuDNN 7.0 to be properly installed on your system. If you don't have CUDA support or prefer CPU-only execution, you can install the CPU version instead:
```bash
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

### Running the Prototype
```bash
python main.py
```

### Sample Data Structure

The repository includes a complete sample dataset in the `data/simple_scene` folder that demonstrates all the prototype's capabilities. This pre-processed scene contains all necessary files to run the movement analysis without requiring additional data capture or processing.

#### Data Folder Organization

```
data/
└── simple_scene/
    ├── detections/
    │   ├── 01_upb_bbox_tracks_by_frame_225_1950.pkl
    │   └── pred_multi_centertrack_coco.pkl
    ├── videos/
    │   └── [multi-view video recordings]
    └── calibration.json
```

#### File Descriptions

**Root Scene Folder** (`data/simple_scene/`):
- Contains all data for a single multi-view capture session
- Represents a room-scale environment with multiple people interacting
- Pre-processed and ready for immediate analysis

**Detection Data** (`detections/`):
- `01_upb_bbox_tracks_by_frame_225_1950.pkl` - Frame-by-frame tracking data with temporal associations
- `pred_multi_centertrack_coco.pkl` - Multi-view 2D pose predictions in COCO format from AlphaPose body estimator

**Video Data** (`videos/`):
- Multi-view synchronized camera recordings
- Each video corresponds to a different camera viewpoint
- Used for visual validation and overlay of pose detection results

**Calibration Data** (`calibration.json`):
- Camera intrinsic and extrinsic parameters
- Multi-view geometry relationships
- Essential for 3D pose reconstruction from 2D detections

#### Data Processing Pipeline

This sample data has been processed through the complete pipeline:

1. **Multi-camera capture** - Synchronized recording from multiple viewpoints
2. **Camera calibration** - Using ChArUco board patterns for precise geometry
3. **2D pose detection** - AlphaPose with YOLOV3 applied to each camera view
4. **Multi-view correspondence** - Matching 2D detections across camera views
5. **3D reconstruction** - Converting 2D poses to 3D coordinates
6. **Temporal tracking** - Associating poses across frames for trajectory analysis

The included data allows you to explore all seven movement-based features immediately without requiring your own motion capture setup or preprocessing pipeline.

### Using the Interface

Once the application launches, you need to select the `data/simple_scene` folder to load the data and enable the feature extraction capabilities. The graphical user interface consists of three main sections:

**Selection Section**: Provides controls for opening scenes and choosing specific movement features. Use the feature dropdown to isolate and compare individual movement characteristics, preventing cognitive overload and enabling focused exploration.

**Multi-view Video Playback Section**: Displays synchronized multi-view camera feeds with 2D pose detection overlays. This allows you to validate system behavior by directly comparing detected poses with actual movement, building trust in the detection accuracy.

**Three-dimensional Representation Section**: The core visualization area that renders feature visualizations within a spatial context. Navigate freely through the 3D space using:
- **Keyboard and mouse input** for six degrees of freedom camera movement
- **Spatial navigation** to explore relationships between features and space from various viewpoints
- **Perspective shifts** by zooming in to examine interactions or panning out to observe overall patterns
- **Feature comparison** through the selection interface

The interface enables experiential exploration through spatial engagement rather than numerical parameter adjustments, making complex movement data accessible through familiar spatial and visual thinking.

## Research Validation

The prototype was evaluated through qualitative studies with interaction designers, demonstrating:
- Enhanced ability to use human motion in design practice
- Improved understanding of group dynamics and social patterns
- Better integration of movement analysis into creative workflows
- Reduced barriers for non-technical designers working with movement data

## Authors

- **Antonio Escamilla** - Universidad Pontificia Bolivariana, Medellín, Colombia & Universitat Oberta de Catalunya, Barcelona, Spain
- **Javier Melenchón** - Universitat Oberta de Catalunya, Barcelona, Spain  
- **Carlos Monzo** - Universitat Oberta de Catalunya, Barcelona, Spain
- **Jose Antonio Morán** - Universitat Oberta de Catalunya, Barcelona, Spain
- **Juan Pablo Carrascal** - Microsoft Corporation, Barcelona, Spain

## Citation

If you use this prototype in your research, please cite:

```bibtex
@article{escamilla2025exploring,
  title={Exploring a user-centered approach for movement-based features in interaction design},
  author={Escamilla, Antonio and Melench{\'o}n, Javier and Monzo, Carlos and Mor{\'a}n, Jose Antonio and Carrascal, Juan Pablo},
  year={2025}
}
```

## License

This project is released under the MIT Open Source License.

## Contributing

We welcome contributions to improve the prototype. Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

This research was conducted as part of ongoing work in movement-based interaction design, with evaluation studies conducted in controlled room-scale environments. Special thanks to the interaction designers who participated in the evaluation studies and provided valuable feedback on the prototype's capabilities and usability.

---

**Note**: Complete hardware and software specifications for system replication, along with supplementary materials including analysis documentation and theme development code, are available in the research paper and supplementary materials.
 
