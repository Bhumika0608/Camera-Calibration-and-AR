# Camera-Calibration-and-AR

This project explores camera calibration, pose estimation, and augmented reality (AR) using both checkerboard markers and arbitrary textured surfaces. Used OpenCV for feature detection, intrinsic parameter estimation, real-time 3D projection, and robust feature matching. The goal was to align virtual 3D objects with real scenes accurately and stably.

---

## Key Features

| Feature | Description |
|--------|-------------|
| **Camera Calibration** | Based on checkerboard corner detection using `findChessboardCorners()` |
| **Pose Estimation** | Real-time pose tracking via `solvePnP()` |
| **AR Projection** | Virtual 3D axes + custom house model projected on checkerboard |
| **Marker-less AR** | ORB feature detection + RANSAC homography for projection on textured surfaces |
| **Multi-Camera Support** | Calibration comparison & best-camera selection |
| **Static Image Support** | AR works on pre-recorded sequences and images |

---

## Techniques & Algorithms
- Checkerboard Corner Detection  
- Camera Calibration (`calibrateCamera`)
- Distortion Estimation
- Pose Estimation (`solvePnP`)
- 3D Projection (`projectPoints`)
- ORB Features + Lowe’s Ratio Test
- Homography Estimation with RANSAC

---

## Results Summary

### 📌 Camera Calibration
- **Frames Used**: 22
- **Corner Points per Frame**: 54
- **Mean Reprojection Error**: **1.93 px** → Good calibration accuracy

### 📌 Pose Tracking
- Smooth and realistic changes in pitch, yaw, roll  
- Accurate tracking of sideways + forward motion

### 📌 AR Object Projection
- Custom 3D **house model** (walls, asymmetric roof, chimney, door)
- Remained stable and properly aligned during camera movement

### 📌 Marker-less AR
- ~500 ORB features detected reliably
- Robust homography-based projection even under rotation

✔  Requirements
Build Tools

    macOS Monterey / Ventura / Sonoma

    CMake ≥ 3.10

    Clang / Apple LLVM Toolchain (comes with macOS)

    C++17 compiler support

Xcode Command Line Tools

    xcode-select --install

Required Dependencies
1️⃣ OpenCV 4.x

Install via Homebrew:

    brew install opencv


This automatically places OpenCV in:

    /opt/homebrew/opt/opencv/


▶️ Quick Start
Automated Build (Recommended)

From the project root:

    ./build_project.sh


This will:

✔ Configure with CMake
✔ Build the project
✔ Output executable to bin/ directory

Manual Build Using CMake

    mkdir build

    cd build

    cmake ..

    make -j4

    ./Camera_Calibration   # Run the application

🎥 Camera Permissions (Important)

macOS may block camera access by default.

Enable:
System Settings → Privacy & Security → Camera → Allow for Terminal / IDE
