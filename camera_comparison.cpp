/*
 * Ishan Chaudhary, Bhumika Yadav
 * Fall 2025
 * CS 5330 Computer Vision
 Camera Comparison Tool (Extension)
---------------------------------------------------------

 * Calibrates multiple cameras and generates detailed comparison report of
 * intrinsic parameters, distortion coefficients, and reprojection errors.
 * 
 * Controls: SPACE=Capture, N=Next camera, R=Reset, ESC=Finish
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

using namespace cv;
using namespace std;

const Size CHECKERBOARD_SIZE(9, 6);
const float SQUARE_SIZE = 25.0f;
const int MIN_IMAGES = 10;
const int TARGET_IMAGES = 15;

struct CameraCalibration {
    int cameraIndex;
    Mat cameraMatrix;
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    double reprojectionError;
    int numImages;
    Size imageSize;
    vector<vector<Point2f>> allImagePoints;
    vector<vector<Point3f>> allObjectPoints;
};

// Generate 3D object points for checkerboard
vector<Point3f> generateObjectPoints() {
    vector<Point3f> objectPoints;
    for (int i = 0; i < CHECKERBOARD_SIZE.height; i++) {
        for (int j = 0; j < CHECKERBOARD_SIZE.width; j++) {
            objectPoints.push_back(Point3f(j * SQUARE_SIZE, i * SQUARE_SIZE, 0));
        }
    }
    return objectPoints;
}

// Detect available cameras
vector<int> detectCameras() {
    vector<int> availableCameras;
    
    cout << "Detecting cameras..." << endl;
    for (int i = 0; i < 5; i++) {
        VideoCapture cap(i);
        if (cap.isOpened()) {
            // Try to read a frame to verify camera works
            Mat frame;
            cap >> frame;
            if (!frame.empty()) {
                availableCameras.push_back(i);
                cout << "  Camera " << i << " detected" << endl;
            }
            cap.release();
        }
    }
    
    return availableCameras;
}

// Capture calibration images for a camera
bool captureCalibrationImages(int cameraIndex, CameraCalibration& calib) {
    VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        cerr << "ERROR: Could not open camera " << cameraIndex << endl;
        return false;
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    cout << "\n=== Calibrating Camera " << cameraIndex << " ===" << endl;
    cout << "Target: " << TARGET_IMAGES << " images (minimum: " << MIN_IMAGES << ")" << endl;
    cout << "\nControls:" << endl;
    cout << "  SPACE: Capture image" << endl;
    cout << "  R: Reset and start over" << endl;
    cout << "  N: Finish this camera (if minimum reached)" << endl;
    cout << "  ESC: Skip this camera\n" << endl;
    
    vector<Point3f> objectPoints = generateObjectPoints();
    calib.cameraIndex = cameraIndex;
    calib.allImagePoints.clear();
    calib.allObjectPoints.clear();
    
    Mat frame;
    int capturedCount = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Failed to capture frame!" << endl;
            break;
        }
        
        calib.imageSize = frame.size();
        Mat display = frame.clone();
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Detect checkerboard
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, CHECKERBOARD_SIZE, corners,
                                          CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        
        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(display, CHECKERBOARD_SIZE, corners, found);
            
            putText(display, "Checkerboard detected - Press SPACE to capture",
                   Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        } else {
            putText(display, "Checkerboard not detected - Adjust position",
                   Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }
        
        // Show progress
        string progress = "Captured: " + to_string(capturedCount) + "/" + to_string(TARGET_IMAGES);
        putText(display, progress, Point(10, display.rows - 40),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
        
        if (capturedCount >= MIN_IMAGES) {
            putText(display, "Press N to finish (minimum reached)",
                   Point(10, display.rows - 10), FONT_HERSHEY_SIMPLEX, 0.5,
                   Scalar(0, 255, 0), 1);
        }
        
        imshow("Camera " + to_string(cameraIndex) + " Calibration", display);
        
        int key = waitKey(1);
        if (key == 27) { // ESC
            cout << "Skipping camera " << cameraIndex << endl;
            cap.release();
            destroyAllWindows();
            return false;
        } else if (key == 'n' || key == 'N') {
            if (capturedCount >= MIN_IMAGES) {
                break;
            } else {
                cout << "Need at least " << MIN_IMAGES << " images!" << endl;
            }
        } else if (key == 'r' || key == 'R') {
            cout << "Resetting calibration..." << endl;
            calib.allImagePoints.clear();
            calib.allObjectPoints.clear();
            capturedCount = 0;
        } else if (key == ' ' && found) {
            calib.allImagePoints.push_back(corners);
            calib.allObjectPoints.push_back(objectPoints);
            capturedCount++;
            cout << "Image " << capturedCount << " captured" << endl;
            
            // Visual feedback
            Mat flash = Mat::ones(frame.size(), frame.type()) * 255;
            imshow("Camera " + to_string(cameraIndex) + " Calibration", flash);
            waitKey(100);
            
            if (capturedCount >= TARGET_IMAGES) {
                cout << "Target reached! Press N to finish or continue capturing..." << endl;
            }
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    if (capturedCount < MIN_IMAGES) {
        cout << "Not enough images captured for camera " << cameraIndex << endl;
        return false;
    }
    
    calib.numImages = capturedCount;
    return true;
}

// Perform calibration
bool performCalibration(CameraCalibration& calib) {
    cout << "\nPerforming calibration for camera " << calib.cameraIndex << "..." << endl;
    
    calib.cameraMatrix = Mat::eye(3, 3, CV_64F);
    calib.distCoeffs = Mat::zeros(8, 1, CV_64F);
    
    try {
        calib.reprojectionError = calibrateCamera(
            calib.allObjectPoints,
            calib.allImagePoints,
            calib.imageSize,
            calib.cameraMatrix,
            calib.distCoeffs,
            calib.rvecs,
            calib.tvecs,
            CALIB_FIX_K4 | CALIB_FIX_K5
        );
        
        cout << "✓ Calibration successful!" << endl;
        cout << "  Reprojection error: " << calib.reprojectionError << " pixels" << endl;
        
        return true;
    } catch (const Exception& e) {
        cerr << "ERROR: Calibration failed: " << e.what() << endl;
        return false;
    }
}

// Save calibration to file
void saveCalibration(const CameraCalibration& calib) {
    string filename = "camera_" + to_string(calib.cameraIndex) + "_intrinsics.yml";
    FileStorage fs(filename, FileStorage::WRITE);
    
    fs << "camera_index" << calib.cameraIndex;
    fs << "calibration_date" << "November 2025";
    fs << "image_width" << calib.imageSize.width;
    fs << "image_height" << calib.imageSize.height;
    fs << "num_images" << calib.numImages;
    fs << "camera_matrix" << calib.cameraMatrix;
    fs << "distortion_coefficients" << calib.distCoeffs;
    fs << "reprojection_error" << calib.reprojectionError;
    
    fs.release();
    
    cout << "✓ Calibration saved to: " << filename << endl;
}

// Generate comparison report
void generateComparisonReport(const vector<CameraCalibration>& calibrations) {
    if (calibrations.empty()) {
        cout << "\nNo calibrations to compare!" << endl;
        return;
    }
    
    cout << "\n" << string(80, '=') << endl;
    cout << "CAMERA CALIBRATION COMPARISON REPORT" << endl;
    cout << string(80, '=') << endl;
    
    // Create comparison table
    cout << "\n1. BASIC INFORMATION" << endl;
    cout << string(80, '-') << endl;
    cout << left << setw(10) << "Camera"
         << setw(15) << "Resolution"
         << setw(12) << "Images"
         << setw(20) << "Reproj. Error (px)" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& calib : calibrations) {
        cout << left << setw(10) << calib.cameraIndex
             << setw(15) << (to_string(calib.imageSize.width) + "x" + to_string(calib.imageSize.height))
             << setw(12) << calib.numImages
             << setw(20) << fixed << setprecision(4) << calib.reprojectionError << endl;
    }
    
    // Focal lengths
    cout << "\n2. FOCAL LENGTHS" << endl;
    cout << string(80, '-') << endl;
    cout << left << setw(10) << "Camera"
         << setw(15) << "fx (pixels)"
         << setw(15) << "fy (pixels)"
         << setw(15) << "Aspect Ratio" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& calib : calibrations) {
        double fx = calib.cameraMatrix.at<double>(0, 0);
        double fy = calib.cameraMatrix.at<double>(1, 1);
        double aspectRatio = fx / fy;
        
        cout << left << setw(10) << calib.cameraIndex
             << setw(15) << fixed << setprecision(2) << fx
             << setw(15) << fixed << setprecision(2) << fy
             << setw(15) << fixed << setprecision(4) << aspectRatio << endl;
    }
    
    // Principal point
    cout << "\n3. PRINCIPAL POINT (Optical Center)" << endl;
    cout << string(80, '-') << endl;
    cout << left << setw(10) << "Camera"
         << setw(15) << "cx (pixels)"
         << setw(15) << "cy (pixels)"
         << setw(20) << "Offset from center" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& calib : calibrations) {
        double cx = calib.cameraMatrix.at<double>(0, 2);
        double cy = calib.cameraMatrix.at<double>(1, 2);
        double centerX = calib.imageSize.width / 2.0;
        double centerY = calib.imageSize.height / 2.0;
        double offset = sqrt(pow(cx - centerX, 2) + pow(cy - centerY, 2));
        
        cout << left << setw(10) << calib.cameraIndex
             << setw(15) << fixed << setprecision(2) << cx
             << setw(15) << fixed << setprecision(2) << cy
             << setw(20) << fixed << setprecision(2) << offset << " px" << endl;
    }
    
    // Distortion coefficients
    cout << "\n4. DISTORTION COEFFICIENTS" << endl;
    cout << string(80, '-') << endl;
    cout << left << setw(10) << "Camera"
         << setw(12) << "k1"
         << setw(12) << "k2"
         << setw(12) << "p1"
         << setw(12) << "p2"
         << setw(12) << "k3" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& calib : calibrations) {
        cout << left << setw(10) << calib.cameraIndex
             << setw(12) << fixed << setprecision(6) << calib.distCoeffs.at<double>(0)
             << setw(12) << fixed << setprecision(6) << calib.distCoeffs.at<double>(1)
             << setw(12) << fixed << setprecision(6) << calib.distCoeffs.at<double>(2)
             << setw(12) << fixed << setprecision(6) << calib.distCoeffs.at<double>(3)
             << setw(12) << fixed << setprecision(6) << calib.distCoeffs.at<double>(4) << endl;
    }
    
    // Analysis and recommendations
    cout << "\n5. ANALYSIS & RECOMMENDATIONS" << endl;
    cout << string(80, '-') << endl;
    
    // Find best camera by reprojection error
    int bestCamera = 0;
    double bestError = calibrations[0].reprojectionError;
    for (size_t i = 1; i < calibrations.size(); i++) {
        if (calibrations[i].reprojectionError < bestError) {
            bestError = calibrations[i].reprojectionError;
            bestCamera = i;
        }
    }
    
    cout << "\n✓ BEST CAMERA (lowest reprojection error):" << endl;
    cout << "  Camera " << calibrations[bestCamera].cameraIndex
         << " with error of " << fixed << setprecision(4)
         << calibrations[bestCamera].reprojectionError << " pixels" << endl;
    
    // Distortion analysis
    cout << "\n✓ DISTORTION ANALYSIS:" << endl;
    for (const auto& calib : calibrations) {
        double k1 = abs(calib.distCoeffs.at<double>(0));
        double k2 = abs(calib.distCoeffs.at<double>(1));
        double totalRadial = k1 + k2;
        
        cout << "  Camera " << calib.cameraIndex << ": ";
        if (totalRadial < 0.1) {
            cout << "Low distortion (good quality lens)" << endl;
        } else if (totalRadial < 0.3) {
            cout << "Moderate distortion (typical webcam)" << endl;
        } else {
            cout << "High distortion (correction recommended)" << endl;
        }
    }
    
    cout << "\n" << string(80, '=') << endl;
    
    // Save report to file
    ofstream reportFile("camera_comparison_report.txt");
    if (reportFile.is_open()) {
        reportFile << "CAMERA CALIBRATION COMPARISON REPORT\n";
        reportFile << "Generated: November 2025\n\n";
        
        for (const auto& calib : calibrations) {
            reportFile << "Camera " << calib.cameraIndex << ":\n";
            reportFile << "  Resolution: " << calib.imageSize << "\n";
            reportFile << "  Reprojection Error: " << calib.reprojectionError << " pixels\n";
            reportFile << "  Focal Length: fx=" << calib.cameraMatrix.at<double>(0,0)
                      << ", fy=" << calib.cameraMatrix.at<double>(1,1) << "\n";
            reportFile << "\n";
        }
        
        reportFile.close();
        cout << "\n✓ Report saved to: camera_comparison_report.txt" << endl;
    }
}

int main() {
    cout << "=== Camera Calibration Comparison Tool ===" << endl;
    cout << "\nThis tool will calibrate all available cameras and compare them." << endl;
    
    // Detect cameras
    vector<int> cameras = detectCameras();
    
    if (cameras.empty()) {
        cerr << "\nERROR: No cameras detected!" << endl;
        return -1;
    }
    
    cout << "\nFound " << cameras.size() << " camera(s)" << endl;
    
    // Calibrate each camera
    vector<CameraCalibration> calibrations;
    
    for (int cameraIndex : cameras) {
        CameraCalibration calib;
        
        if (captureCalibrationImages(cameraIndex, calib)) {
            if (performCalibration(calib)) {
                saveCalibration(calib);
                calibrations.push_back(calib);
            }
        }
        
        cout << "\nPress ENTER to continue to next camera (or ESC to finish)..." << endl;
        int key = waitKey(0);
        if (key == 27) {
            break;
        }
    }
    
    // Generate comparison report
    if (!calibrations.empty()) {
        generateComparisonReport(calibrations);
    }
    
    return 0;
}
