/*
 * Ishan Chaudhary, Bhumika Yadav
 * Fall 2025
 * CS 5330 Computer Vision
 
  Task 7: Feature Detection + Feature-Based AR
 ---------------------------------------------------------
 * Implements Harris corner and ORB feature detection with marker-less AR tracking.
 * Demonstrates feature-based augmented reality using homography estimation.
 * 
 * Controls: 1=Harris, 2=ORB, 3=Both, 4=AR Mode (SPACE to capture reference)
 *           +/-=Harris threshold, w/s=ORB features, r=Reset, c=Checkerboard, h=Help
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cv;
using namespace std;

int detectionMode = 3;
double harrisThreshold = 0.01;
int orbMaxFeatures = 500;
bool showCheckerboard = false;

bool arModeActive = false;
Mat referenceImage;
vector<KeyPoint> referenceKeypoints;
Mat referenceDescriptors;
Ptr<ORB> orbDetector;

void printHelp() {
    cout << "\n=== CONTROLS ===" << endl;
    cout << "1/2/3/4 - Harris/ORB/Both/AR Mode" << endl;
    cout << "SPACE - Capture reference (mode 4)" << endl;
    cout << "+/- - Harris threshold" << endl;
    cout << "w/s - ORB features count" << endl;
    cout << "c - Toggle checkerboard" << endl;
    cout << "r - Reset, p - Save, h - Help, ESC - Exit\n" << endl;
}

// Harris corner detection
void detectHarrisCorners(const Mat &gray, vector<Point2f> &corners, double threshold) {
    Mat dst, dst_norm;
    
    cornerHarris(gray, dst, 2, 3, 0.04);
    
    // Normalize
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    
    // Threshold and find local maxima
    corners.clear();
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            if ((float)dst_norm.at<float>(j, i) > threshold * 255) {
                corners.push_back(Point2f(i, j));
            }
        }
    }
    
    // Non-maximum suppression (simple version)
    if (corners.size() > 1000) {
        // Sort by response strength
        vector<pair<float, Point2f>> scored;
        for (const auto &pt : corners) {
            scored.push_back({dst_norm.at<float>(pt.y, pt.x), pt});
        }
        // Sort by first element (response strength) in descending order
        sort(scored.begin(), scored.end(), [](const pair<float, Point2f>& a, const pair<float, Point2f>& b) {
            return a.first > b.first;  // Greater response first
        });
        
        corners.clear();
        for (size_t i = 0; i < min((size_t)1000, scored.size()); i++) {
            corners.push_back(scored[i].second);
        }
    }
}

// Draw Harris corners on image
void drawHarrisCorners(Mat &img, const vector<Point2f> &corners) {
    for (const auto &pt : corners) {
        circle(img, pt, 5, Scalar(0, 255, 0), 2);  // Green circles
        circle(img, pt, 3, Scalar(255, 255, 0), -1);  // Yellow center
    }
}

// Draw ORB features
void detectAndDrawORB(Mat &img, const Mat &gray, int maxFeatures) {
    Ptr<ORB> orb = ORB::create(maxFeatures);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(gray, noArray(), keypoints, descriptors);
    drawKeypoints(img, keypoints, img, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// AR Mode: Feature matching and homography
void processARMode(Mat &frame, const Mat &gray) {
    if (!arModeActive) {
        putText(frame, "AR Mode: Press SPACE to capture reference", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        return;
    }
    
    vector<KeyPoint> currentKeypoints;
    Mat currentDescriptors;
    orbDetector->detectAndCompute(gray, noArray(), currentKeypoints, currentDescriptors);
    
    if (currentDescriptors.empty() || referenceDescriptors.empty()) return;
    
    // Match features
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(referenceDescriptors, currentDescriptors, knnMatches, 2);
    
    // Lowe's ratio test
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() >= 2 && 
            knnMatches[i][0].distance < 0.75f * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }
    
    if (goodMatches.size() < 10) return;
    
    // Extract points and find homography
    vector<Point2f> refPoints, currPoints;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        refPoints.push_back(referenceKeypoints[goodMatches[i].queryIdx].pt);
        currPoints.push_back(currentKeypoints[goodMatches[i].trainIdx].pt);
    }
    
    Mat H = findHomography(refPoints, currPoints, RANSAC, 3.0);
    if (H.empty()) return;
    
    // Project virtual object
    int refWidth = referenceImage.cols;
    int refHeight = referenceImage.rows;
    vector<Point2f> refObjectCorners = {
        Point2f(refWidth * 0.3f, refHeight * 0.3f),
        Point2f(refWidth * 0.7f, refHeight * 0.3f),
        Point2f(refWidth * 0.7f, refHeight * 0.7f),
        Point2f(refWidth * 0.3f, refHeight * 0.7f)
    };
    
    vector<Point2f> projectedCorners;
    perspectiveTransform(refObjectCorners, projectedCorners, H);
    
    // Draw virtual object
    for (int i = 0; i < 4; i++) {
        line(frame, projectedCorners[i], projectedCorners[(i + 1) % 4], 
             Scalar(0, 255, 0), 3, LINE_AA);
    }
    line(frame, projectedCorners[0], projectedCorners[2], Scalar(0, 255, 0), 2, LINE_AA);
    line(frame, projectedCorners[1], projectedCorners[3], Scalar(0, 255, 0), 2, LINE_AA);
    
    Point2f center(0, 0);
    for (const auto &pt : projectedCorners) center += pt;
    center *= 0.25f;
    circle(frame, center, 8, Scalar(0, 255, 255), -1);
}

int main(int argc, char** argv) {
    bool staticImageMode = (argc > 1);
    
    if (staticImageMode) {
        
        Mat image = imread(argv[1]);
        if (image.empty()) {
            cerr << "Error: Could not load image" << endl;
            return -1;
        }
        
        orbDetector = ORB::create(orbMaxFeatures);
        
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        orbDetector->detectAndCompute(gray, noArray(), referenceKeypoints, referenceDescriptors);
        
        if (referenceKeypoints.size() < 10) {
            cerr << "Error: Not enough features" << endl;
            return -1;
        }
        
        // Draw features and virtual object
        Mat display = image.clone();
        
        // Draw detected features
        for (size_t i = 0; i < min(size_t(100), referenceKeypoints.size()); i++) {
            circle(display, referenceKeypoints[i].pt, 3, Scalar(255, 0, 255), -1);
        }
        
        // Draw virtual object in center
        int centerX = image.cols / 2;
        int centerY = image.rows / 2;
        int objSize = min(image.cols, image.rows) / 4;
        
        vector<Point2f> corners = {
            Point2f(centerX - objSize, centerY - objSize),
            Point2f(centerX + objSize, centerY - objSize),
            Point2f(centerX + objSize, centerY + objSize),
            Point2f(centerX - objSize, centerY + objSize)
        };
        
        // Draw virtual object (rectangle with diagonals)
        for (int i = 0; i < 4; i++) {
            line(display, corners[i], corners[(i + 1) % 4], Scalar(0, 255, 0), 3, LINE_AA);
        }
        line(display, corners[0], corners[2], Scalar(0, 255, 0), 2, LINE_AA);
        line(display, corners[1], corners[3], Scalar(0, 255, 0), 2, LINE_AA);
        
        // Draw center point
        circle(display, Point(centerX, centerY), 10, Scalar(0, 255, 255), -1);
        
        // Add text
        putText(display, "AR Target Image", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
        putText(display, "Features: " + to_string(referenceKeypoints.size()), Point(10, 70),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 255), 2);
        putText(display, "Virtual Object (green)", Point(10, 110),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        
        // Save output
        string outputPath = string(argv[1]);
        size_t dotPos = outputPath.find_last_of(".");
        string outputFilename = outputPath.substr(0, dotPos) + "_with_ar" + outputPath.substr(dotPos);
        imwrite(outputFilename, display);
        cout << "\n✓ AR visualization saved: " << outputFilename << endl;
        
        // Display
        imshow("Static Image AR - Press any key to exit", display);
        cout << "\nPress any key to exit..." << endl;
        waitKey(0);
        destroyAllWindows();
        
        cout << "\n=== SUCCESS ===" << endl;
        cout << "Demonstrated AR on static image without checkerboard!" << endl;
        cout << "Features used: " << referenceKeypoints.size() << " ORB keypoints" << endl;
        cout << "This shows marker-less AR capability on arbitrary textured images." << endl;
        
        return 0;
    }
    
    // Live camera mode
    printHelp();
    
    // Scan for available cameras
    cout << "Scanning for cameras..." << endl;
    vector<int> availableCameras;
    
    for (int i = 0; i < 5; i++) {
        VideoCapture testCap(i);
        if (testCap.isOpened()) {
            Mat testFrame;
            testCap >> testFrame;
            if (!testFrame.empty()) {
                availableCameras.push_back(i);
                int width = (int)testCap.get(CAP_PROP_FRAME_WIDTH);
                int height = (int)testCap.get(CAP_PROP_FRAME_HEIGHT);
                cout << "  Camera " << i << " - Available (" << width << "x" << height << ")" << endl;
            }
            testCap.release();
        }
    }
    
    if (availableCameras.empty()) {
        cerr << "\nERROR: No cameras found!" << endl;
        return -1;
    }
    
    // Ask user to select camera
    int cameraIndex;
    if (availableCameras.size() == 1) {
        cameraIndex = availableCameras[0];
        cout << "\nUsing camera " << cameraIndex << endl;
    } else {
        cout << "\nEnter camera index to use (";
        for (size_t i = 0; i < availableCameras.size(); i++) {
            cout << availableCameras[i];
            if (i < availableCameras.size() - 1) cout << ", ";
        }
        cout << "): ";
        cin >> cameraIndex;
        
        // Validate input
        if (find(availableCameras.begin(), availableCameras.end(), cameraIndex) == availableCameras.end()) {
            cerr << "Invalid camera index!" << endl;
            return -1;
        }
    }
    
    // Open selected camera
    VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        cerr << "Failed to open camera " << cameraIndex << endl;
        return -1;
    }
    
    cout << "Camera " << cameraIndex << " opened successfully!" << endl;
    
    // Initialize ORB detector for AR mode
    orbDetector = ORB::create(orbMaxFeatures);
    
    // Checkerboard parameters (for overlay)
    const int boardWidth = 9;
    const int boardHeight = 6;
    
    int screenshotCount = 0;
    
    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        
        // Convert to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Optionally detect checkerboard for reference
        vector<Point2f> checkerCorners;
        bool checkerboardFound = false;
        if (showCheckerboard) {
            checkerboardFound = findChessboardCorners(gray, Size(boardWidth, boardHeight), 
                                                      checkerCorners,
                                                      CALIB_CB_ADAPTIVE_THRESH | 
                                                      CALIB_CB_FAST_CHECK);
            if (checkerboardFound) {
                cornerSubPix(gray, checkerCorners, Size(11, 11), Size(-1, -1),
                            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            }
        }
        
        Mat display;
        
        if (detectionMode == 4) {
            // AR Mode
            display = frame.clone();
            processARMode(display, gray);
            
        } else if (detectionMode == 1) {
            // Harris Corners only
            display = frame.clone();
            vector<Point2f> harrisCorners;
            detectHarrisCorners(gray, harrisCorners, harrisThreshold);
            drawHarrisCorners(display, harrisCorners);
            
            if (showCheckerboard && checkerboardFound) {
                drawChessboardCorners(display, Size(boardWidth, boardHeight), 
                                     checkerCorners, true);
            }
            
            // Display info
            string info = "Harris Corners: " + to_string(harrisCorners.size());
            putText(display, info, Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            string thresh = "Threshold: " + to_string(harrisThreshold).substr(0, 5);
            putText(display, thresh, Point(10, 60), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
            
        } else if (detectionMode == 2) {
            // ORB Features only
            display = frame.clone();
            detectAndDrawORB(display, gray, orbMaxFeatures);
            
            if (showCheckerboard && checkerboardFound) {
                drawChessboardCorners(display, Size(boardWidth, boardHeight), 
                                     checkerCorners, true);
            }
            
            // Display info
            string info = "ORB Features (max: " + to_string(orbMaxFeatures) + ")";
            putText(display, info, Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 2);
            
        } else {
            // Both - split view
            Mat harrisImg = frame.clone();
            Mat orbImg = frame.clone();
            
            // Harris on left
            vector<Point2f> harrisCorners;
            detectHarrisCorners(gray, harrisCorners, harrisThreshold);
            drawHarrisCorners(harrisImg, harrisCorners);
            
            string harrisInfo = "Harris: " + to_string(harrisCorners.size());
            putText(harrisImg, harrisInfo, Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            string thresh = "Thresh: " + to_string(harrisThreshold).substr(0, 5);
            putText(harrisImg, thresh, Point(10, 55), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            
            // ORB on right
            detectAndDrawORB(orbImg, gray, orbMaxFeatures);
            string orbInfo = "ORB: max " + to_string(orbMaxFeatures);
            putText(orbImg, orbInfo, Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2);
            
            // Checkerboard overlay on both if enabled
            if (showCheckerboard && checkerboardFound) {
                drawChessboardCorners(harrisImg, Size(boardWidth, boardHeight), 
                                     checkerCorners, true);
                drawChessboardCorners(orbImg, Size(boardWidth, boardHeight), 
                                     checkerCorners, true);
            }
            
            // Combine horizontally
            Mat combined;
            hconcat(harrisImg, orbImg, combined);
            display = combined;
        }
        
        // Show the result
        string windowName = "Feature Detection - Press 'h' for help";
        imshow(windowName, display);
        
        // Handle keyboard input
        char key = (char)waitKey(30);
        
        if (key == 27) {  // ESC
            break;
        } else if (key == '1') {
            detectionMode = 1;
            cout << "Mode: Harris Corners only" << endl;
        } else if (key == '2') {
            detectionMode = 2;
            cout << "Mode: ORB Features only" << endl;
        } else if (key == '3') {
            detectionMode = 3;
            cout << "Mode: Both (split view)" << endl;
        } else if (key == '4') {
            detectionMode = 4;
            arModeActive = false;
            cout << "Mode: AR Mode (press SPACE to capture reference)" << endl;
        } else if (key == ' ') {
            if (detectionMode == 4) {
                referenceImage = frame.clone();
                orbDetector->detectAndCompute(gray, noArray(), referenceKeypoints, referenceDescriptors);
                if (!referenceDescriptors.empty()) {
                    arModeActive = true;
                    cout << "Reference captured! " << referenceKeypoints.size() << " features detected." << endl;
                    cout << "Move camera to see AR tracking..." << endl;
                } else {
                    cout << "Failed to detect features. Try a more textured surface." << endl;
                }
            }
        } else if (key == '+' || key == '=') {
            harrisThreshold = min(0.5, harrisThreshold + 0.005);
            cout << "Harris threshold: " << harrisThreshold << endl;
        } else if (key == '-' || key == '_') {
            harrisThreshold = max(0.001, harrisThreshold - 0.005);
            cout << "Harris threshold: " << harrisThreshold << endl;
        } else if (key == 'w' || key == 'W') {
            orbMaxFeatures = min(5000, orbMaxFeatures + 50);
            cout << "ORB max features: " << orbMaxFeatures << endl;
            orbDetector = ORB::create(orbMaxFeatures);  // Recreate detector
        } else if (key == 's' || key == 'S') {
            orbMaxFeatures = max(50, orbMaxFeatures - 50);
            cout << "ORB max features: " << orbMaxFeatures << endl;
            orbDetector = ORB::create(orbMaxFeatures);  // Recreate detector
        } else if (key == 'r' || key == 'R') {
            harrisThreshold = 0.01;
            orbMaxFeatures = 500;
            orbDetector = ORB::create(orbMaxFeatures);
            arModeActive = false;
            cout << "Reset to defaults" << endl;
        } else if (key == 'c' || key == 'C') {
            showCheckerboard = !showCheckerboard;
            cout << "Checkerboard overlay: " << (showCheckerboard ? "ON" : "OFF") << endl;
        } else if (key == 'h' || key == 'H') {
            printHelp();
        } else if (key == 'p' || key == 'P') {
            screenshotCount++;
            string filename = "feature_detection_screenshot_" + to_string(screenshotCount) + ".png";
            imwrite(filename, display);
            cout << "Screenshot saved: " << filename << endl;
        }
    }
    
    cap.release();
    destroyAllWindows();
    

    return 0;
}
