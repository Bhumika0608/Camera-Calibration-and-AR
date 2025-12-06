/*
 * Ishan Chaudhary, Bhumika Yadav
 * Fall 2025
 * CS 5330 Computer Vision
 * 
 Task 6: Virtual Object Projection
 ---------------------------------------------------------
 * Projects 3D virtual house with pyramid roof onto checkerboard pattern.
 * Supports both live camera and static image modes with auto-scaling calibration.
 * 
 * Usage: task6_virtual_object.exe [image_path]
 * Controls: ESC=Exit, s=Screenshot
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Read camera calibration from YAML file
bool readCameraParameters(const string &filename, Mat &cameraMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open camera parameters file: " << filename << endl;
        return false;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();
    return true;
}

// Create 3D house structure (base, walls, roof, chimney, door)
void createVirtualObject(vector<pair<Point3f, Point3f>> &lines) {
    
    float centerX = 4.5f;
    float centerY = 2.5f;
    float baseSize = 3.0f;
    float baseZ = -3.0f;  // Float above board
    float wallHeight = 3.0f;
    float roofHeight = 3.0f;
    
    // Base square corners
    Point3f base_tl(centerX - baseSize/2, centerY - baseSize/2, baseZ);
    Point3f base_tr(centerX + baseSize/2, centerY - baseSize/2, baseZ);
    Point3f base_bl(centerX - baseSize/2, centerY + baseSize/2, baseZ);
    Point3f base_br(centerX + baseSize/2, centerY + baseSize/2, baseZ);
    
    // Base square edges
    lines.push_back({base_tl, base_tr});
    lines.push_back({base_tr, base_br});
    lines.push_back({base_br, base_bl});
    lines.push_back({base_bl, base_tl});
    
    // Wall corners
    float wallTop = baseZ - wallHeight;
    Point3f wall_tl(centerX - baseSize/2, centerY - baseSize/2, wallTop);
    Point3f wall_tr(centerX + baseSize/2, centerY - baseSize/2, wallTop);
    Point3f wall_bl(centerX - baseSize/2, centerY + baseSize/2, wallTop);
    Point3f wall_br(centerX + baseSize/2, centerY + baseSize/2, wallTop);
    
    // Vertical wall edges
    lines.push_back({base_tl, wall_tl});
    lines.push_back({base_tr, wall_tr});
    lines.push_back({base_bl, wall_bl});
    lines.push_back({base_br, wall_br});
    
    // Top of walls square
    lines.push_back({wall_tl, wall_tr});
    lines.push_back({wall_tr, wall_br});
    lines.push_back({wall_br, wall_bl});
    lines.push_back({wall_bl, wall_tl});
    
    // Pyramid roof
    float roofApexZ = wallTop - roofHeight;
    Point3f apex(centerX + 0.5f, centerY - 0.3f, roofApexZ);
    
    // Roof edges
    lines.push_back({wall_tl, apex});
    lines.push_back({wall_tr, apex});
    lines.push_back({wall_bl, apex});
    lines.push_back({wall_br, apex});
    
    // Chimney
    float chimneyWidth = 0.6f;
    float chimneyHeight = 1.5f;
    Point3f chimney_base1(centerX + baseSize/2 - 1.0f, centerY - baseSize/2, wallTop);
    Point3f chimney_base2(centerX + baseSize/2 - 1.0f + chimneyWidth, centerY - baseSize/2, wallTop);
    Point3f chimney_top1(centerX + baseSize/2 - 1.0f, centerY - baseSize/2, wallTop - chimneyHeight);
    Point3f chimney_top2(centerX + baseSize/2 - 1.0f + chimneyWidth, centerY - baseSize/2, wallTop - chimneyHeight);
    
    lines.push_back({chimney_base1, chimney_top1});
    lines.push_back({chimney_base2, chimney_top2});
    lines.push_back({chimney_top1, chimney_top2});
    
    // Door
    float doorWidth = 1.0f;
    float doorHeight = 1.8f;
    Point3f door_bl(centerX - doorWidth/2, centerY + baseSize/2, baseZ);
    Point3f door_br(centerX + doorWidth/2, centerY + baseSize/2, baseZ);
    Point3f door_tl(centerX - doorWidth/2, centerY + baseSize/2, baseZ - doorHeight);
    Point3f door_tr(centerX + doorWidth/2, centerY + baseSize/2, baseZ - doorHeight);
    
    lines.push_back({door_bl, door_tl});
    lines.push_back({door_br, door_tr});
    lines.push_back({door_tl, door_tr});
    lines.push_back({door_bl, door_br});
}

int main(int argc, char** argv) {
    const int boardWidth = 9;
    const int boardHeight = 6;
    const float squareSize = 1.0f;
    
    // Generate 3D checkerboard points
    vector<Point3f> objectPoints;
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    
    // Load calibration
    Mat cameraMatrix, distCoeffs;
    if (!readCameraParameters("camera_intrinsics.yml", cameraMatrix, distCoeffs)) {
        return -1;
    }
    
    int calibWidth = 640;
    int calibHeight = 480;
    
    // Create virtual object
    vector<pair<Point3f, Point3f>> virtualObjectLines;
    createVirtualObject(virtualObjectLines);
    
    // Check mode
    bool staticImageMode = (argc > 1);
    
    // Static image mode
    if (staticImageMode) {
        Mat frame = imread(argv[1]);
        if (frame.empty()) {
            cerr << "Error: Could not load image" << endl;
            return -1;
        }
        
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Detect checkerboard
        vector<Point2f> corners2D;
        bool found = findChessboardCorners(gray, Size(boardWidth, boardHeight), corners2D,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        
        if (found) {
            // Auto-scale calibration for different resolutions
            Mat scaledCameraMatrix = cameraMatrix.clone();
            if (frame.cols != calibWidth || frame.rows != calibHeight) {
                double scaleX = (double)frame.cols / calibWidth;
                double scaleY = (double)frame.rows / calibHeight;
                
                scaledCameraMatrix.at<double>(0, 0) *= scaleX;  // fx
                scaledCameraMatrix.at<double>(1, 1) *= scaleY;  // fy
                scaledCameraMatrix.at<double>(0, 2) *= scaleX;  // cx
                scaledCameraMatrix.at<double>(1, 2) *= scaleY;  // cy
            }
            
            // Refine corners
            cornerSubPix(gray, corners2D, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            
            drawChessboardCorners(frame, Size(boardWidth, boardHeight), corners2D, found);
            
            // Solve pose and project virtual object
            Mat rvec, tvec;
            solvePnP(objectPoints, corners2D, scaledCameraMatrix, distCoeffs, rvec, tvec);
            
            // Project virtual object edges
            for (const auto &line : virtualObjectLines) {
                vector<Point3f> points3D = {line.first, line.second};
                vector<Point2f> points2D;
                projectPoints(points3D, rvec, tvec, scaledCameraMatrix, distCoeffs, points2D);
                
                if (points2D.size() == 2) {
                    cv::line(frame, points2D[0], points2D[1], Scalar(255, 255, 0), 3, LINE_AA);
                }
            }
            
            // Draw coordinate axes
            vector<Point3f> axisPoints = {
                Point3f(0, 0, 0),
                Point3f(2*squareSize, 0, 0),
                Point3f(0, 2*squareSize, 0),
                Point3f(0, 0, -2*squareSize)
            };
            vector<Point2f> imageAxisPoints;
            projectPoints(axisPoints, rvec, tvec, scaledCameraMatrix, distCoeffs, imageAxisPoints);
            
            cv::line(frame, imageAxisPoints[0], imageAxisPoints[1], Scalar(0, 0, 255), 2);
            cv::line(frame, imageAxisPoints[0], imageAxisPoints[2], Scalar(0, 255, 0), 2);
            cv::line(frame, imageAxisPoints[0], imageAxisPoints[3], Scalar(255, 0, 0), 2);
            
            // Save output
            string outputPath = string(argv[1]);
            size_t dotPos = outputPath.find_last_of(".");
            string outputFilename = outputPath.substr(0, dotPos) + "_with_ar" + outputPath.substr(dotPos);
            imwrite(outputFilename, frame);
            
            imshow("Static Image AR", frame);
            waitKey(0);
        } else {
            cerr << "No checkerboard detected" << endl;
            return -1;
        }
        
        return 0;
    }
    
    // Live camera mode
    vector<int> availableCameras;
    
    for (int i = 0; i < 5; i++) {
        VideoCapture testCap(i);
        if (testCap.isOpened()) {
            Mat testFrame;
            testCap >> testFrame;
            if (!testFrame.empty()) {
                availableCameras.push_back(i);
            }
            testCap.release();
        }
    }
    
    if (availableCameras.empty()) {
        cerr << "ERROR: No cameras found" << endl;
        return -1;
    }
    
    // Select camera
    int cameraIndex;
    if (availableCameras.size() == 1) {
        cameraIndex = availableCameras[0];
    } else {
        cout << "Enter camera index: ";
        cin >> cameraIndex;
    }
    
    // Open camera
    VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        cerr << "Failed to open camera" << endl;
        return -1;
    }
    
    int screenshotCount = 0;
    
    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;
        
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        vector<Point2f> corners2D;
        bool found = findChessboardCorners(gray, Size(boardWidth, boardHeight), corners2D,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        
        if (found) {
            cornerSubPix(gray, corners2D, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            
            drawChessboardCorners(frame, Size(boardWidth, boardHeight), corners2D, found);
            
            Mat rvec, tvec;
            solvePnP(objectPoints, corners2D, cameraMatrix, distCoeffs, rvec, tvec);
            
            // Project virtual object
            for (const auto &line : virtualObjectLines) {
                vector<Point3f> points3D = {line.first, line.second};
                vector<Point2f> points2D;
                projectPoints(points3D, rvec, tvec, cameraMatrix, distCoeffs, points2D);
                
                if (points2D.size() == 2) {
                    cv::line(frame, points2D[0], points2D[1], Scalar(255, 255, 0), 2, LINE_AA);
                }
            }
            
            // Draw axes
            vector<Point3f> axisPoints = {
                Point3f(0, 0, 0),
                Point3f(2*squareSize, 0, 0),
                Point3f(0, 2*squareSize, 0),
                Point3f(0, 0, -2*squareSize)
            };
            vector<Point2f> imageAxisPoints;
            projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imageAxisPoints);
            
            cv::line(frame, imageAxisPoints[0], imageAxisPoints[1], Scalar(0, 0, 255), 2);
            cv::line(frame, imageAxisPoints[0], imageAxisPoints[2], Scalar(0, 255, 0), 2);
            cv::line(frame, imageAxisPoints[0], imageAxisPoints[3], Scalar(255, 0, 0), 2);
        }
        
        imshow("Virtual Object", frame);
        
        char key = (char)waitKey(30);
        if (key == 27) break;
        else if (key == 's' || key == 'S') {
            screenshotCount++;
            string filename = "virtual_object_screenshot_" + to_string(screenshotCount) + ".png";
            imwrite(filename, frame);
        }
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}
