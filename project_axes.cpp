/*
  Bhumika Yadav, Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision

  Task 5: 3D Reprojection and Visualization
  Projects 3D axes and checkerboard corners back into the image using the camera’s calibration parameters. 
  Displays the reprojected axes aligned with the detected checkerboard in real time, and 
  saves screenshots showing correct alignment between 3D projections and image corners.
*/


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to read camera parameters from a YAML file
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

int main() {
    // Checkerboard dimensions
    const int boardWidth = 9;
    const int boardHeight = 6;
    const float squareSize = 1.0f;

    // Prepare 3D object points for checkerboard corners
    vector<Point3f> objectPoints;
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            objectPoints.push_back(Point3f(j*squareSize, i*squareSize, 0));
        }
    }

    // Define 3D points to project: 4 corners
    vector<Point3f> corners3D;
    corners3D.push_back(Point3f(0,0,0)); // top-left
    corners3D.push_back(Point3f((boardWidth-1)*squareSize,0,0)); // top-right
    corners3D.push_back(Point3f(0,(boardHeight-1)*squareSize,0)); // bottom-left
    corners3D.push_back(Point3f((boardWidth-1)*squareSize,(boardHeight-1)*squareSize,0)); // bottom-right

    // Load camera calibration
    Mat cameraMatrix, distCoeffs;
    if (!readCameraParameters("camera_intrinsics.yml", cameraMatrix, distCoeffs)) {
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    bool screenshotTaken = false; 

    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners2D;
        bool found = findChessboardCorners(gray, Size(boardWidth, boardHeight), corners2D,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corners2D, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            drawChessboardCorners(frame, Size(boardWidth, boardHeight), corners2D, found);

            Mat rvec, tvec;
            solvePnP(objectPoints, corners2D, cameraMatrix, distCoeffs, rvec, tvec);

            // Project the 4 corners
            vector<Point2f> projectedPoints;
            projectPoints(corners3D, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

            // Draw the projected points
            for (size_t i = 0; i < projectedPoints.size(); i++) {
                circle(frame, projectedPoints[i], 8, Scalar(0,255,255), -1); // yellow dots
            }

            // Draw 3D axes from the origin
            vector<Point3f> axisPoints = {Point3f(0,0,0), Point3f(3*squareSize,0,0),
                                          Point3f(0,3*squareSize,0), Point3f(0,0,-3*squareSize)};
            vector<Point2f> imagePoints;
            projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

            line(frame, imagePoints[0], imagePoints[1], Scalar(0,0,255), 2); // X-axis red
            line(frame, imagePoints[0], imagePoints[2], Scalar(0,255,0), 2); // Y-axis green
            line(frame, imagePoints[0], imagePoints[3], Scalar(255,0,0), 2); // Z-axis blue

            // Save a screenshot once
            if (!screenshotTaken) {
                imwrite("checkerboard_axes_screenshot.png", frame);
                cout << "Screenshot saved as checkerboard_axes_screenshot.png" << endl;
                screenshotTaken = true;
            }
        }

        imshow("Projected 3D Corners and Axes", frame);
        char key = (char)waitKey(30);
        if (key == 27) break; // ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
