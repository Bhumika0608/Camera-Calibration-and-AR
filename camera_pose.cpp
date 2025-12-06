/*
  Bhumika Yadav, Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision

  Task 4: Pose Estimation
  Detects checkerboard corners in calibration frames and computes the camera's rotation (pitch, yaw, roll)
  and translation (Tx, Ty, Tz) relative to the pattern in real time. 
  Displays these values as the camera moves, allowing observation of pose changes 
  as the target is shifted side to side or rotated.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

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

// Function to convert rotation vector to Euler angles (degrees)
Vec3f rotationVectorToEulerAngles(const Mat &rvec) {
    Mat R;
    Rodrigues(rvec, R);

    double sy = sqrt(R.at<double>(0,0)*R.at<double>(0,0) + R.at<double>(1,0)*R.at<double>(1,0));
    double x, y, z;

    if (sy >= 1e-6) {
        x = atan2(R.at<double>(2,1), R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }

    x = x * 180.0 / CV_PI;
    y = y * 180.0 / CV_PI;
    z = z * 180.0 / CV_PI;

    return Vec3f(x, y, z);
}

int main() {
    // Checkerboard dimensions (internal corners)
    const int boardWidth = 9;
    const int boardHeight = 6;
    const float squareSize = 1.0f;

    // Prepare 3D object points for checkerboard corners
    vector<Point3f> objectPoints;
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    // Load camera calibration
    Mat cameraMatrix, distCoeffs;
    if (!readCameraParameters("camera_intrinsics.yml", cameraMatrix, distCoeffs)) {
        return -1;
    }

    // Open video capture
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera" << endl;
        return -1;
    }

    // Open CSV file for writing
    ofstream csvFile("camera_pose_log.csv");
    csvFile << "Frame,Pitch,Yaw,Roll,Tx,Ty,Tz\n";

    int frameCount = 0;

    while (true) {
        Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, Size(boardWidth, boardHeight), corners,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            drawChessboardCorners(frame, Size(boardWidth, boardHeight), corners, found);

            Mat rvec, tvec;
            solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            Vec3f eulerAngles = rotationVectorToEulerAngles(rvec);

            // Print to console
            cout << "Frame " << frameCount << ": ";
            cout << "Pitch: " << eulerAngles[0] << "°, Yaw: " << eulerAngles[1] << "°, Roll: " << eulerAngles[2] << "°" << endl;
            cout << "Translation: [" << tvec.at<double>(0) << ", " 
                                      << tvec.at<double>(1) << ", " 
                                      << tvec.at<double>(2) << "]\n" << endl;

            // Write to CSV
            csvFile << frameCount << ","
                    << eulerAngles[0] << "," 
                    << eulerAngles[1] << "," 
                    << eulerAngles[2] << ","
                    << tvec.at<double>(0) << "," 
                    << tvec.at<double>(1) << "," 
                    << tvec.at<double>(2) << "\n";

            // Draw 3D axes
            vector<Point3f> axisPoints = {Point3f(0,0,0), Point3f(3*squareSize,0,0),
                                          Point3f(0,3*squareSize,0), Point3f(0,0,-3*squareSize)};
            vector<Point2f> imagePoints;
            projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

            line(frame, imagePoints[0], imagePoints[1], Scalar(0,0,255), 2);
            line(frame, imagePoints[0], imagePoints[2], Scalar(0,255,0), 2);
            line(frame, imagePoints[0], imagePoints[3], Scalar(255,0,0), 2);
        }

        imshow("Checkerboard Pose Estimation", frame);
        char key = (char)waitKey(30);
        if (key == 27) break;

        frameCount++;
    }

    csvFile.close();
    cap.release();
    destroyAllWindows();
    return 0;
}
