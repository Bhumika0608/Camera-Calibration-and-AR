/*
  Bhumika Yadav, Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision

  Task 1: Target Detection
  ---------------------------------------------------------
  Detects a calibration target (checkerboard pattern) in live
  video using OpenCV's findChessboardCorners() function.
  Displays the detected corner points on the video stream in
  real time.

  Used to verify that the camera can consistently locate the
  calibration target before proceeding to the calibration step.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // --- Define the checkerboard dimensions ---
    // These are the number of internal corners per row and column
    cv::Size patternSize(9, 6);  // 9 columns and 6 rows of internal corners

    // Create vectors to store detected corners
    std::vector<cv::Point2f> corner_set;

    // --- Open video stream (0 = default camera) ---
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera.\n";
        return -1;
    }

    std::cout << "Press 'q' to quit.\n";

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Try to find the checkerboard corners
        bool found = cv::findChessboardCorners(gray, patternSize, corner_set,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // Refine corner locations for subpixel accuracy
            cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // Draw corners on the image
            cv::drawChessboardCorners(frame, patternSize, corner_set, found);

            // Print info
            std::cout << "Corners found: " << corner_set.size() << std::endl;
            if (!corner_set.empty()) {
                std::cout << "First corner: (" << corner_set[0].x << ", " << corner_set[0].y << ")\n";
            }
        }

        // Display result
        cv::imshow("Checkerboard Detection", frame);

        // Exit when 'q' is pressed
        if (cv::waitKey(10) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
