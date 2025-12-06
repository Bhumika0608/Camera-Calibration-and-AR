/*
  Bhumika Yadav, Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision

  Task 2: Data Collection for Camera Calibration
  ---------------------------------------------------------
  Captures multiple frames from a webcam containing a visible
  checkerboard pattern. Detects and highlights internal corners
  using findChessboardCorners() and saves valid frames for
  calibration.

  Ensures that a sufficient number of diverse viewpoints are
  collected for accurate camera parameter estimation.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Define checkerboard dimensions (number of internal corners)
    const int CHECKERBOARD[2]{9, 6};  // 9 columns, 6 rows

    // Storage for calibration data
    std::vector<std::vector<cv::Point2f>> corner_list;   // 2D image points
    std::vector<std::vector<cv::Vec3f>> point_list;      // 3D world points

    // Prepare the 3D coordinates of the chessboard corners (assuming each square = 1 unit)
    std::vector<cv::Vec3f> point_set;
    for (int i = 0; i < CHECKERBOARD[1]; i++) {         // rows
        for (int j = 0; j < CHECKERBOARD[0]; j++) {     // cols
            point_set.push_back(cv::Vec3f(j, -i, 0));   // y is negative to go downward in image
        }
    }

    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera.\n";
        return -1;
    }

    std::cout << "Press 's' to save a calibration frame, 'q' to quit.\n";

    cv::Mat frame, gray;
    std::vector<cv::Point2f> corner_set;
    bool found = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Find the chessboard corners
        found = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_set,
                                          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // Refine corner positions
            cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            // Draw corners on the frame
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_set, found);

            std::cout << "Corners found: " << corner_set.size()
                      << " | First corner: (" << corner_set[0].x << ", " << corner_set[0].y << ")\n";
        }

        cv::imshow("Calibration", frame);
        char key = (char)cv::waitKey(1);

        // --- SAVE CALIBRATION FRAME ---
        if (key == 's' || key == 'S') {
            if (found) {
                corner_list.push_back(corner_set);
                point_list.push_back(point_set);

                std::cout << "Calibration frame saved! Total frames: " << corner_list.size() << "\n";

                // save the image
                std::string filename = "calib_frame_" + std::to_string(corner_list.size()) + ".jpg";
                cv::imwrite(filename, frame);
                std::cout << "Saved image: " << filename << "\n";
            } else {
                std::cout << "No checkerboard detected — frame not saved.\n";
            }
        }

        if (key == 'q' || key == 'Q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    // --- Summary ---
    std::cout << "\nCalibration data collection complete.\n";
    std::cout << "Total frames saved: " << corner_list.size() << "\n";
    std::cout << "Each frame has " << point_set.size() << " 3D points and " 
              << (corner_list.empty() ? 0 : corner_list[0].size()) << " 2D points.\n";

    return 0;
}
