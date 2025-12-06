/*
  Bhumika Yadav, Ishan Chaudhary
  Fall 2025
  CS 5330 Computer Vision

  Task 3: Camera Calibration
  ---------------------------------------------------------
  Loads saved checkerboard images from Task 2, detects corner
  points, and performs single-camera calibration using OpenCV’s
  calibrateCamera() function.

  Computes and saves the intrinsic camera matrix, distortion
  coefficients, rotation and translation vectors, and overall
  RMS reprojection error to 'camera_intrinsics.yml'.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem> 
namespace fs = std::filesystem;

int main() {
    const cv::Size CHECKERBOARD(9, 6); // internal corners (columns, rows)
    const std::string frames_dir = "calibration_frames";
    const std::string saved_data_file = "calibration_data.yml"; // saved corner/point lists
    const std::string output_file = "camera_intrinsics.yml";

    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Point3f>> point_list;

    if (fs::exists(saved_data_file)) {
        std::cout << "Found saved calibration data: " << saved_data_file << " — loading...\n";
        cv::FileStorage fsr(saved_data_file, cv::FileStorage::READ);
        if (!fsr.isOpened()) {
            std::cerr << "Failed to open " << saved_data_file << "\n";
            return -1;
        }

        // read corner_list
        cv::FileNode cn = fsr["corner_list"];
        for (auto it = cn.begin(); it != cn.end(); ++it) {
            std::vector<cv::Point2f> corners;
            (*it) >> corners;
            if (!corners.empty()) corner_list.push_back(corners);
        }

        // read point_list (Vec3f)
        cv::FileNode pn = fsr["point_list"];
        for (auto it = pn.begin(); it != pn.end(); ++it) {
            std::vector<cv::Vec3f> v3;
            (*it) >> v3;
            std::vector<cv::Point3f> pts;
            for (auto &vv : v3) pts.emplace_back(vv[0], vv[1], vv[2]);
            if (!pts.empty()) point_list.push_back(pts);
        }
        fsr.release();
        std::cout << "Loaded " << corner_list.size() << " saved frames from " << saved_data_file << "\n";
    }

    if (corner_list.size() < 5) {
        std::cout << "Detecting corners from images in '" << frames_dir << "'...\n";
        if (!fs::exists(frames_dir)) {
            std::cerr << "Folder '" << frames_dir << "' not found. Put your images there or save calibration_data.yml.\n";
            return -1;
        }

        // prepare object points (single point set)
        std::vector<cv::Point3f> single_objp;
        for (int r = 0; r < CHECKERBOARD.height; ++r) {
            for (int c = 0; c < CHECKERBOARD.width; ++c) {
                // using (x, -y, 0) like in Task2 (y negative goes downward)
                single_objp.emplace_back(static_cast<float>(c), static_cast<float>(-r), 0.0f);
            }
        }

        // iterate over image files (.jpg, .png)
        std::vector<fs::path> files;
        for (auto &p : fs::directory_iterator(frames_dir)) {
            if (!p.is_regular_file()) continue;
            std::string ext = p.path().extension().string();
            for (auto &ch : ext) ch = (char)std::tolower(ch);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff")
                files.push_back(p.path());
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            std::cerr << "No image files found in " << frames_dir << "\n";
            return -1;
        }

        cv::Mat example_img;
        for (auto &fp : files) {
            cv::Mat img = cv::imread(fp.string());
            if (img.empty()) continue;
            if (example_img.empty()) example_img = img.clone();

            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(gray, CHECKERBOARD, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

            if (found) {
                cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
                corner_list.push_back(corners);
                point_list.push_back(single_objp);

                // show progress
                cv::Mat disp = img.clone();
                cv::drawChessboardCorners(disp, CHECKERBOARD, corners, found);
                cv::imshow("Detected (press any key to continue)", disp);
                cv::waitKey(200); 
                std::cout << "Found corners in " << fp.filename() << " (saved)\n";
            } else {
                std::cout << "Checkerboard not found in " << fp.filename() << "\n";
            }
        }

        cv::destroyAllWindows();
        std::cout << "Total detected frames: " << corner_list.size() << "\n";
    }

    // --- Validate count ---
    if (corner_list.size() < 5) {
        std::cerr << "Need at least 5 calibration images with detected corners. Found: " << corner_list.size() << "\n";
        return -1;
    }

    cv::Mat sample;
    if (fs::exists("calib_frame_1.jpg")) sample = cv::imread("calib_frame_1.jpg");
    if (sample.empty()) {
        // fallback to using an image from frames_dir if present
        for (auto &p : fs::directory_iterator("calibration_frames")) {
            if (!p.is_regular_file()) continue;
            sample = cv::imread(p.path().string());
            if (!sample.empty()) break;
        }
    }
    if (sample.empty()) {
        std::cerr << "Cannot find an example image to determine image size.\n";
        return -1;
    }
    int img_w = sample.cols, img_h = sample.rows;

    // --- Initialize camera matrix (CV_64F)---
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0,0) = 1.0;
    cameraMatrix.at<double>(1,1) = 1.0;
    cameraMatrix.at<double>(0,2) = img_w / 2.0; // u0
    cameraMatrix.at<double>(1,2) = img_h / 2.0; // v0

    // initial distortion coefficients (8-parameter full model)
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    std::cout << "\nInitial camera matrix:\n" << cameraMatrix << "\n";
    std::cout << "\nInitial distortion coefficients:\n" << distCoeffs.t() << "\n";

    // --- Convert data types to what calibrateCamera expects ---
    std::vector<std::vector<cv::Point3f>> objectPoints = point_list;
    std::vector<std::vector<cv::Point2f>> imagePoints = corner_list;

    // --- Run calibration ---
    std::vector<cv::Mat> rvecs, tvecs;
    int flags = cv::CALIB_FIX_ASPECT_RATIO; 
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-9);

    double rms = cv::calibrateCamera(objectPoints, imagePoints, cv::Size(img_w, img_h),
                                     cameraMatrix, distCoeffs, rvecs, tvecs, flags, criteria);

    std::cout << "\nCalibration finished. RMS re-projection error reported by calibrateCamera: " << rms << "\n";

    std::cout << "\nCalibrated camera matrix:\n" << cameraMatrix << "\n";
    std::cout << "\nCalibrated distortion coefficients:\n" << distCoeffs.t() << "\n";

    // --- Compute per-image and overall reprojection error (per-pixel) ---
    double total_error = 0;
    size_t total_points = 0;
    std::vector<double> per_image_errors;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projected);

        double err_sq = 0.0;
        for (size_t j = 0; j < projected.size(); ++j) {
            double dx = imagePoints[i][j].x - projected[j].x;
            double dy = imagePoints[i][j].y - projected[j].y;
            err_sq += dx*dx + dy*dy;
        }
        double rmse = std::sqrt(err_sq / projected.size());
        per_image_errors.push_back(rmse);
        total_error += err_sq;
        total_points += projected.size();
    }
    double mean_rmse = std::sqrt(total_error / total_points);

    std::cout << "\nPer-image reprojection RMS errors (pixels):\n";
    for (size_t i = 0; i < per_image_errors.size(); ++i) {
        std::cout << "  image " << i+1 << ": " << per_image_errors[i] << " px\n";
    }
    std::cout << "\nOverall mean reprojection RMSE: " << mean_rmse << " pixels\n";

    // --- Save camera matrix, distCoeffs, rvecs, tvecs to YAML ---
    cv::FileStorage fsw(output_file, cv::FileStorage::WRITE);
    if (!fsw.isOpened()) {
        std::cerr << "Failed to open " << output_file << " for writing\n";
        return -1;
    }
    fsw << "image_width" << img_w;
    fsw << "image_height" << img_h;
    fsw << "camera_matrix" << cameraMatrix;
    fsw << "distortion_coefficients" << distCoeffs;
    // save rvecs/tvecs
    fsw << "rvecs" << "[";
    for (auto &r: rvecs) fsw << r;
    fsw << "]";
    fsw << "tvecs" << "[";
    for (auto &t: tvecs) fsw << t;
    fsw << "]";
    fsw << "per_image_rmse" << "[";
    for (auto &e: per_image_errors) fsw << e;
    fsw << "]";
    fsw << "overall_rmse" << mean_rmse;
    fsw.release();

    std::cout << "\nSaved calibration to: " << output_file << "\n";
    std::cout << "Done.\n";
    return 0;
}
