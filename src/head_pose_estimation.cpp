#include <opencv2/core/types_c.h>  // cvIplImage
#include <opencv2/imgproc/imgproc_c.h>

#include <cmath>
#include <ctime>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef HEAD_POSE_ESTIMATION_DEBUG
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#endif

#include "head_pose_estimation.hpp"
#include "face_reconstruction.hpp"
#include "find_eye_center.hpp"

const float EYE_ROI_ENLARGE_FACTOR = 25; // (percentage of the detected eye width)

using namespace dlib;
using namespace std;
using namespace cv;

inline Point3f toPoint3f(const Vec4f coords)
{
    return Point3f(coords[0], coords[1], coords[2]);
}

inline Point3d toPoint3d(const Vec4d coords)
{
    return Point3d(coords[0], coords[1], coords[2]);
}

inline Vec3d toVec3d(const Vec4d coords)
{
    return Vec3d(coords[0], coords[1], coords[2]);
}


template<typename T> inline Point3d toPoint3d(const T coords)
{
    return Point3d(coords(0,0), coords(1,0), coords(2,0));
}

template<typename T> inline Vec3d toVec3d(const T coords)
{
    return Vec3d(coords(0,0), coords(1,0), coords(2,0));
}

inline Point toCv(const dlib::point& p)
{
    return Point(p.x(), p.y());
}

// Integer version
inline Point toCvI(const dlib::point& p)
{
    return Point(p.x(), p.y());
}


HeadPoseEstimation::HeadPoseEstimation(const string& face_detection_model, float focalLength) :
        focalLength(focalLength),
        opticalCenterX(-1),
        opticalCenterY(-1)
{
    // Load face detection and pose estimation models.
    detector = get_frontal_face_detector();
    deserialize(face_detection_model) >> pose_model;
}


tuple<std::array<Point, 6>, Rect, std::array<Point, 6>, Rect> HeadPoseEstimation::eyesROI(const full_object_detection& face) const
{
    // Left eye
    Rect leye_roi(Point2f(toCv(face.part(36)).x, min(toCv(face.part(37)).y,
                                                     toCv(face.part(38)).y)),
                  Point2f(toCv(face.part(39)).x, max(toCv(face.part(40)).y,
                                                     toCv(face.part(41)).y)));

    auto lmargin = EYE_ROI_ENLARGE_FACTOR/100 * leye_roi.width;

    leye_roi.x -= lmargin; leye_roi.y -= lmargin;
    leye_roi.width += 2 * lmargin; leye_roi.height += 2 * lmargin;

    std::array<Point, 6> leye = {toCvI(face.part(36)) - leye_roi.tl(),
                                 toCvI(face.part(37)) - leye_roi.tl(),
                                 toCvI(face.part(38)) - leye_roi.tl(),
                                 toCvI(face.part(39)) - leye_roi.tl(),
                                 toCvI(face.part(40)) - leye_roi.tl(),
                                 toCvI(face.part(41)) - leye_roi.tl()
                                };

    // Right eye
    Rect reye_roi(Point2f(toCv(face.part(42)).x, min(toCv(face.part(43)).y,
                                                     toCv(face.part(44)).y)),
                  Point2f(toCv(face.part(45)).x, max(toCv(face.part(46)).y,
                                                     toCv(face.part(47)).y)));

    auto rmargin = EYE_ROI_ENLARGE_FACTOR/100 * leye_roi.width;

    reye_roi.x -= rmargin; reye_roi.y -= rmargin;
    reye_roi.width += 2 * rmargin; reye_roi.height += 2 * rmargin;

    std::array<Point, 6> reye = {toCvI(face.part(42)) - reye_roi.tl(),
                                 toCvI(face.part(43)) - reye_roi.tl(),
                                 toCvI(face.part(44)) - reye_roi.tl(),
                                 toCvI(face.part(45)) - reye_roi.tl(),
                                 toCvI(face.part(46)) - reye_roi.tl(),
                                 toCvI(face.part(47)) - reye_roi.tl()
                                };

    return make_tuple(leye, leye_roi, reye, reye_roi);
}

pair<Point2f, Point2f>
HeadPoseEstimation::pupilsRelativePose(cv::InputArray _image,
                                       const full_object_detection& face) const
{
    Mat image = _image.getMat();

    std::array<Point, 6> left_eye, right_eye;
    Rect left_eye_roi, right_eye_roi;

    tie(left_eye, left_eye_roi, right_eye, right_eye_roi) = eyesROI(face);

    // Create masks for the left and right eyes
    Mat left_mask = Mat::zeros(left_eye_roi.size(), CV_8UC1);
    fillConvexPoly(left_mask, left_eye.data(), left_eye.size(), Scalar(255,255,255));

    Mat right_mask = Mat::zeros(right_eye_roi.size(), CV_8UC1);
    fillConvexPoly(right_mask, right_eye.data(), right_eye.size(), Scalar(255,255,255));

    auto left_pupil = findEyeCenter(image, left_eye_roi, left_mask);
    auto right_pupil = findEyeCenter(image, right_eye_roi, right_mask);

#ifdef HEAD_POSE_ESTIMATION_DEBUG
    circle(_debug, Point(left_pupil.x, left_pupil.y) + left_eye_roi.tl(), 4, Scalar(0,0,255), 2);
    circle(_debug, Point(right_pupil.x, right_pupil.y) + right_eye_roi.tl(), 4, Scalar(0,0,255), 2);
#endif
    Point2f left_center = left_eye_roi.br() - left_eye_roi.tl();
    Point2f left_pupil_relative = left_pupil - left_center * 0.5;
    left_pupil_relative.x /= left_eye_roi.width/2;
    left_pupil_relative.y /= left_eye_roi.height/2;

    Point2f right_center = right_eye_roi.br() - right_eye_roi.tl();
    Point2f right_pupil_relative = right_pupil - right_center * 0.5;
    right_pupil_relative.x /= right_eye_roi.width/2;
    right_pupil_relative.y /= right_eye_roi.height/2;

    return make_pair(left_pupil_relative, right_pupil_relative);
}

std::vector<std::vector<Point>> HeadPoseEstimation::update(cv::InputArray _image)
{

    Mat image = _image.getMat();

    if (opticalCenterX == -1) // not initialized yet
    {
        opticalCenterX = image.cols / 2;
        opticalCenterY = image.rows / 2;
#ifdef HEAD_POSE_ESTIMATION_DEBUG
        cerr << "Setting the optical center to (" << opticalCenterX << ", " << opticalCenterY << ")" << endl;
#endif
    }

    // intermediate value to avoid potential compilation error:
    //     conversion from ‘const cv::Mat’ to non-scalar type ‘IplImage’
    auto ipl_img = cvIplImage(image);
    current_image = cv_image<bgr_pixel>(&ipl_img);

    faces = detector(current_image);

    // Find the pose of each face.
    shapes.clear();
    for (auto face : faces){
        shapes.push_back(pose_model(current_image, face));
    }

    std::vector<std::vector<Point>> all_features;

    for (size_t j = 0; j < shapes.size(); ++j)
    {
        std::vector<Point> features;
        const full_object_detection& d = shapes[j];

        for (size_t i = 0; i < 68; ++i)
        {
            features.push_back(toCv(d.part(i)));
        }

        all_features.push_back(features);
    }

#ifdef HEAD_POSE_ESTIMATION_DEBUG
    // Draws the contours of the face and face features onto the image
    
    _debug = image.clone();


    Mat reconstructed_face;

    auto color = Scalar(0,128,128);

    for (unsigned long idx = 0; idx < shapes.size(); ++idx)
    {

        const full_object_detection& d = shapes[idx];
//

//        for (unsigned long i = 1; i <= 16; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//
//        for (unsigned long i = 28; i <= 30; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//
//        for (unsigned long i = 18; i <= 21; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        for (unsigned long i = 23; i <= 26; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        for (unsigned long i = 31; i <= 35; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(30)), toCv(d.part(35)), color, 2, CV_AA);
//
        for (unsigned long i = 37; i <= 41; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 1, CV_AA);
        line(_debug, toCv(d.part(36)), toCv(d.part(41)), color, 1, CV_AA);

        for (unsigned long i = 43; i <= 47; ++i)
            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 1, CV_AA);
        line(_debug, toCv(d.part(42)), toCv(d.part(47)), color, 1, CV_AA);
//
//        for (unsigned long i = 49; i <= 59; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(48)), toCv(d.part(59)), color, 2, CV_AA);
//
//        for (unsigned long i = 61; i <= 67; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(60)), toCv(d.part(67)), color, 2, CV_AA);

//
//        for (auto i = 0; i < 68 ; i++) {
//            putText(_debug, to_string(i), toCv(d.part(i)), FONT_HERSHEY_DUPLEX, 0.3, Scalar(255,255,255));
//        }
        
        FaceReconstruction::reconstruct(image, d, reconstructed_face);
    }

#endif

    for (const auto& shape :shapes) {
        Point2f left_pupil, right_pupil;
        tie(left_pupil, right_pupil) = pupilsRelativePose(image, shape);
    }

    return all_features;
}

head_pose HeadPoseEstimation::pose(size_t face_idx) const
{

    cv::Mat projectionMat = cv::Mat::zeros(3,3,CV_32F);
    cv::Matx33f projection = projectionMat;
    projection(0,0) = focalLength;
    projection(1,1) = focalLength;
    projection(0,2) = opticalCenterX;
    projection(1,2) = opticalCenterY;
    projection(2,2) = 1;

    std::vector<Point3f> head_points;

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
    head_points.push_back(P3D_STOMMION);

    std::vector<Point2f> detected_points;

    detected_points.push_back(coordsOf(face_idx, SELLION));
    detected_points.push_back(coordsOf(face_idx, RIGHT_EYE));
    detected_points.push_back(coordsOf(face_idx, LEFT_EYE));
    detected_points.push_back(coordsOf(face_idx, RIGHT_SIDE));
    detected_points.push_back(coordsOf(face_idx, LEFT_SIDE));
    detected_points.push_back(coordsOf(face_idx, MENTON));
    detected_points.push_back(coordsOf(face_idx, NOSE));

    auto stomion = (coordsOf(face_idx, MOUTH_CENTER_TOP) + coordsOf(face_idx, MOUTH_CENTER_BOTTOM)) * 0.5;
    detected_points.push_back(stomion);


    // Initializing the head pose 1m away, roughly facing the robot
    // This initialization is important as it prevents solvePnP to find the
    // mirror solution (head *behind* the camera)
    Mat tvec = (Mat_<double>(3,1) << 0., 0., 1000.);
    Mat rvec = (Mat_<double>(3,1) << 1.2, 1.2, -1.2);

    // Find the 3D pose of our head
    solvePnP(head_points, detected_points,
            projection, noArray(),
            rvec, tvec, true,
#ifdef OPENCV3
            cv::SOLVEPNP_ITERATIVE);
#else
            cv::ITERATIVE);
#endif

    Matx33d rotation;
    Rodrigues(rvec, rotation);

    head_pose pose = {
        rotation(0,0),    rotation(0,1),    rotation(0,2),    tvec.at<double>(0)/1000,
        rotation(1,0),    rotation(1,1),    rotation(1,2),    tvec.at<double>(1)/1000,
        rotation(2,0),    rotation(2,1),    rotation(2,2),    tvec.at<double>(2)/1000,
                    0,                0,                0,                     1
    };

    return pose;
}

std::vector<head_pose> HeadPoseEstimation::poses() const {

    std::vector<head_pose> res;

    for (auto i = 0; i < faces.size(); i++){
        res.push_back(pose(i));
    }

    return res;

}

Mat HeadPoseEstimation::drawDetections(const cv::Mat& original_image, const std::vector<std::vector<Point>>& detected_features, const std::vector<head_pose>& detected_poses) {
    auto result = original_image.clone();
    if (!detected_features.empty()) {
        drawFeatures(detected_features, result);
    }
    for (size_t i = 0; i < detected_poses.size(); ++i)
    {
        drawPose(detected_poses[i], i, result);
    }
    return result;
}

void HeadPoseEstimation::drawFeatures(const std::vector<std::vector<Point>>& detected_features, Mat& result) const {

    static const auto line_color = Scalar(0,128,128);
    static const auto text_color = Scalar(255,255,255);

    for (size_t j = 0; j < detected_features.size(); ++j)
    {
        const auto& feature_points = detected_features[j];

        for (size_t i = 1; i <= 16; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);

        for (size_t i = 28; i <= 30; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);

        for (size_t i = 18; i <= 21; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        for (size_t i = 23; i <= 26; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        for (size_t i = 31; i <= 35; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        cv::line(result, feature_points[30], feature_points[35], line_color, 2, CV_AA);

        for (size_t i = 37; i <= 41; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        cv::line(result, feature_points[36], feature_points[41], line_color, 2, CV_AA);

        for (size_t i = 43; i <= 47; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        cv::line(result, feature_points[42], feature_points[47], line_color, 2, CV_AA);

        for (size_t i = 49; i <= 59; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        cv::line(result, feature_points[48], feature_points[59], line_color, 2, CV_AA);

        for (size_t i = 61; i <= 67; ++i)
            cv::line(result, feature_points[i], feature_points[i-1], line_color, 2, CV_AA);
        cv::line(result, feature_points[60], feature_points[67], line_color, 2, CV_AA);

        // for (size_t i = 0; i < 68 ; i++) {
            // putText(result, to_string(i), feature_points[i], FONT_HERSHEY_DUPLEX, 0.6, text_color);
        // }
    }
}

void HeadPoseEstimation::drawPose(const head_pose& detected_pose, size_t face_idx, cv::Mat& result) const {
    const auto rotation = Mat(detected_pose)(Range(0, 3), Range(0, 3));
    auto rvec = Mat_<double>(3, 1);
    Rodrigues(rotation, rvec);

    auto tvec = Mat(detected_pose).col(3).rowRange(0, 3);

    
    cv::Matx33f projection(focalLength, 0.0,         opticalCenterX,
                           0.0,         focalLength, opticalCenterY,
                           0.0,         0.0,         1.0);

    
    std::vector<Point3f> head_points;

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
    head_points.push_back(P3D_STOMMION);

    // std::vector<Point2f> reprojected_points;
    // projectPoints(head_points, rvec, tvec, projection, noArray(), reprojected_points);
 
    // static const auto circle_color = Scalar(0, 255, 255);
    // for (auto point : reprojected_points) {
        // circle(result, point,2, circle_color,2);
    // }

    std::vector<Point3f> axes;
    axes.push_back(Point3f(0,0,0));
    axes.push_back(Point3f(50,0,0));
    axes.push_back(Point3f(0,50,0));
    axes.push_back(Point3f(0,0,50));

    std::vector<Point2f> projected_axes;
    projectPoints(axes, rvec, tvec, projection, noArray(), projected_axes);

    static const auto x_axis_color = Scalar(255, 0, 0);
    static const auto y_axis_color = Scalar(0, 255, 0);
    static const auto z_axis_color = Scalar(0, 0, 255);
    cv::line(result, projected_axes[0], projected_axes[3], x_axis_color,2,CV_AA);
    cv::line(result, projected_axes[0], projected_axes[2], y_axis_color,2,CV_AA);
    cv::line(result, projected_axes[0], projected_axes[1], z_axis_color,2,CV_AA);

    static const auto text_color = Scalar(0,0,255);
    putText(result, "(" + to_string(int(detected_pose(0,3) * 100)) + "cm, " + to_string(int(detected_pose(1,3) * 100)) + "cm, " + to_string(int(detected_pose(2,3) * 100)) + "cm)", coordsOf(face_idx, SELLION), FONT_HERSHEY_SIMPLEX, 0.5, text_color,2);
}

Point2f HeadPoseEstimation::coordsOf(size_t face_idx, FACIAL_FEATURE feature) const
{
    return toCv(shapes[face_idx].part(feature));
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// taken from: http://stackoverflow.com/a/7448287/828379
bool HeadPoseEstimation::intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                                      Point2f &r) const
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

