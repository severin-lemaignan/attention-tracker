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

using namespace dlib;
using namespace std;
using namespace cv;


const std::vector<std::array<int,3>> FACE_TRIANGLES = {
    // chicks
    {35, 13, 45}, {31, 3, 36},
    {35, 54, 13}, {31, 48, 3},
    {35, 42, 47}, {31, 39, 40},
    {35, 45, 47}, {31, 36, 40},
    {13, 16, 45}, { 3, 0, 36},
    {26, 16, 45}, {17, 0, 36},

    // eyebrows
    {26, 24, 45}, {17, 19, 36},
    {43, 24, 45}, {36, 19, 38},

    // eyes
    {45, 43, 47}, {36, 38, 40},
    {42, 43, 47}, {39, 38, 40},

    {42, 43, 22}, {21, 38, 39},
    {24, 43, 22}, {19, 38, 21},

    {42, 27, 22}, {21, 27, 39},

    {21, 27, 22},
    // nose
    {35, 27, 30}, {31, 27, 30},
    {31, 30, 35},
    {31, 33, 35},
    {42, 27, 35}, {31, 27, 39},
    {54, 33, 35}, {48, 33, 31},
    // mouth
    {54, 33, 51}, {48, 33, 51},
    {54, 62, 51}, {48, 62, 51},
    {54, 62, 66}, {48, 62, 66},
    {54, 57, 66}, {48, 57, 66},
    // chin zone
    {54, 13, 10}, {48, 3, 6},
    {54, 57, 10}, {48, 57, 6},
    {8, 57, 10}, {8, 57, 6}
};

// 2D position of selected facial landmarks when frontally viewed
const cv::Point2f FRONT_FACE_LANDMARKS[68] = {
    /* landmark  0*/ {0,54},
    /* landmark  1*/ {0,0},
    /* landmark  2*/ {0,0},
    /* landmark  3*/ {10,193},
    /* landmark  4*/ {0,0},
    /* landmark  5*/ {0,0},
    /* landmark  6*/ {74,312},
    /* landmark  7*/ {0,0},
    /* landmark  8*/ {150,344},
    /* landmark  9*/ {0,0},
    /* landmark 10*/ {220,312},
    /* landmark 11*/ {0,0},
    /* landmark 12*/ {0,0},
    /* landmark 13*/ {290,193},
    /* landmark 14*/ {0,0},
    /* landmark 15*/ {0,0},
    /* landmark 16*/ {299,54},
    /* landmark 17*/ {12,32},
    /* landmark 18*/ {0,0},
    /* landmark 19*/ {64,0},
    /* landmark 20*/ {0,0},
    /* landmark 21*/ {121,20},
    /* landmark 22*/ {176,20},
    /* landmark 23*/ {0,0},
    /* landmark 24*/ {236,0},
    /* landmark 25*/ {0,0},
    /* landmark 26*/ {280,32},
    /* landmark 27*/ {150,45},
    /* landmark 28*/ {0,0},
    /* landmark 29*/ {0,0},
    /* landmark 30*/ {150,150},
    /* landmark 31*/ {114,164},
    /* landmark 32*/ {0,0},
    /* landmark 33*/ {150,179},
    /* landmark 34*/ {0,0},
    /* landmark 35*/ {180,164},
    /* landmark 36*/ {45,45},
    /* landmark 37*/ {0,0},
    /* landmark 38*/ {86,36},
    /* landmark 39*/ {104,50},
    /* landmark 40*/ {83,55},
    /* landmark 41*/ {0,0},
    /* landmark 42*/ {192,50},
    /* landmark 43*/ {211,35},
    /* landmark 44*/ {0,0},
    /* landmark 45*/ {250,46},
    /* landmark 46*/ {0,0},
    /* landmark 47*/ {213,55},
    /* landmark 48*/ {89,226},
    /* landmark 49*/ {0,0},
    /* landmark 50*/ {0,0},
    /* landmark 51*/ {150,217},
    /* landmark 52*/ {0,0},
    /* landmark 53*/ {0,0},
    /* landmark 54*/ {208,226},
    /* landmark 55*/ {0,0},
    /* landmark 56*/ {0,0},
    /* landmark 57*/ {150,250},
    /* landmark 58*/ {0,0},
    /* landmark 59*/ {0,0},
    /* landmark 60*/ {0,0},
    /* landmark 61*/ {0,0},
    /* landmark 62*/ {150,227},
    /* landmark 63*/ {0,0},
    /* landmark 64*/ {0,0},
    /* landmark 65*/ {0,0},
    /* landmark 66*/ {150,228},
    /* landmark 67*/ {0,0}
};

inline Point toCv(const dlib::point& p)
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
    
    //_debug = image.clone();
    _debug.create(image.size(), image.type());
    _debug.setTo(Scalar(0,0,0));

    auto reconstructed_size = Size(300,345);

    Mat _reconstructed_face;
    _reconstructed_face.create(reconstructed_size, image.type());
    Mat transformed_part;
    transformed_part.create(reconstructed_size, image.type());

    Mat mask;
    mask.create(image.size(), image.type());

    auto color = Scalar(0,128,128);

    for (unsigned long i = 0; i < shapes.size(); ++i)
    {
        _reconstructed_face.setTo(Scalar(0,0,0));
        transformed_part.setTo(Scalar(0,0,0));

        const full_object_detection& d = shapes[i];

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
//        for (unsigned long i = 37; i <= 41; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(36)), toCv(d.part(41)), color, 2, CV_AA);
//
//        for (unsigned long i = 43; i <= 47; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(42)), toCv(d.part(47)), color, 2, CV_AA);
//
//        for (unsigned long i = 49; i <= 59; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(48)), toCv(d.part(59)), color, 2, CV_AA);
//
//        for (unsigned long i = 61; i <= 67; ++i)
//            line(_debug, toCv(d.part(i)), toCv(d.part(i-1)), color, 2, CV_AA);
//        line(_debug, toCv(d.part(60)), toCv(d.part(67)), color, 2, CV_AA);

        //for (auto i = 0; i < 68 ; i++) {
        //    putText(_debug, to_string(i), toCv(d.part(i)), FONT_HERSHEY_DUPLEX, 0.6, Scalar(255,255,255));
        //}
        
        for (auto trgl : FACE_TRIANGLES) {

            Point face[3] = {toCv(d.part(trgl[0])), toCv(d.part(trgl[1])), toCv(d.part(trgl[2]))};

            mask.setTo(Scalar(0,0,0));
            fillConvexPoly(mask, face, 3, Scalar(255,255,255));

            line(_debug, toCv(d.part(trgl[0])), toCv(d.part(trgl[1])), Scalar(255,255,255), 1, CV_AA);
            line(_debug, toCv(d.part(trgl[1])), toCv(d.part(trgl[2])), Scalar(255,255,255), 1, CV_AA);
            line(_debug, toCv(d.part(trgl[2])), toCv(d.part(trgl[0])), Scalar(255,255,255), 1, CV_AA);


            Mat face_part;
            image.copyTo(face_part, mask);

            Point2f facef[3] = {toCv(d.part(trgl[0])), toCv(d.part(trgl[1])), toCv(d.part(trgl[2]))};
            Point2f flat_facef[3] = {FRONT_FACE_LANDMARKS[trgl[0]],
                                     FRONT_FACE_LANDMARKS[trgl[1]],
                                     FRONT_FACE_LANDMARKS[trgl[2]]};

            auto trans = getAffineTransform(facef, flat_facef);

            warpAffine(face_part,
                       transformed_part, 
                       trans, 
                       transformed_part.size(), 
                       INTER_LINEAR,
                       BORDER_REPLICATE);
            cv::max(transformed_part, _reconstructed_face, _reconstructed_face);

        }

        imshow("reconstructed face " + to_string(i), _reconstructed_face);

    }
#endif

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

