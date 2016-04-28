
#ifdef HEAD_POSE_ESTIMATION_DEBUG
#include <opencv2/highgui/highgui.hpp>
#endif

#include "find_eye_center.hpp"

#include "pupils.hpp"

using namespace std;
using namespace cv;


const float EYE_ROI_ENLARGE_FACTOR = 10; // (percentage of the detected eye width)


Pupils::Pupils() {

    // In the reprojected face, the location of the eyes is constant.
    // Therefore, we can initialize at construction time the masks of the eyes

    tie(left_eye, left_eye_roi, right_eye, right_eye_roi) = eyesROI();

    // Create masks for the left and right eyes
    left_mask = Mat::zeros(left_eye_roi.size(), CV_8UC1);
    fillConvexPoly(left_mask, left_eye.data(), left_eye.size(), Scalar(255,255,255));

    right_mask = Mat::zeros(right_eye_roi.size(), CV_8UC1);
    fillConvexPoly(right_mask, right_eye.data(), right_eye.size(), Scalar(255,255,255));

}

tuple<std::array<Point, 6>, Rect, std::array<Point, 6>, Rect> Pupils::eyesROI() const
{
    // Left eye
    Rect leye_roi(Point(194, 31),
                  Point(252,64));

    auto lmargin = EYE_ROI_ENLARGE_FACTOR/100 * leye_roi.width;

    leye_roi.x -= lmargin; leye_roi.y -= lmargin;
    leye_roi.width += 2 * lmargin; leye_roi.height += 2 * lmargin;

    std::array<Point, 6> leye = {Point(199,48) - leye_roi.tl(),
                                 Point(214,38) - leye_roi.tl(),
                                 Point(230,35) - leye_roi.tl(),
                                 Point(246,42) - leye_roi.tl(),
                                 Point(235,59) - leye_roi.tl(),
                                 Point(214,59) - leye_roi.tl()
                                };

    // Right eye
    Rect reye_roi(Point(45,32),
                  Point(105,63));

    auto rmargin = EYE_ROI_ENLARGE_FACTOR/100 * leye_roi.width;

    reye_roi.x -= rmargin; reye_roi.y -= rmargin;
    reye_roi.width += 2 * rmargin; reye_roi.height += 2 * rmargin;

    std::array<Point, 6> reye = {Point(51,45) - reye_roi.tl(),
                                 Point(68,38) - reye_roi.tl(),
                                 Point(87,39) - reye_roi.tl(),
                                 Point(99,48) - reye_roi.tl(),
                                 Point(81,59) - reye_roi.tl(),
                                 Point(59,54) - reye_roi.tl()
                                };

    return make_tuple(leye, leye_roi, reye, reye_roi);
}

pair<Point2f, Point2f>
Pupils::findRelativePose(cv::InputArray _image) const
{

    Mat image = _image.getMat();

    auto left_pupil = findEyeCenter(image, left_eye_roi, left_mask);
    auto right_pupil = findEyeCenter(image, right_eye_roi, right_mask);

#ifdef HEAD_POSE_ESTIMATION_DEBUG
    //circle(_debug, Point(left_pupil.x, left_pupil.y) + left_eye_roi.tl(), 1, Scalar(0,0,255), 1);
    //circle(_debug, Point(right_pupil.x, right_pupil.y) + right_eye_roi.tl(), 4, Scalar(0,0,255), 2);

    //Mat left_eye_debug;
    //resize(_debug(left_eye_roi), left_eye_debug, Size(0,0), 10, 10);
    //imshow("left eye", left_eye_debug);
    
    circle(image, Point(left_pupil.x, left_pupil.y) + left_eye_roi.tl(), 4, Scalar(0,0,255), 2);
    circle(image, Point(right_pupil.x, right_pupil.y) + right_eye_roi.tl(), 4, Scalar(0,0,255), 2);
    imshow("pupils",image);


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

