#ifndef _PUPILS_HPP
#define _PUPILS_HPP

#include <opencv2/core/core.hpp>

class Pupils {

public:

    Pupils();

    /** Returns a tuple (left eye contour, left eye ROI, right eye contour,
     * right eye ROI) in a fixed-sized reprojected face as created by face_reconstruction.hpp.
     *
     * Note that the contour coordinates are relative to the top-left corner
     * of the ROI.
     */
    std::tuple<std::array<cv::Point, 6>, cv::Rect,
               std::array<cv::Point, 6>, cv::Rect>
    eyesROI() const;

    /** Returns the position of the left and right pupils, relative to the left
     * and right center of the eye, normalized with respect to the width/height
     * of each eye.
     *
     *   (0.0, 0.0)    (-1.0, 0.0)      (0.0, 1.0)   (1.0, 0.0) ...
     *    _______        _______        _______       _______
     *   /       \      /       \      /   O   \     /       \
     *  <    O    >    < O       >    <         >   <       O >
     *   \_______/      \_______/      \_______/     \_______/
     *
     */
    std::pair<cv::Point2f, cv::Point2f>
    findRelativePose(cv::InputArray reprojected_face) const;

private:
    std::array<cv::Point, 6> left_eye, right_eye;
    cv::Rect left_eye_roi, right_eye_roi;

    cv::Mat left_mask, right_mask;

};
#endif

