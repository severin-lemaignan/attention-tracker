#ifndef _FACE_RECONSTRUCTION_HPP
#define _FACE_RECONSTRUCTION_HPP

#include <opencv2/core/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>

class FaceReconstruction {

    public:
        static cv::Size RECONSTRUCTED_FACE_SIZE;
        
        static void reconstruct(cv::InputArray image,
                                const dlib::full_object_detection& face,
                                cv::OutputArray reconstructed_face);

        static bool isdegenerate(const cv::Point2f pts[3],
                                 const cv::Point2f reproj_pts[3]);
};

#endif
