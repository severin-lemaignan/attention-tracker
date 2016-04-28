#include <opencv2/imgproc/imgproc.hpp>

#ifdef HEAD_POSE_ESTIMATION_DEBUG
#include <opencv2/highgui/highgui.hpp>
#endif


#include "face_reconstruction.hpp"

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

inline Point2f toCv(const dlib::point& p)
{
    return Point2f(p.x(), p.y());
}

cv::Size FaceReconstruction::RECONSTRUCTED_FACE_SIZE = cv::Size(300,345);

void FaceReconstruction::reconstruct(cv::InputArray _image,
                                     const dlib::full_object_detection& face,
                                     cv::OutputArray _reconstructed_face) {

        // access the InputArray/OutputArray
        Mat image = _image.getMat();
        _reconstructed_face.create(RECONSTRUCTED_FACE_SIZE, image.type());

        Mat reconstructed_face = _reconstructed_face.getMat();
        reconstructed_face.setTo(Scalar(0,0,0));
        ///////////////

        Mat transformed_part;
        transformed_part.create(RECONSTRUCTED_FACE_SIZE, image.type());

        Mat mask;
        mask.create(image.size(), image.type());


        for (auto trgl : FACE_TRIANGLES) {

            Point orig_face[3] = {toCv(face.part(trgl[0])), 
                                  toCv(face.part(trgl[1])), 
                                  toCv(face.part(trgl[2]))};

            mask.setTo(Scalar(0,0,0));
            fillConvexPoly(mask, orig_face, 3, Scalar(255,255,255));

            //line(_debug, toCv(face.part(trgl[0])), toCv(face.part(trgl[1])), Scalar(255,255,255), 1, CV_AA);
            //line(_debug, toCv(face.part(trgl[1])), toCv(face.part(trgl[2])), Scalar(255,255,255), 1, CV_AA);
            //line(_debug, toCv(face.part(trgl[2])), toCv(face.part(trgl[0])), Scalar(255,255,255), 1, CV_AA);


            Mat face_part;
            image.copyTo(face_part, mask);

            Point2f orig_facef[3] = {toCv(face.part(trgl[0])), 
                                     toCv(face.part(trgl[1])), 
                                     toCv(face.part(trgl[2]))};

            Point2f reproj_facef[3] = {FRONT_FACE_LANDMARKS[trgl[0]],
                                       FRONT_FACE_LANDMARKS[trgl[1]],
                                       FRONT_FACE_LANDMARKS[trgl[2]]};

            auto trans = getAffineTransform(orig_facef, reproj_facef);

            warpAffine(face_part,
                       transformed_part, 
                       trans, 
                       transformed_part.size(), 
                       INTER_LINEAR,
                       BORDER_REPLICATE);

            // 'add' the reprojected part to the whole face image
            cv::max(transformed_part, reconstructed_face, reconstructed_face);

        }

#ifdef HEAD_POSE_ESTIMATION_DEBUG
        imshow("reconstructed face" , reconstructed_face);
#endif
};

