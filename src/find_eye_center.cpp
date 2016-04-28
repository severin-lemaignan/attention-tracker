/*
Copyright (C) 2013-2015 Tristan Hume

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// This is an implementation of an algorithm proposed by
// Fabian Timm and Erhardt Barth in their paper
// "Accurate Eye Centre Localisation by Means of
// Gradients"

#define FIND_EYE_DEBUG false

#include <queue>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "helpers.hpp"

using namespace std;
using namespace cv;

// Algorithm Parameters
const float kFastEyeWidth = 50.f;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.f;
const double kGradientThreshold = 50.f;
const int kWeightBlurSize = 5;


Mat computeMatXGradient(const Mat &mat) {
    Mat out(mat.rows,mat.cols,CV_64F);

    for (int y = 0; y < mat.rows; ++y) {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);

        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }
        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }

    return out;
}

void testPossibleCentersFormula(int x, int y, const Mat &weight,double gx, double gy, Mat &out) {
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy) {
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
        for (int cx = 0; cx < out.cols; ++cx) {
            if (x == cx && y == cy) {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            double dx = x - cx;
            double dy = y - cy;
            // normalize d
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gx + dy*gy;
            dotProduct = max(0.0,dotProduct);
            // square and multiply by the weight
            if (kEnableWeight) {
                Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
            } else {
                Or[cx] += dotProduct * dotProduct;
            }
        }
    }
}

void show(string name, Mat img) {

    Mat tmp;
    resize(img, tmp, Size(0,0), 10, 10);
    imshow(name, tmp);
}

Point2f findEyeCenter(InputArray _face, Rect eye_roi, InputArray _eye_mask) {

    Mat face = _face.getMat();
    Mat eye_mask_unscaled = _eye_mask.getMat();

    Mat eyeROIUnscaled;
    cvtColor( face(eye_roi), eyeROIUnscaled, CV_BGR2GRAY );

    Mat eyeROI;
    Mat eye_mask;

    if (eyeROIUnscaled.cols > kFastEyeWidth) {
        resize(eyeROIUnscaled, eyeROI,
               Size(kFastEyeWidth,
                    (kFastEyeWidth / eyeROIUnscaled.cols * eyeROIUnscaled.rows)));

        resize(eye_mask_unscaled, eye_mask,eyeROI.size(), 0,0, INTER_NEAREST);
    }
    else {
        eyeROI = eyeROIUnscaled;
        eye_mask = eye_mask_unscaled;
    }

    //-- Find the gradient
    Mat gradientX = computeMatXGradient(eyeROI);
    Mat gradientY = computeMatXGradient(eyeROI.t()).t();

    //-- Normalize and threshold the gradient
    // compute all the magnitudes
    Mat mags = matrixMagnitude(gradientX, gradientY);

    //compute the threshold
    double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);

    //normalize
    for (int y = 0; y < eyeROI.rows; ++y) {
        double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        const double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < eyeROI.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if (magnitude > gradientThresh) {
                Xr[x] = gX/magnitude;
                Yr[x] = gY/magnitude;
            } else {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }

    //-- Create a blurred and inverted image for weighting
    Mat weight;
    GaussianBlur( eyeROI, weight, Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x) {
            row[x] = (255 - row[x]);
        }
    }

    //-- Run the algorithm!
    Mat outSum = Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);

    // for each possible gradient location
    // Note: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.
    for (int y = 0; y < weight.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < weight.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            if (gX == 0.0 && gY == 0.0) {
                continue;
            }
            testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
        }
    }

    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);


    //-- Find the maximum point
    Point maxP;
    double maxVal;
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP, eye_mask);

#if FIND_EYE_DEBUG
    Mat debug;
    addWeighted( eyeROI, 0.8, eye_mask, 0.2, 0.0, debug);
    cout << maxP  << " ->(unscaled) ";
    circle(debug, maxP, 2, Scalar(0,0,255), 2);
    show("mask", debug);
#endif

    float ratio = (float)(eyeROIUnscaled.size().width) / eyeROI.size().width;
#if FIND_EYE_DEBUG
    cout << "(x" << ratio << ") ";
    cout << Point2f(maxP.x * ratio, maxP.y * ratio) << endl;
#endif
    return Point2f(maxP.x * ratio, maxP.y * ratio);
}

bool floodShouldPushPoint(const Point &np, const Mat &mat) {
    return inMat(np, mat.rows, mat.cols);
}

