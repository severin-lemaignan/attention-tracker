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

#include <queue>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helpers.hpp"

using namespace std;
using namespace cv;

// Algorithm Parameters
const float kFastEyeWidth = 50.f;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.f;
const double kGradientThreshold = 50.f;
const int kWeightBlurSize = 5;

// Postprocessing
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;

// Forward declarations
Mat floodKillEdges(Mat &mat);

Point2f unscalePoint(Point p, Rect origSize) {
    float ratio = (kFastEyeWidth/origSize.width);
    int x = p.x / ratio;
    int y = p.y / ratio;
    return Point2f(x,y);
}

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

Point2f findEyeCenter(Mat face, Rect eye) {
    Mat eyeROIUnscaled;
    cvtColor( face(eye), eyeROIUnscaled, CV_BGR2GRAY );

    Mat eyeROI;

    resize(eyeROIUnscaled, eyeROI,
           Size(kFastEyeWidth,
                (kFastEyeWidth / eyeROIUnscaled.cols * eyeROIUnscaled.rows)));

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
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP);

    //-- Flood fill the edges
    if(kEnablePostProcess) {
        Mat floodClone;
        //double floodThresh = computeDynamicThreshold(out, 1.5);
        double floodThresh = maxVal * kPostProcessThreshold;
        threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);

        Mat mask = floodKillEdges(floodClone);

        // redo max
        minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }
    return unscalePoint(maxP,eye);
}

bool floodShouldPushPoint(const Point &np, const Mat &mat) {
    return inMat(np, mat.rows, mat.cols);
}

// returns a mask
Mat floodKillEdges(Mat &mat) {

    Mat mask(mat.rows, mat.cols, CV_8U, 255);
    queue<Point> toDo;
    toDo.push(Point(0,0));
    while (!toDo.empty()) {
        Point p = toDo.front();
        toDo.pop();
        if (mat.at<float>(p) == 0.0f) {
            continue;
        }

        // add in every direction
        Point np(p.x + 1, p.y); // right
        if (floodShouldPushPoint(np, mat)) toDo.push(np);

        np.x = p.x - 1; np.y = p.y; // left
        if (floodShouldPushPoint(np, mat)) toDo.push(np);

        np.x = p.x; np.y = p.y + 1; // down
        if (floodShouldPushPoint(np, mat)) toDo.push(np);

        np.x = p.x; np.y = p.y - 1; // up
        if (floodShouldPushPoint(np, mat)) toDo.push(np);

        // kill it
        mat.at<float>(p) = 0.0f;
        mask.at<uchar>(p) = 0;
    }
    return mask;
}
