/*
Copyright (C) 2013-2015 Tristan Hume

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

#include "helpers.hpp"

using namespace cv;

bool rectInImage(Rect rect, Mat image) {

    return rect.x > 0 && 
           rect.y > 0 && 
           rect.x+rect.width < image.cols &&
           rect.y+rect.height < image.rows;
}

bool inMat(Point p,int rows,int cols) {
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

Mat matrixMagnitude(const Mat &matX, const Mat &matY) {
    Mat mags(matX.rows,matX.cols,CV_64F);
    for (int y = 0; y < matX.rows; ++y) {
        const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
        double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < matX.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor) {
    Scalar stdMagnGrad, meanMagnGrad;
    meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}
