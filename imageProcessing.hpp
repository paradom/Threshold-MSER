/***********************************************
 *
 *  imageProcessing.hpp
 *
 *  Copyright © 2022 Oregon State University
 *
 *  Dominic W. Daprano
 *  Sheng Tse Tsai 
 *  Moritz S. Schmid
 *  Christopher M. Sullivan
 *  Robert K. Cowen
 *
 *  Hatfield Marine Science Center
 *  Center for Qualitative Life Sciences
 *  Oregon State University
 *  Corvallis, OR 97331
 *
 *  This program is distributed WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * 
 *  This program is distributed under the GNU GPL v 2.0 or later license.
 *
 *  Any User wishing to make commercial use of the Software must contact the authors 
 *  or Oregon State University directly to arrange an appropriate license.
 *  Commercial use includes (1) use of the software for commercial purposes, including 
 *  integrating or incorporating all or part of the source code into a product 
 *  for sale or license by, or on behalf of, User to third parties, or (2) distribution 
 *  of the binary or source code to third parties for use with a commercial 
 *  product sold or licensed by, or on behalf of, User.
 *
***********************************************/ 

#ifndef IMAGE_PROCESSING
#define IMAGE_PROCESSING

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp> // used for the video preprocessing

struct Options
{
    std::string input;
    std::string outputDirectory;
    int numConcatenate;
    int signalToNoise;
    int minimum;
    int maximum;
    float epsilon;
    int delta;
    int variation;
    float outlierPercent;
};

bool containExt(const std::string s, std::string arr[], int len);

bool isInt(std::string str);

std::string convertInt(int number, int fill=4);

cv::Rect rescaleRect(const cv::Rect rect, float scale=.5);

void preprocess(const cv::Mat& src, cv::Mat& dst, float erosion_size);

float SNR(cv::Mat& img);

void flatField(cv::Mat& src, cv::Mat& dst, float outlierPercent);

void trimMean(const cv::Mat& img, cv::Mat& tMean, float percent,
    int nChannels);

void groupRect(std::vector<cv::Rect>& rectList, int groupThreshold, 
    double eps);

void mser(cv::Mat img, std::vector<cv::Rect>& bboxes, int delta=5, 
        int max_variation=5, float eps=.3);

void getFrame(cv::VideoCapture cap, cv::Mat& img, int& frameCounter, 
        int numConcatenate);

void segmentImage(cv::Mat img, cv::Mat& imgCorrect, std::vector<cv::Rect>& bboxes,
        Options options);

void saveCrops(cv::Mat img, cv::Mat imgCorrect, std::vector<cv::Rect> bboxes,
        std::string imgDir, std::string imgName, std::ofstream& measurePtr,
        Options options);

#endif
