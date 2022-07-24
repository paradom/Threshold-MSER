/** @file imageProcessing.hpp
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
 */

#ifndef IMAGE_PROCESSING
#define IMAGE_PROCESSING

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp> // used for the video preprocessing

/**
 * @class OverlapRects
 * 
 * This is a modified version of the OverlapRectangles class that has different 
 * properties. Like the original, this class can be used to pass into the 
 * partition function in order to create a grouping of the rectangles 
 * accross the image.
 *
 * If eps < 1, then rectangles need to overlap more to be grouped
 * If eps >= 1, then rectangles close to each other will be grouped
 *
 */
class OverlapRects
{
public:
    OverlapRects(double _eps) : eps(_eps) {}
    inline bool operator()(const cv::Rect& r1, const cv::Rect& r2) const
    {
        double deltax = eps * (r1.width + r2.width) * 0.5;
        double deltay = eps * (r1.height + r2.height) * 0.5;

        return std::abs((r1.x - r2.x) + (r1.width - r2.width) * 0.5) <= deltax &&
            std::abs((r1.y - r2.y) - (r1.height - r2.height) * 0.5) <= deltay;
    }
    double eps;
};

class OverlapRects2
{
public:
    OverlapRects2(double _eps) : eps(_eps) {}
    inline bool operator()(const cv::Rect& r1, const cv::Rect& r2) const
    {
        // FIXME: Add scaling with eps
        return (r1 & r2).area() > 0;
    }
    double eps;
};

/**
 * @struct Options 
 *
 * Options struct allows parameters to be passed easily to each function.
 */
struct Options
{
    std::string input;
    std::string outputDirectory;
    int numConcatenate;
    int signalToNoise;
    int minArea;
    int maxArea;
    float epsilon;
    int delta;
    int variation;
    int threshold;
    float outlierPercent;
    bool fullOutput;
    int left;
    int right;
};

bool containExt(const std::string s, std::string arr[], int len);

bool isInt(std::string str);

/**
 * Converts an integer to a string and adds padding 0's.
 *
 * @param number The number to add zeros.
 * @param fill Add zeros at the beginning of the number until it is fill long.
 * 
 * @return std::string object with concatenated 0's at the end.
 */
std::string convertInt(int number, int fill=4);

/**
 * Rescales a cv::Rect around its' centroid.
 *
 * This function rescales each side of the rectangle by the scale factor around
 * same centroid.
 *
 * @param rect The rectangle to scale
 * @param scale The scale factor to scale the rectangle. A scale of 1 would
 *              leave the rectangle unchanged.
 *
 * @return The rescaled rectangle.
 */
cv::Rect rescaleRect(const cv::Rect& rect, float scale=.5);

/**
 * Any pixel values above thresh is set to 255 (white).
 *
 * \begin{equation}
 * dst =
 * \begin{cases}
 *   255 & \text{if } x < thresh \\
 *   src & \text{if } x > thresh
 * \end{cases}
 * \end{equation}
 *
 * @param src Input image.
 * @param dst Output image.
 * @param thresh Threshold to determine which pixels should be turned white
 *
 * @return A new cv::Rect that has the rescaled sides.
 */
void chopThreshold(const cv::Mat& src, cv::Mat& dst, int thresh);

/**
 * Calculate the signal to noise ratio (SNR) for an image.
 *
 * \begin{equation}
 *  SNR = 20 * log(\frac{img_{signal}}{img_{noise}})
 * \end{equation}
 *
 * @param img The image to calculate the SNR.
 *
 * @return SNR
 */
float SNR(const cv::Mat& img);

/**
 * Performs flat fielding on the image.
 *
 * Using the outlier percentage determine the column-wise mean value. Divide 
 * the image by this columnwise mean value to reduce the overal noise of the image.
 * Since the image is created from a linescan, this helps remove noise.
 *
 * @param src The image to perform flat fielding on.
 * @param dst The destination image to save the results of flat fielding.
 * @param percent The top and bottom percent to remove in the calculation of the mean.
 * percent should be a float between 0 and 1.
 */
void flatField(const cv::Mat& src, cv::Mat& dst, float percent);

/**
 * Fills the left and right columns of the input img with pixels with value fill.
 * The original cv::Mat img is modified in place.
 * This function was added for shadows that appeared on the side of isiis images.
 *
 * @param img The image to fill the left and right sides.
 * @param left The number of columns on the left to fill.
 * @param right The number of columns on the right to fill.
 *
 * @return If left and right are valid returns 0, if not returns 1.
 */
int fillSides(cv::Mat& img, int left, int right, int fill=255);

/**
 * Finds the verticle mean of each row of the iages.
 *
 * trimMean calculates the trimmed mean of each vertical strip of the image. trimMean
 * will trim the top k brightest and darkest pixel values where k is given by
 *
 * \f[
 *   k = img.rows * \frac{percent}{2}
 * \f]
 *
 * @param img The image to calculate the trimmed mean on.
 * @param tMean A row vector containing the trimmed mean of each column of the image.
 * @param percent The top and bottom percent to remove in the calculation of the mean.
 * percent should be a float between 0 and 1.
 *
 */
void trimMean(const cv::Mat& img, cv::Mat& tMean, float percent);

void preprocess(const cv::Mat& src, cv::Mat& dst, float erosion_size);

/**
 * Different implementation of the OpenCV groupRectangles function.
 *
 * This function allows control over how much of the rectangles need to overlap
 * in order to be joined into the same ROI
 *
 * @param rectList List of bounding bboxes to group
 * @param groupThreshold The number of rectangles required in a grouping to make a grouped
 * rectangle.
 * @param eps The percent threshold to count boxes as overlapping.
 *
 */
void groupRect(std::vector<cv::Rect>& rectList, int groupThreshold, 
    double eps);

/**
 * Helper function to access the MSER method easier.
 *
 * @param img The source image to perform MSER on.
 * @param bboxes The vector to store the bounding boxes in.
 * @param delta Delta is the number of steps (changes in pixel brighness) 
 * MSER uses when comparing the size of connected regions.
 * @param max_variation Maximum variation of the regions area between delta 
 * threshold.
 * @param eps How much do the bounding boxes produced by MSER need to overlap 
 * so that they merge.
 *
 */
void mser(const cv::Mat& img, std::vector<cv::Rect>& bboxes, int minArea=50, 
        int maxArea=400000, int delta=5, int max_variation=5, float eps=.3);

void contourBbox(const cv::Mat& img, std::vector<cv::Rect>& bboxes, 
        int threshold, int minArea, int maxArea, float eps);

/**
 * Gets the next n frames from the video capture device capture device. Stores
 * them in the Mat, img.
 *
 * @param cap The video capture device
 * @param img The cv::Mat to store the n concatenated images in.
 * @param n The number of frames to grab from the capture device.
 * @param frameCounter REMOVE
 */
void getFrame(cv::VideoCapture cap, cv::Mat& img, int n);

/**
 * Segment the image by performing fielding and preprocessing the image, then
 * using the MSER algorithm create bounding boxes.
 *
 * @param img The cv::Mat image to segment.
 * @param imgCorrect The cv::Mat to store the flatfielded and preprocessed image.
 * @param bboxes The vector to store the bounding boxes produced by MSER.
 * @param options The Options struct that is used to store all of the command line
 * configurations.
 *
 */
void segmentImage(const cv::Mat& img, cv::Mat& imgCorrect, 
        std::vector<cv::Rect>& bboxes, Options options);

/**
 * Saves the segments produced by the segment image function.
 *
 * Crops the rectanbles, bboxes, out of the img and imgCorrect cv::Mat objects.
 * saveCrops will then save these crops in the imgDir.
 *
 * @param img The original image.
 * @param imgCorrect The flatfielded and preprocessed image.
 * @param bboxes The bounding boxes generated by the MSER algorithm.
 * @param imgDir The directory string to store the images.
 * @param imgName The name to store the image under in the measure file and imgDir.
 * @param measurePtr A ptr to the measure file. For example std::ofstream measurePtr(measureFile).
 * @param options The Options struct that is used to store all of the command line
 * configurations.
 */
void saveCrops(const cv::Mat& img, const cv::Mat& imgCorrect, 
        std::vector<cv::Rect>& bboxes, std::string imgDir, 
        std::string imgName, std::ofstream& measurePtr, Options options);

#endif
