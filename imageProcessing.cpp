/***********************************************
 *
 *  imageProcessing.cpp
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

#include "imageProcessing.hpp"
#include <iostream>
#include <fstream> // write output csv files
#include <iomanip>  // std::setw
#include <filesystem>
#include <tuple>

#include <opencv2/videoio.hpp> // used for the video preprocessing
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp> // simpleBlobDector
#include <opencv2/objdetect.hpp> // groupRectangles
#include <opencv2/imgproc.hpp> // gaussian blur

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

bool containExt(const std::string s, string arr[], int len) {
    for (int i=0; i<len; i++) {
        if(s == arr[i]) {
            return true;
        }
    }
    return false;
}

bool isInt(std::string str) {
    for (int i = 0; i < str.length(); i++) {
        if (! isdigit(str[i]))
            return false;
    }
    return true;
}

/*
 * Function: convertInt
 *
 * Adds "fill" padding 0's to an input "number". Fill has a default value of 4.
 *
 * return: std::string object with concatenated 0's
 */
std::string convertInt(int number, int fill) {
	std::stringstream ss; //create a stringstream
	ss << std::setw(fill) << std::setfill('0') << number; // add number to the stream
	return ss.str(); //return a string with the contents of the stream
}

/*
 * Combines multiple frames into a single frame
 *
 *
 */
void getFrame(cv::VideoCapture cap, cv::Mat& img, int& frameCounter, int numConcatenate) {
    cv::Mat frame;
    cv::Mat frameArray[numConcatenate] = {};
    for(int k=0; k<numConcatenate; k++){
        cap.read(frame);
        cv::cvtColor(frame, frameArray[k], cv::COLOR_RGB2GRAY);
    }
    cv::vconcat(frameArray, numConcatenate, img);
    frameCounter += numConcatenate - 1;
}

/*
 * dst = src
 *
 * Anything above thresh is set to 255 (white)
 */
void chopThreshold(const cv::Mat &src, cv::Mat &dst, int thresh){
    cv::Mat imgOtsu;
    cv::threshold(src, imgOtsu, thresh, 255, THRESH_TOZERO_INV);

    cv::Mat mask;
    cv::compare(src,imgOtsu,mask,CMP_EQ);
    cv::Mat imgThresh(src.size(), CV_8UC1, Scalar(255));
    src.copyTo(dst, mask);
}

void segmentImage(cv::Mat img, Options options, std::string imgDir, std::ofstream& measurePtr, std::string imgName) {
    cv::Mat imgCorrect;
    flatField(img, imgCorrect, options.outlierPercent);

    std::vector<cv::Rect> bboxes;

    #if defined(VISUAL_MODE)
    cv::imshow("viewer", img);
    cv::waitKey(0); 

    cv::imshow("viewer", imgCorrect);
    cv::waitKey(0); 
    #endif

    cv::Mat imgProcess;
    preprocess(imgCorrect, imgProcess);

    // If the SNR is less than options.signalToNoise then the image will have many false segments
    float imgSNR = SNR(img);

    #if defined(VISUAL_MODE)
    cout << "Image SNR: " << imgSNR << endl;
    #endif
    if (imgSNR > options.signalToNoise) {
        mser(imgProcess, bboxes, options.delta, options.variation, options.epsilon);
    } else {
        // Create a mask that includes all of the regions of the image with
        // the darkest pixels which MSER method can be performed on.
        int thresh = 160; // Larger thresh means more segments
        cv::Mat imgThresh;
        cv::threshold(imgProcess, imgThresh, thresh, 255, THRESH_BINARY);

	    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
	    cv::Mat imgCorrectMask(img.size(), img.type(), cv::Scalar(255));

        bool contour = true;
        bool mesh = false;
        if (contour) {
            std::vector<std::vector <cv::Point> > contours; // Vector for storing contour
            std::vector<Vec4i> hierarchy;
            cv::findContours(imgThresh, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_L1);
            cv::Mat imgCorrectBbox;
                cv::cvtColor(imgCorrect, imgCorrectBbox, cv::COLOR_GRAY2RGB);

            std::vector<cv::Rect> boundRect(contours.size());

            // create mask based on darkest regions
            cv::Rect maskRect(0, 0, mask.cols, mask.rows); // use maskRect to make sure box doesn't go off the edge
            for(int i=0; i<boundRect.size(); i++){
                boundRect[i] = cv::boundingRect(contours[i]);
                if (boundRect[i].area() > options.maximum)
                    continue;

                float scaleFactor = 1.2;
                cv::Rect largeRect = rescaleRect(boundRect[i], scaleFactor);
                cv::Mat roi = mask(largeRect & maskRect);
                roi.setTo(255);
            }
        }
        if (mesh) {
            Point anchor = cv::Point(-1, -1); // default anchor value
            cv::Mat openKernel = getStructuringElement( cv::MORPH_ELLIPSE,
                        cv::Size(4, 4),
                        anchor);
            cv::morphologyEx(imgThresh, imgThresh, cv::MORPH_HITMISS, openKernel);
            cv::erode(imgThresh, imgThresh, openKernel);
            cv::bitwise_not(imgThresh, mask);
        }

        imgCorrect.copyTo(imgCorrectMask, mask);

        #if defined(VISUAL_MODE)
        cv::imshow("viewer", imgCorrectMask);
        cv::waitKey(0);
        #endif

        mser(imgCorrectMask, bboxes, options.delta, options.variation, options.epsilon);
    }

    saveCrops(img, imgCorrect, bboxes, options, imgDir, measurePtr, imgName);
}


void saveCrops(cv::Mat img, cv::Mat imgCorrect, std::vector<cv::Rect> bboxes, Options options, std::string imgDir, std::ofstream& measurePtr, std::string imgName) {
	// Create crop directories
	std::string correctCropDir = imgDir + "/corrected_crop";
    fs::create_directory(correctCropDir);
	std::string rawCropDir = imgDir + "/original_crop";
    fs::create_directory(rawCropDir);
	std::string frameDir = imgDir + "/frame/";
    fs::create_directory(frameDir);

	// Write full video frames to files
	std::string correctedFrame = frameDir + "/" + imgName + "_corrected.tif";
	cv::imwrite(correctedFrame, imgCorrect);
	std::string originalFrame = frameDir + "/" + imgName + "_original.tif";
	cv::imwrite(originalFrame, img);
    
    // Save image with bounding boxes
	cv::Mat imgBboxes;
	cv::cvtColor(imgCorrect, imgBboxes, cv::COLOR_GRAY2RGB);

	int num_bboxes = bboxes.size();
    for (int k=0; k<num_bboxes; k++) {
        // Get measurement data
        float area = bboxes[k].area();

        // Determine if the bbox is too large or small
        if ( area < options.minimum || area > options.maximum )
            continue;

        float perimeter = bboxes[k].height + bboxes[k].width * 2.0;
        float x = bboxes[k].x;
        float y = bboxes[k].y;
        float major;
        float minor;
        float height = bboxes[k].height;
        float width = bboxes[k].width;
        
        // Determine if box is irregularly shapped (Abnormally long and thin)
        int hwRatio = 10;
        if ( width < 30 && height > hwRatio * width )
            continue;

        if ( height > width ) {
            major = height;
            minor = width;
        } else {
            major = width;
            minor = height;
        }

        // Determine if the bbox is too large or small
        if ( area < options.minimum || area > options.maximum )
            continue;
        auto tl = bboxes[k].tl(); 
        auto br = bboxes[k].br();

        std::string correctImgFile = correctCropDir + "/" + imgName + "_" + "crop_" + convertInt(k) + ".png";
        std::string rawImgFile = rawCropDir + "/" + imgName + "_" + "crop_" + convertInt(k) + ".png";
        #pragma omp critical(write)
        {
            // Format: img,area,major,minor,perimeter,x,y,height
            // TODO: store additional fields (squiggly vs straight), gray level, etc
            measurePtr << correctImgFile << "," << area << "," << major << "," << minor << "," 
                << perimeter << "," << x << "," << y << "," << br.x << "," << br.y
                << "," << height << endl; 
        }

        // Re-scale the crop of the image after getting the measurement data written to a file
        cv::Rect scaledBbox = rescaleRect(bboxes[k], 1.5);

        // Create a new crop using the intersection of rectangle objects and the image
        cv::Rect imgRect(0, 0, imgCorrect.cols, imgCorrect.rows); // use imgRect to make sure box doesn't go off the edge
        cv::Mat imgCropCorrect = Mat(imgCorrect, scaledBbox & imgRect); // TODO: check the speed of this operation
        cv::imwrite(correctImgFile, imgCropCorrect);

        // Crop the original image
        cv::Mat imgCropRaw = Mat(img, scaledBbox & imgRect);
        cv::imwrite(rawImgFile, imgCropRaw);

	    // Draw the cropped frames on the image to be saved
	    cv::rectangle(imgBboxes, bboxes[k], cv::Scalar(0, 0, 255));
    }

    #if defined(VISUAL_MODE)
    cv::imshow("viewer", imgBboxes);
    cv::waitKey(0);
    #endif
}

// Rescale the crop of the image after getting the measurement data written to a file
cv::Rect rescaleRect(const cv::Rect &rect, float scale)
{
    float scaleWidth = rect.width * scale;
    float scaleHeight = rect.height * scale;
    cv::Rect scaledRect(rect.x - scaleWidth / 2, rect.y - scaleHeight / 2, 
            rect.width + scaleWidth, rect.height + scaleHeight);

    return scaledRect;
} 

/*
 * This class can be used to pass into the partition function in order to create 
 * a grouping of the rectangles accross the image
 *
 */
class OverlapRects
{
public:
    OverlapRects(double _eps) : eps(_eps) {}
    inline bool operator()(const cv::Rect& r1, const cv::Rect& r2) const
    {
        // TODO: Throw error for invalid input
        // If eps = 0: Any parts of r1 and r2 overlapping return true
        // If eps = 1: Smaller rectangle needs to be entirely within larger rectangle to match
        float min_area = std::min(r1.area(), r2.area()) * eps; 

        return (r1 & r2).area() >= min_area;
    }
    double eps;
};

/*
 * This class can be used to pass into the partition function in order to create
 * a grouping of the rectangles accross the image
 *
 * If eps < 1, then rectangles need to overlap more to be grouped
 * If eps >= 1, then rectangles close to each other will be grouped
 *
 */
class OverlapRects2
{
public:
    OverlapRects2(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double deltax = eps * (r1.width + r2.width) * 0.5;
        double deltay = eps * (r1.height + r2.height) * 0.5;
        return std::abs(r1.x - r2.x) <= deltax &&
            std::abs(r1.y - r2.y) <= deltay;
    }
    double eps;
};

/*
 * Function: groupRect
 *
 * Reimplements the OpenCV groupRectangles function to group rectangles by our own
 * mechanism. 
 * This function allows control over how much of the rectangles need to overlap as
 * well as how the grouped rectangles are used to produce the final ROI (TODO).
 *
 */
void groupRect(std::vector<cv::Rect>& rectList, int groupThreshold, double eps)
{ 
    if( rectList.empty() )
    {
        return;
    }

    std::vector<int> labels;
    // Third argument of partion is a predicate operator that looks for a method of the class
    // that will return true when elements are apart of the same partition
    int nclasses = partition(rectList, labels, OverlapRects2(eps));

    // labels correspond to the location of the rectangle in space 
    std::vector<cv::Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    int nlabels = (int)labels.size();

    bool method1 = true;
    bool method2 = false;

    if(method1) {
        int uniqueLabels = 0;
        for (int i = 0; i < nlabels; i++)
        {
            int cls = labels[i];
            if (rectList[i].width > rrects[cls].width)
            {
                rrects[cls].width = rectList[i].width;
                rrects[cls].x = rectList[i].x;
                rrects[cls].y = rectList[i].y;
            }
            if (rectList[i].height > rrects[cls].height)
            {
                rrects[cls].height = rectList[i].height;
                rrects[cls].x = rectList[i].x;
                rrects[cls].y = rectList[i].y;
            }
            if (rweights[cls] == 0)
                uniqueLabels++;
            rweights[cls]++;
        }
    }
    if(method2){
        for (int i = 0; i < nlabels; i++)
        {
            int cls = labels[i];
            rrects[cls] = rrects[cls] | rectList[i];
            rweights[cls]++;
        }
    }

    rectList.clear();

    for (int i = 0; i < nclasses; i++)
    {
        if (rweights[i] >= groupThreshold) {
            rectList.push_back(rrects[i]);
        }
    }

}


void mser(cv::Mat img, std::vector<cv::Rect>& bboxes, int delta, int max_variation, float eps) {
    int min_area = 50;
    int max_area = 400000;
    
    cv::Ptr<cv::MSER> detector = cv::MSER::create(delta, min_area, max_area, max_variation); 
	std::vector<std::vector<cv::Point>> msers;
	detector->detectRegions(img, msers, bboxes);

    #if defined(VISUAL_MODE)
	cv::Mat img_bboxes;
	cv::cvtColor(img, img_bboxes, cv::COLOR_GRAY2RGB);
	int size_more = bboxes.size();
	for(int i=0;i<size_more;i++){
		cv::rectangle(img_bboxes, bboxes[i], cv::Scalar(0, 0, 255));
	}
    cv::imshow("viewer", img_bboxes);
    cv::waitKey(0);
    #endif

    // Reimpliment OpenCV groupRectangles with different funtionality to merge MSER 
    // bounding boxes
    int minBboxes = 2;

    groupRect(bboxes, minBboxes, eps);
    // groupRectangles(bboxes, minBboxes, eps);
}


void preprocess(const cv::Mat &src, cv::Mat &dst) {
    // Perform image pre processing
    float erosion_size = 2;
    cv::Mat erodeElement = getStructuringElement( cv::MORPH_ERODE,
               cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
               cv::Point( erosion_size, erosion_size ) );
    
    // open is useful in removing noise in the image
    Point anchor = cv::Point(-1, -1);
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
               cv::Size(7, 7),
               anchor);

    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel); // open is a combination of erosion and dialation
}

float SNR(cv::Mat& img) {
    // perform histogram equalization
    cv::Mat imgHeq;
    cv::equalizeHist(img, imgHeq);

    // Calculate Signal To Noise Ratio (SNR)
    cv::Mat imgClean, imgNoise;
    cv::medianBlur(imgHeq, imgClean, 3);
    imgNoise = imgHeq - imgClean;
    double SNR = 20*( cv::log(cv::norm(imgClean,cv::NORM_L2) / cv::norm(imgNoise,cv::NORM_L2)) );

    return SNR;
}

void flatField(cv::Mat& src, cv::Mat& dst, float outlierPercent) {
    cv::Mat imgBlack = cv::Mat::zeros(src.size(), src.type());
    cv::Mat imgCalib = cv::Mat::zeros(src.size(), src.type());
    
    // Get the calibration image
    trimMean(src, imgCalib, outlierPercent, 1); // 1 use only one channel
    
    cv::Mat imgCorrect(src.size(), src.type()); // creates mat of the correct size and type
    cv::addWeighted(src, 1, imgBlack, -1, 0, src); // subtracts an all black array
    cv::addWeighted(imgCalib, 1, imgBlack, -1, 0, imgCalib);
    cv::divide(src, imgCalib, dst, 255); // performs the flat fielding by dividing the arrays
}

void trimMean(const cv::Mat& img, cv::Mat& tMean, float outlierPercent, int nChannels) {
	// trimMean calculates the trimmed mean of the values in img.
	// If img is a vector, tMean is the mean of img, excluding the highest and lowest k data values,
	// where k=n*(outlierPercent/100)/2 and where n is the number of values in img. For a matrix input,
	// tMean is a row vector containing the trimmed mean of each column of img. For n-D arrays,
	// trimMean operates along the first non-singleton dimension. outlierPercent is a scalar between 0 and 100d

	switch (nChannels){
	    case 1: {
	    	tMean.create(img.rows, img.cols, CV_8UC1);

	    	cv::Mat bChannel(img.rows, img.cols, CV_8UC1); // black color channel
	    	cv::Mat out[] = {bChannel};
	    	cv::Mat tempCol;
	    	cv::Mat maskCol;

	    	cv::Scalar meanCol;
	    	double meanCol1;
	    	int cnt1, cnt2; 
	    	int k, n;
	    	int height = img.rows, width = img.cols;
	    	int from_to[] = {0,0};

            // mask removes the top and bottom n elements
	    	n = height;
	    	k = round(n*(outlierPercent/100)/2);
	    	maskCol.create(n, 1, CV_8UC1);

	    	for(cnt1=0;cnt1<height;cnt1++){
	    		if ((cnt1>=0 && cnt1<=k) || ((cnt1>=(height-k) && cnt1<=height)) ) {
	    			maskCol.at<int8_t>(cnt1,0) = 0;
	    		} else {
	    			maskCol.at<int8_t>(cnt1,0) = 1;
	    		}
	    	}

	    	// Split the color channels
	    	mixChannels(&img, 1, out, 1, from_to, 1); 

	    	cv::sort(bChannel, bChannel, cv::SORT_EVERY_COLUMN);
	    	for(cnt2=0;cnt2<width;cnt2++){
	    		tempCol = bChannel.col(cnt2);
	    		tempCol = tempCol.mul(maskCol);

	    		meanCol = mean(tempCol);
	    		meanCol1 = meanCol[0];
	    		bChannel.col(cnt2) = cv::Mat::ones(height, 1, bChannel.type())*meanCol1; // make a column that has the average value of the column
	    	}
	    	merge(out, 1, tMean);

	    	break;
	    }
	    case 3: {
	    	tMean.create(img.rows,img.cols, CV_8UC3);

	    	cv::Mat rChannel(img.rows,img.cols, CV_8UC1),gChannel(img.rows,img.cols, CV_8UC1),bChannel(img.rows,img.cols, CV_8UC1);
	    	cv::Mat out[] = {bChannel,gChannel,rChannel};
	    	cv::Mat tempCol;
	    	cv::Mat maskCol;//( 53, 71, cv::CV_64FC1, cv::Scalar::all( 0.0 ) );

	    	cv::Scalar meanCol;
	    	double meanCol1;
	    	int cnt1,cnt2;//,cnt3; // counters for all dimensions of image (cnt1:row, cnt2:col, cnt3:channels)
	    	int k,n;
	    	int height = img.rows, width = img.cols;// channels = img.channels();
	    	int from_to[] = {0,0, 1,1, 2,2};

	    	n = height;
	    	k = round(n*(outlierPercent/100)/2);
	    	maskCol.create(n, 1, CV_8UC1);

	    	for(cnt1=0;cnt1<height;cnt1++){
	    		if ((cnt1>=0 && cnt1<=k) || ((cnt1>=(height-k) && cnt1<=height)) ) {
	    			maskCol.at<int8_t>(cnt1,0) = 0;
	    		} else {
	    			maskCol.at<int8_t>(cnt1,0) = 1;
	    		}
	    	}

	    	// Split color channels
	    	mixChannels(&img, 1, out, 3, from_to, 3);

	    	// Blue channel
	    	cv::sort(bChannel,bChannel, cv::SORT_EVERY_COLUMN);
	    	for(cnt2=0;cnt2<width;cnt2++){
	    		tempCol = bChannel.col(cnt2);
	    		tempCol = tempCol.mul(maskCol);

	    		meanCol = mean(tempCol);
	    		meanCol1 = meanCol[0];
	    		bChannel.col(cnt2) = cv::Mat::ones(height, 1, bChannel.type())*meanCol1;
	    	}

	    	// Green channel
	    	cv::sort(gChannel,gChannel, cv::SORT_EVERY_COLUMN);
	    	for(cnt2=0;cnt2<width;cnt2++){
	    		tempCol = gChannel.col(cnt2);
	    		tempCol = tempCol.mul(maskCol);
	    		meanCol = mean(tempCol);
	    		meanCol1 = meanCol[0];
	    		gChannel.col(cnt2) = cv::Mat::ones(height, 1, gChannel.type())*meanCol1;
	    	}

	    	// Red channel
	    	cv::sort(rChannel,rChannel, cv::SORT_EVERY_COLUMN);
	    	for(cnt2=0;cnt2<width;cnt2++){
	    		tempCol = rChannel.col(cnt2);
	    		tempCol = tempCol.mul(maskCol);
	    		meanCol = mean(tempCol);
	    		meanCol1 = meanCol[0];
	    		rChannel.col(cnt2) = cv::Mat::ones(height, 1, rChannel.type())*meanCol1;
	    	}

	    	merge(out, 3, tMean);
	    	break;
	    }
	    default:{
	    	std::cout << "Raw image should have either one channel (grayscale image) or three channels (RGB image)"<< std::endl;
	    	break;
	    }
	}
}
