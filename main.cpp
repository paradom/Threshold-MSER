/***********************************************
 *
 *  main.cpp
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

#include <iostream>
#include <fstream> // write output csv files
#include <iomanip>  // For the function std::setw

#include <filesystem>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "imageProcessing.hpp"

#if defined(WITH_OPENMP) and !defined(WITH_VISUAL)
    #include "omp.h" // OpenMP
#endif

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void helpMsg(string executable, Options options) {
    cout << "Usage: " << executable << " -i <input> [OPTIONS]\n"
        << left << setw(30) << "Segment a directory of images by utilizing the MSER method.\n\n"
        << left << setw(30) << "  -i, --input" << "Directory of video files to segment\n"
        << left << setw(30) << "  -o, --output-directory" << "Output directory where segmented images should be stored (Default: " << options.outputDirectory << ")\n"
        << left << setw(30) << "  -n, --num-concatenate" << "The number of frames that will be vertically concatenated (Default: " << options.numConcatenate <<  ")\n"
        << left << setw(30) << "  -s, --signal-to-noise" << "The cutoff signal to noise ratio that is used in determining\n"
        << left << setw(30) << "" << "which frames from the video file get segmented (Default: " << options.signalToNoise << ")\n"
        << left << setw(30) << "  -p, --outlier-percent" << "Percentage of darkest and lightest pixels to throw out before flat-fielding (Default: " << options.outlierPercent << ")\n"
        << left << setw(30) << "  -M, --maximum" << "Maximum area of a segmented blob (Default: " << options.maximum << ")\n" 
        << left << setw(30) << "  -m, --minimum" << "Minimum area of a segmented blob. (Default: " << options.minimum << ")\n"
        << left << setw(30) << "  -d, --delta" << "Delta is a parameter for MSER. Delta is the number of steps (changes\n"
        << left << setw(30) << "" << "in pixel brightness) MSER uses to compare the size of connected regions.\n" 
        << left << setw(30) << "" <<  "A smaller delta will produce more segments. (Default: " << options.delta << ")\n"
        << left << setw(30) << "  -v, --variation" << "Maximum variation of the region's area between delta threshold.\n"
        << left << setw(30) << "" <<  "Larger values lead to more segments. (Default: " << options.variation << ")\n" 
        << left << setw(30) << "  -e, --epsilon" << "Float between 0 and 1 that represents the maximum overlap between\n"
        << left << setw(30) << "" << "two rectangle bounding boxes. 0 means that any overlap will mean\n"
        << left << setw(30) << "" << "that the bounding boxes are treated as the same. (Default: " << options.epsilon << ")\n" << endl;
}

int main(int argc, char **argv) {
    // Set the default options
    Options options;
    options.input = "";
    options.outputDirectory = "out";
    options.signalToNoise = 50;
    options.outlierPercent = .20;
    options.numConcatenate = 1;
    options.minimum = 50;
    options.maximum = 400000;
    options.epsilon = 1;
    options.delta = 10;
    options.variation = 20;

    // TODO: more robust options with std::find may be worth it
    if (argc == 1) {
        helpMsg(argv[0], options);
    }

    int i = 1;
	while (i < argc) {
        // Display the help message
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            helpMsg(argv[0], options);

            return 1;
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            options.input = argv[i + 1];
            if ( !fs::exists(options.input) ) {
                cerr << options.input << " does not exist." << endl;
                return 1;
            }
            i+=2;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-directory") == 0) {
            options.outputDirectory = argv[i + 1];
            try {
                fs::create_directories(options.outputDirectory);
            }
            catch(fs::filesystem_error const& ex){
                cerr << ex.what() <<  endl;
                return 1;
            }
            i+=2;
		} else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--signal-to-noise") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                cerr << argv[i+1] << " is not a valid input. Signal To Noise ration must be an integer." << endl;
                return 1;
            }
            options.signalToNoise = stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num-concatenate") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                cerr << argv[i+1] << " is not a valid input. The number of frames to concatenate must be an integer." << endl;
                return 1;
            }
            options.numConcatenate = stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--maximum") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                cerr << argv[i+1] << " is not a valid input. Maximum must be an integer." << endl;
                return 1;
            }
            options.maximum = stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--minimum") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                cerr << argv[i+1] << " is not a valid input. Minimum must be an integer." << endl;
                return 1;
            }
            options.minimum = stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epsilon") == 0) {
            // Validate the input type
            options.epsilon = stof(argv[i+1]); // FIXME: may throw error if not int

            if (options.epsilon < 0) {
                cerr << options.epsilon << " is not a valid input. Epsilon must be a non-negative float." << endl;
                return 1;
            }
            i+=2;
		} else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--delta") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                cerr << argv[i+1] << " is not a valid input. Delta must be an integer." << endl;
                return 1;
            }
            options.delta = stoi(argv[i+1]);
            i+=2;

		} else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--variation") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                cerr << argv[i+1] << " is not a valid input. Variation must be an integer." << endl;
                return 1;
            }
            options.variation = stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--outlier-percent") == 0) {
            // Validate the input type
            options.outlierPercent = stof(argv[i+1]); // FIXME: may throw error if not int

            if (options.outlierPercent > 1 or options.outlierPercent < 0) {
                cerr << options.outlierPercent << " is not a valid input. Outlier percent must be a float between 0 and 1." << endl;
                return 1;
            }
            i+=2;
		} else {
            // Display invalid option message
            cerr << argv[0] << ": invalid option \'" << argv[i] << "\'" <<
                "\nTry \'" << argv[0] << " --help\' for more information." << endl;

            return 1;
            i+=2;
		}
	}
    if ( options.input == "" ) {
        cerr << "Must have either an input directory or an input file for segmentation to be run on" << endl;
        return 1;
    }

	// create output directories
    fs::create_directory(options.outputDirectory);
	std::string measureDir = options.outputDirectory + "/measurements";
    fs::create_directory(measureDir);
	std::string segmentDir = options.outputDirectory + "/segmentation";
    fs::create_directory(segmentDir);
    
    // Create vector of video files from the input which can either be a directory 
    // or a single avi file.
    std::vector<fs::path> files;
    if (fs::is_directory(options.input)) {
        for(auto& p: fs::directory_iterator{options.input}) {
            fs::path file(p);

            string ext = file.extension();

            string valid_ext[] = {".avi", ".mp4", ".png"};
            int len = sizeof(valid_ext)/sizeof(valid_ext[0]);
            if (!containExt(ext, valid_ext, len)) {
                continue;
            }
            files.push_back(file);
        }
    } 
    else {
        files.push_back(fs::path(options.input));
    }

    int numFiles = files.size();
    if (numFiles < 1 ) {
        cerr << "No files to segment were found." << endl;
        return 1;
    }

    #pragma omp parallel for
    for (int i=0; i<numFiles; i++) {
        fs::path file = files[i];
        std::string fileName = file.stem();

        // Create a measurement file to save crop info to
        std::string measureFile = measureDir + "/" + fileName + ".csv";
        std::ofstream measurePtr(measureFile);
        measurePtr << "image,area,major,minor,perimeter,x1,y1,x2,y2,height" << endl; 

        // TODO: Add a way to check if file is valid

        // FIXME: cap.read() and cap.grad() are not working properly, aren't throwing errors when reading image
        // This is a temporary solution to determine if the input file is an image or video
        cv::Mat testFrame;
        string ext = file.extension();
        bool validImage = (ext == ".png");

        // If the file is a video
        if (!validImage) {
            cv::VideoCapture cap(file.string());
            // cap.open(file.string(), CAP_ANY);
            if (!cap.isOpened()) {
                cerr << "Invalid file: " << file.string() << endl;
                continue;
            }

	        int image_stack_counter = 0;
            int frameNumber = cap.get(CAP_PROP_FRAME_COUNT);

            for (int j=0; j<frameNumber-1; j++) 
            {   
	        	cv::Mat imgRaw;
                #pragma omp critical(getframe)
                {
                    getFrame(cap, imgRaw, j, options.numConcatenate);
                }

                image_stack_counter += options.numConcatenate;
                std::string imgName = fileName + "_" + convertInt(image_stack_counter, 4);
                std::string imgDir = segmentDir + "/" + fileName + "/" + imgName;
                fs::create_directories(imgDir);

                segmentImage(imgRaw, options, imgDir, measurePtr, imgName);
	        }
	        // When video is done being processed release the capture object
	        cap.release();
        }
        // If the file is an image 
        else {
	    	cv::Mat imgRaw = cv::imread(file.string());
            cv::Mat imgGray;
	        cv::cvtColor(imgRaw, imgGray, cv::COLOR_RGB2GRAY);

            if (imgGray.empty()) {
                cerr << "Error reading the image file " << file.string() << endl;
                continue;
            }
            // TODO: Add the ability to concatenate frames like with videos

            std::string imgName = fileName;
            std::string imgDir = segmentDir + "/" + imgName;
            fs::create_directories(imgDir);

            segmentImage(imgGray, options, imgDir, measurePtr, imgName);
        }

        measurePtr.close();
    }
    return 0;
}
