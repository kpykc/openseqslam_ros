//
//  main.cpp
//  OpenSeqSLAM
//
//  Created by Saburo Okita on 14/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <stdio.h>
#include "OpenSeqSLAM.h"

using namespace std;
using namespace cv;


/**
 * Load the Nordland dataset
 **/
vector<Mat> loadDataset( string path ) {
    //char temp[100];
    std::ostringstream temp;
    vector<Mat> images;
    
    for( int i = 1; i < 35700; i += 100) {
        //sprintf( temp, "images-%05d.png", i );
	temp << "images-" << std::setw(5) << std::setfill('0') << i <<".png";
	//cout << temp.str() << endl;
        Mat image = imread( path + temp.str() );
	temp.str(std::string());
        images.push_back( image );
    }

    return images;
}



int main(int argc, const char * argv[])
{
	std::string winter_path;
    std::string spring_path;
	
	if (argc == 1) {
		winter_path.assign("/home/kp/datasets/nordland/64x32-grayscale-1fps/winter/");
		spring_path.assign("/home/kp/datasets/nordland/64x32-grayscale-1fps/spring/");
	} 
	else if (argc == 2) { // parent dir given
		winter_path = string(argv[1])+"/winter/";
		spring_path = string(argv[1])+"/spring/";
	}
	else if (argc == 3) { // datasets given separetely
		winter_path = string(argv[1]);
		spring_path = string(argv[2]);
	}
	else {
		//incorrect arguments
		cout << "Usage: fabmap_sample <sample data directory>" << endl;
		return -1;
	}
	
	cout << "winter dataset path: " << winter_path << endl 
		<< "spring dataset path: " << spring_path << endl;
	
    //winter_path("/home/kp/kpykc/repos/openseqslam/trunk/datasets/nordland/64x32-grayscale-1fps/winter/");
    //spring_path("/home/kp/kpykc/repos/openseqslam/trunk/datasets/nordland/64x32-grayscale-1fps/spring/");

    vector<Mat> spring = loadDataset( spring_path );
    vector<Mat> winter = loadDataset( winter_path );
    
    OpenSeqSLAM seq_slam;
    
    /* Preprocess the image set first */
    vector<Mat> preprocessed_spring = seq_slam.preprocess( spring );
    vector<Mat> preprocessed_winter = seq_slam.preprocess( winter );
    
    /* Find the matches */
    Mat matches = seq_slam.apply( preprocessed_spring, preprocessed_winter );
    
    
    // opencv w/o enabled QT (WITH_QT=ON)
    //CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(255, 0, 0) );
    //CvFont font;
    //cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4, 0, 1, 8);

    namedWindow("");
    moveWindow("", 0, 0);
    
    //char temp[100];
    std::ostringstream temp;
    float threshold = 0.99;
    
    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);
    
    for( int x = 0; x < spring.size(); x++ ) {
        int index = static_cast<int>(index_ptr[x]);

        /* Append the images together */
        Mat appended( 32, 64 * 2, CV_8UC3, Scalar(0) );
        spring[x].copyTo( Mat(appended, Rect(0, 0, 64, 32) ));
  
        if( score_ptr[x] < threshold )
            winter[index].copyTo( Mat(appended, Rect(64, 0, 64, 32) ));
        
        resize(appended, appended, Size(), 8.0, 8.0 );
        
        //sprintf( temp, "Spring [%03d]", x );
	temp << "Spring ["<< std::setfill('0') << std::setw(3) << x << "]";
        //addText( appended, temp, Point( 10, 20 ), font );
        cv::putText(appended, temp.str(), cv::Point(10,20), FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(0, 0, 255), 2, 8);

        /* The lower the score, the lower the differences between images */
        if( score_ptr[x] < threshold )
            //sprintf( temp, "Winter [%03d]", index );
            temp << "Spring [" << std::setfill('0') << std::setw(3) << index << "]";
        else
            //sprintf( temp, "Winter [None]" );
            temp << "Winter [None]";
        
        //addText( appended, temp, Point( 522, 20 ), font );
        cv::putText(appended, temp.str(), cv::Point(522, 20), FONT_HERSHEY_COMPLEX_SMALL, 1.5,  Scalar(0, 0, 255), 2, 8);
        
        imshow( "", appended );
	temp.str(std::string());
	waitKey(100);
	
    }
    
    return 0;
}
