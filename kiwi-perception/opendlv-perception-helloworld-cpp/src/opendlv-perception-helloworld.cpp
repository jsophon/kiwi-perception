/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <ctime>
using namespace std;
using namespace cv;

int32_t main(int32_t argc, char **argv) {
	
    time_t now = time(0);
    tm *ltm = localtime(&now);
    std::string timeString = std::to_string(1 + ltm->tm_hour) + "-" + std::to_string(1 + ltm->tm_min); // + "-" + std::to_string(1 + ltm->tm_sec);

    cluon::data::TimeStamp startTime = cluon::time::now();
    int64_t startTimeUs = cluon::time::toMicroseconds(startTime);
    ofstream fs;
    const std::string filename = "/tmp/blueCones-" + timeString + "-.csv";
    fs.open(filename, ios::out | ios::app);
    
    ofstream fs2;
    const std::string filename2 = "/tmp/yellowCones-" + timeString + "-.csv";
    fs2.open(filename2, ios::out | ios::app);
    fs << "X," << "Y," << "Frame," << "Time (s)," << std::endl;
    fs2 << "X," << "Y," << "Frame," << "Time (s)," << std::endl;
    
    std::string savedFilepath = "/tmp/Kiwi/";
    std::string savedFilePrefix = "Kiwi";
    uint8_t kernelData[] = {1,1,1,1,1,1};
    cv::Mat kernel(3,3,CV_8U,kernelData);
    cv::CascadeClassifier cascadeKiwi;
    if(!cascadeKiwi.load("/tmp/mixKiwi.xml"))
    {
       std::cout << "cannot load training file" << std::endl;
    }
    double currentFrame = 0;
    double stepFrame = 1;
    double skipFrame = 7;
    double countFrame = 0;
    double lowerLimit = 150;
    double upperLimit = 350;
    double distance = 100;
    double pdPosition = 0;
    bool kiwiExist = false;
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ) {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=640 --height=480 --verbose" << std::endl;
    }
    else {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        const bool VERBOSE{commandlineArguments.count("verbose") != 0};

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid()) {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            // Interface to a running OpenDaVINCI session; here, you can send and receive messages.
            cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

            // Handler to receive distance readings (realized as C++ lambda).
            std::mutex distancesMutex;
            float front{0};
            float rear{0};
            float left{0};
            float right{0};
            auto onDistance = [&distancesMutex, &front, &rear, &left, &right](cluon::data::Envelope &&env){
                auto senderStamp = env.senderStamp();
                // Now, we unpack the cluon::data::Envelope to get the desired DistanceReading.
                opendlv::proxy::DistanceReading dr = cluon::extractMessage<opendlv::proxy::DistanceReading>(std::move(env));

                // Store distance readings.
                std::lock_guard<std::mutex> lck(distancesMutex);
                switch (senderStamp) {
                    case 0: front = dr.distance(); break;
                    case 2: rear = dr.distance(); break;
                    case 1: left = dr.distance(); break;
                    case 3: right = dr.distance(); break;
                }
            };
            // Finally, we register our lambda for the message identifier for opendlv::proxy::DistanceReading.
            od4.dataTrigger(opendlv::proxy::DistanceReading::ID(), onDistance);

            // Endless loop; end the program by pressing Ctrl-C.
            while (od4.isRunning()) {
                cv::Mat img;
                cv::Mat imgCopy;

                // Wait for a notification of a new frame.
                sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock
                    cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
                    img = wrapped.clone();
                }
                sharedMemory->unlock();
                //imgCopy =  img;

                // TODO: Do something with the frame.

                cv::Mat hsv;
                cv::cvtColor(img,hsv,cv::COLOR_BGR2HSV);
					
                //////Blue Cones
                cv::Scalar hsvLowBlue(110,50,50);
                cv::Scalar hsvHighBlue(130,255,255);
                cv::Mat blueCones;
                cv::inRange(hsv,hsvLowBlue,hsvHighBlue,blueCones);
                
                cv::Mat dilate;
                uint32_t iterations(5);
                cv::dilate(blueCones,dilate,cv::Mat(),cv::Point(-1,-1),iterations,1,1);  

                cv::Mat erode;
                cv::erode(dilate,erode,cv::Mat(),cv::Point(-1,-1),iterations,1,1);

                std::vector<std::vector<cv::Point> > contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(erode,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
                //////Blue Cones

                //////Yellow Cones
                cv::Scalar hsvLowYellow(20,30,30);
                cv::Scalar hsvHighYellow(70,255,255);
                cv::Mat yellowCones;
                cv::inRange(hsv,hsvLowYellow,hsvHighYellow,yellowCones);
								
                cv::Mat dilateYellow;
                cv::dilate(yellowCones,dilateYellow,cv::Mat(),cv::Point(-1,-1),iterations,1,1);  

                cv::Mat erodeYellow;
                cv::erode(dilateYellow,erodeYellow,cv::Mat(),cv::Point(-1,-1),iterations,1,1);

                std::vector<std::vector<cv::Point> > contoursYellow;
                std::vector<cv::Vec4i> hierarchyYellow;
                cv::findContours(erodeYellow,contoursYellow,hierarchyYellow,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
                //////Yellow Cones

                cv::Mat drawingBlue = cv::Mat::zeros(erode.size(), CV_8UC3 );
                auto contourSize = contours.size();


                currentFrame = currentFrame + stepFrame;
                cluon::data::TimeStamp sampleTime = cluon::time::now();
                int64_t currentTime = cluon::time::toMicroseconds(sampleTime);
                double deltaTime = (currentTime - startTimeUs)/1000000.0;

                for( size_t i = 0; i < contourSize; i++ )
                {
                  auto cnt = contours[i];
                  //std::cout << "Area : " << cv::contourArea(cnt) << std::endl; SHOW THE AREA OF CONTOUR
                  auto area = cv::contourArea(cnt);

                  cv::Rect rect = cv::boundingRect(cnt);
                  cv::Point pt1, pt2;
                  pt1.x = rect.x;
                  pt1.y = rect.y;
                  pt2.x = rect.x + rect.width;
                  pt2.y = rect.y + rect.height;
                  if((area > 100 && area < 4000) && (rect.width > 15 && rect.width < 40) && (rect.height > 30 && rect.height < 80) && rect.width*1.3 < rect.height) //Condition to detect cone
                  {
                    auto xPoint = rect.x  + (rect.width/2);
                    auto yPoint = rect.y  + rect.height;
                    cv::Scalar color = cv::Scalar( 217, 255, 0 );
                    std::vector<cv::Point> hull;
                    cv::convexHull(contours[i], hull);
                    cv::polylines(img, hull, true, color);
                    cv::rectangle(img, pt1, pt2, CV_RGB(255,0,0), 1);
                    cv::circle(drawingBlue, cv::Point(xPoint, yPoint),1, color, 2 );
                    
                    if(fs.is_open())
                    {
                      fs << xPoint << "," << yPoint << "," << currentFrame << "," << deltaTime << std::endl;
                    }
                    else
                    {
                      std::cout << "File did not open. "<< std::endl;
                    }
                    
                  }
                }

                cv::Mat drawingYellow = cv::Mat::zeros(erodeYellow.size(), CV_8UC3 );
                for( size_t i = 0; i< contoursYellow.size(); i++ )
                {
                  auto cnt = contoursYellow[i];
                  auto area = cv::contourArea(cnt);
                  cv::Rect rect = cv::boundingRect(cnt);
                  cv::Point pt1, pt2;
                  pt1.x = rect.x;
                  pt1.y = rect.y;
                  pt2.x = rect.x + rect.width;
                  pt2.y = rect.y + rect.height;
                  if(area > 100 && area < 3000 && rect.width*1.3 < rect.height) //Condition to detect cone
                  {
                    auto xPoint = rect.x  + (rect.width/2);
                    auto yPoint = rect.y  + rect.height;
                    cv::Scalar color = cv::Scalar( 0, 232, 255 );				
                    std::vector<cv::Point> hull;
                    cv::convexHull(contoursYellow[i], hull);
                    cv::polylines(img, hull, true, color);		
                    cv::rectangle(img, pt1, pt2, CV_RGB(255,0,0), 1);	
                    cv::circle(drawingYellow, cv::Point(xPoint, yPoint),1, color, 2 );	
                    if(fs2.is_open())
                    {
                      fs2 << xPoint << "," << yPoint << "," << currentFrame << "," << deltaTime << std::endl;
                    }
                    else
                    {
                      std::cout << "File did not open. "<< std::endl;
                    }
                  }
                }
                cv::Mat twoColor;
                twoColor = drawingYellow + drawingBlue;
                /*  
                if(currentFrame >= 500)
                {
                  fs.close();
                  fs2.close();
                }
                */
                //Added Code
								
                ///////Added Code KIWI
                countFrame = countFrame + 1;
                if(countFrame >= skipFrame)
                {
                  //cv::imwrite(savedFilepath + savedFilePrefix + to_string(int(currentFrame)) + ".png" ,imgCopy);
                  countFrame = 0;
                  cv::Mat kiwi_split[3];
                  cv::Mat kiwiROI;
                  cv::split(hsv,kiwi_split);
                  cv::Mat kiwi_gray = kiwi_split[2];
                  std::vector<Rect> foundKiwi;
                  cascadeKiwi.detectMultiScale( kiwi_gray, foundKiwi, 1.05, 1, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
                  //rects = detector.detectMultiScale(kiwi_gray[2], scaleFactor=1.05, minNeighbors=1, minSize=(100,100))
                  if(foundKiwi.size() > 0)
                  {
                    kiwiExist = true;
                    cv::Point pt1, pt2;
                    double kiwiWidth = 0;
                    for( size_t i = 0; i < foundKiwi.size(); i++ )
                    {
                      auto cntCascade = foundKiwi[i];
                      pt1.x = cntCascade.x;
                      pt1.y = cntCascade.y;
                      pt2.x = cntCascade.x + cntCascade.width;
                      pt2.y = cntCascade.y + cntCascade.height;
                      Point center( int(round(foundKiwi[i].x + foundKiwi[i].width*0.5)), int(round(foundKiwi[i].y + foundKiwi[i].height*0.5 )));
                      //ellipse( kiwi_gray, center, Size( int(round(foundKiwi[i].width*0.5)), int(round(foundKiwi[i].height*0.5))), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
                      kiwiROI = kiwi_gray( foundKiwi[i] );
                      kiwiWidth = cntCascade.width;
                    }
                    distance = round(14*2714.3/kiwiWidth);
                    cv::rectangle(img, pt1, pt2, CV_RGB(255,0,0), 1);
                  }
                  else
                  {
                    kiwiExist = false;
                  }
                }

                ///////Added Code KIWI


                // Display image.
                if (VERBOSE) {
                    cv::imshow(sharedMemory->name().c_str(), img);
                    //cv::imshow("Processed", twoColor);//img);
                    /*if(foundKiwi.size() > 0)
                    {
                      cv::imshow("cascade", kiwiROI);
                    }*/
                    
                    cv::waitKey(1);
                }

                ////////////////////////////////////////////////////////////////
                // Do something with the distance readings if wanted.
                {
                    std::lock_guard<std::mutex> lck(distancesMutex);
                    /*std::cout << "front = " << front << ", "
                              << "rear = " << rear << ", "
                              << "left = " << left << ", "
                              << "right = " << right << "." << std::endl;*/
                }

                ////////////////////////////////////////////////////////////////
                // Example for creating and sending a message to other microservices; can
                // be removed when not needed.
                //opendlv::proxy::AngleReading ar;
                //ar.angle(123.45f);
                //od4.send(ar);

                ////////////////////////////////////////////////////////////////
                // Steering and acceleration/decelration.
                //
                // Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
                // Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
                //opendlv::proxy::GroundSteeringRequest gsr;
                //gsr.groundSteering(0);
                //od4.send(gsr);

                // Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
                // Be careful!
                opendlv::proxy::PedalPositionRequest ppr;
                if(kiwiExist)
                {
                  if(distance < lowerLimit)
                  {
                    pdPosition = 0.1;
                  }
                  else if(distance >= lowerLimit && distance < upperLimit)
                  {
                    pdPosition = 0.3;
                  }
                  else if(distance >= upperLimit)
                  {
                    pdPosition = 0.5;
                  }
                }
                else
                {
                  pdPosition = 0.7;
                }
                cluon::data::TimeStamp sampleTime2 = cluon::time::now();
                ppr.position(static_cast<float>(pdPosition));
                od4.send(ppr,sampleTime2,1);
                std::cout << "distance = " << distance << ", "
                              << "pedal = " << pdPosition << std::endl;
            }
        }
        retCode = 0;
    }
    //fs.close();
    return retCode;
}

