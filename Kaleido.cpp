#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;


struct XYZColors {
  float X;
  float Y;
  float Z;
} XYZ_color;

struct YXYColors {
  float Y;
  float x;
  float y;
} Yxy_color;

int value = 0;
float delta = .20;
int frame_rate = 120;
int BLOCKSIZE = 16;
int previousBlock = 0;

YXYColors convertToYxy(XYZColors c);

String getNextName();

bool isInRGBRange(float x, float y);

float calculateNewXPoint(float dist, float angle, float x);

float calculateNewYPoint(float dist, float angle, float y);

float getDistanceBetweenPoints(float x1, float y1,float x2,float y2 );

float getAngleInRadians (float x1, float y1,float x2,float y2 );

float getRandomNumberInRange();

void runKaleidoRandom();

void createVideo();

void runKaleidoBlocks();

void runKaleidoMixedAndSmoothed();

float convertBackX(float Y, float x, float y );
float convertBackY(float Y );
float convertBackZ(float Y, float x, float y );

bool isEvenBlock(int y, bool isEven);

int main(){

	//runKaleidoBlocks();
	//runKaleidoRandom();
	runKaleidoMixedAndSmoothed();

	return (0);
}

bool isEvenBlock(int y, bool isEven){

	if(y > previousBlock){
		if (isEven){
			isEven = false;
		}else{
			isEven = true;
		}
	}

	previousBlock = y;

	return isEven;
}

float convertBackX(float Y, float x, float y ){

	return x * ( Y*255 / y);

}

float convertBackY(float Y ){

	return Y*255;
}

float convertBackZ(float Y, float x, float y ){

	return ( 1 - x - y ) * ( Y*255 / y );
}

String getNextName(){

	std::stringstream ss;
	ss << "images/"<< std::setw(5) << std::setfill('0') << value <<".png";
	std::string str = ss.str();
	//std::cout <<"Writing: "<< str<<endl;

	value++;


	return str;
}

float getDistanceBetweenPoints(float x1, float y1,float x2,float y2 ){

	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

float getAngleInRadians (float x1, float y1,float x2,float y2 ){

	return atan2(y1 - y2, x1 - y2);
}

bool isInRGBRange(float x, float y){

	bool range = false;

	//Green Point
	float greenX = 0.28;
	float greenY = 0.595;

	//Blue Point
	float blueX = 0.155;
	float blueY = 0.07;

	//Red Point
	float redX = 0.625;
	float redY = 0.34;


	float alpha = ((blueY - redY)*(x - redX) + (redX - blueX)*(y - redY)) /((blueY - redY)*(greenX - redX) + (redX - blueX)*(greenY - redY));
	float beta = ((redY - greenY)*(x - redX) + (greenX - redX)*(y - redY)) /((blueY - redY)*(blueX - redX) + (redX - blueX)*(greenY - redY));
	float gamma = 1.0f - alpha - beta;


	if (alpha > 0 && beta > 0 && gamma > 0){

		range = true;
	}

	return range;
}

float getRandomNumberInRange(){
	return ((float) rand()) / (float) RAND_MAX;
}

YXYColors convertToYxy(XYZColors c){

	YXYColors newColor;

	newColor.Y = c.X /255;
	newColor.x = c.X / ( c.X + c.Y + c.Z );
	newColor.y = c.Y / ( c.X + c.Y + c.Z );

	return newColor;

}

float calculateNewXPoint(float dist, float angle, float x){

	return (x + dist * cos(angle));

}

float calculateNewYPoint(float dist, float angle, float y){

	return  (y + dist * sin(angle));

}

void createVideo(){

	// Write to Video when finished reassembling four sub-frames
	cv::VideoCapture in_capture("images/%05d.png");

	Mat img;

	cout<< "video written @"<<frame_rate<<" frames/sec"<<endl;
	VideoWriter out_capture("videos/video.avi", CV_FOURCC('M','J','P','G'), frame_rate, Size(512,512));

	while (true){

		in_capture >> img;
		if(img.empty()){break;}

		out_capture.write(img);
	}

}

void runKaleidoBlocks(){

	cout<<"runKaleidoBlocks() running"<<endl;

	// open the video file for reading
	VideoCapture cap("out.mp4");
	//VideoCapture cap("slides.mov");

	if ( !cap.isOpened() )  // if not success, exit program
	{
		 cout << "Cannot open the video file" << endl;

	}else{
		cout<<"Video File Opened."<<endl;
	}

	//used to keep track of the number of frames
	int frame_number = 0;

	while(1)
	{

		/*
		 *  BGR is the original frame which will converted to Lab and split into 4 sub-frames
		 *  (two fusion pairs)
		 */
		Mat BGR;

		// read the next frame from video
		bool bSuccess = cap.read(BGR);

		//if not success or end of frames, break loop
		if (!bSuccess )
		{
			createVideo();
			break;
		}

		resize(BGR, BGR, Size(512, 512), 0, 0, INTER_CUBIC);

		//increase the number of frames to show how many original frames that have been processed
		frame_number++;

		/*
		 * CIE XYZ is a matrix with format (x,z) Y Luminance
		 */
		Mat XYZ;

		//convert BGR to XYZ
		cvtColor(BGR,XYZ, COLOR_BGR2XYZ);

		//create fusion pair for random color pollution
		Mat block_fusion_pair_1 = Mat::zeros( XYZ.size(), XYZ.type() );
		Mat block_fusion_pair_2 = Mat::zeros( XYZ.size(), XYZ.type() );

		Mat pair_1 = Mat::zeros( XYZ.size(), XYZ.type() );
		Mat pair_2 = Mat::zeros( XYZ.size(), XYZ.type() );

		bool isEven = true;

		// check if divisors fit to image dimensions
		if(XYZ.cols % BLOCKSIZE == 0 && XYZ.rows % BLOCKSIZE == 0)
		{

			for(int y = 0; y < XYZ.cols; y += XYZ.cols / BLOCKSIZE)
			{
				for(int x = 0; x < XYZ.rows; x += XYZ.rows / BLOCKSIZE)
				{

					//creating the block
					cv::Rect rect = cv::Rect(y, x, (XYZ.cols / BLOCKSIZE), (XYZ.rows / BLOCKSIZE));

					isEven = isEvenBlock(y, isEven);

					if (isEven){

						isEven = false;
						// Change ever other block for the two fusion pair
						for (int i = 0; i < XYZ(rect).rows; i++){
							for( int j = 0; j < XYZ(rect).cols; j++){

									 float new_delta;

									 Vec3b pixel = XYZ(rect).at<Vec3b>(i,j);

									 delta = .10;

									//store color in a struct
									XYZ_color.X = (float)pixel[0];
									XYZ_color.Y = (float)pixel[1];
									XYZ_color.Z = (float)pixel[2];

									// convert XYZ values to the Yxy values needed.
									YXYColors Yxy = convertToYxy(XYZ_color);



									if (Yxy.Y + delta > 1 || Yxy.Y - delta < 0){

										 if(Yxy.Y + delta > 1){

											 new_delta = 1 - Yxy.Y;
										 }else{

											 new_delta = Yxy.Y - 0;
										 }

									 }else{

									 new_delta = delta;
									 }



									 //cout<<"delta: "<<new_delta<<" y: "<<Yxy.Y <<endl;

									 // return two new coordinates to xyz format and add to Mat
									block_fusion_pair_1.at<Vec3b>(i+x,j+y)[0] = convertBackX(Yxy.Y+ new_delta, Yxy.x, Yxy.y );
									block_fusion_pair_1.at<Vec3b>(i+x,j+y)[1] = convertBackY(Yxy.Y + new_delta);
									block_fusion_pair_1.at<Vec3b>(i+x,j+y)[2] =  convertBackZ(Yxy.Y + new_delta, Yxy.x, Yxy.y );

									 // return two new coordinates to xyz format and add to Mat
									block_fusion_pair_2.at<Vec3b>(i+x,j+y)[0] = convertBackX(Yxy.Y- new_delta, Yxy.x, Yxy.y );
									block_fusion_pair_2.at<Vec3b>(i+x,j+y)[1] = convertBackY(Yxy.Y - new_delta);
									block_fusion_pair_2.at<Vec3b>(i+x,j+y)[2] =  convertBackZ(Yxy.Y- new_delta, Yxy.x, Yxy.y );



								 }//end for j

							}//end for i

					}else{// odd blocks

						isEven = true;


						// Leave every other block as is
						for (int i = 0; i < XYZ(rect).rows; i++){
							for( int j = 0; j < XYZ(rect).cols; j++){
								for( int c = 0; c < 3; c++ ){

									 //Here we can alternate the _+ delta, right now its left as is
									block_fusion_pair_1.at<Vec3b>(x+j,y+i)[c] =  (int)XYZ(rect).at<Vec3b>(j,i)[c] ;
									block_fusion_pair_2.at<Vec3b>(x+j,y+i)[c] =  (int)XYZ(rect).at<Vec3b>(j,i)[c] ;

									//cout<<"odd: "<<(int)XYZ(rect).at<Vec3b>(j,i)[c]<<" c: "<<c<<endl;
								}
							}
						}



					}//end else

				}//end for j
			}//end for i
		}else{
			cout << "Error: Please use another divisor for the block split." << endl;
			exit(1);
		}


		//conversion color pollution pair to BGR
		cvtColor(block_fusion_pair_1,block_fusion_pair_1,COLOR_XYZ2BGR);
		cvtColor(block_fusion_pair_2,block_fusion_pair_2,COLOR_XYZ2BGR);


		//BILATERAL FILTER	used to smooth blocks
		bilateralFilter ( block_fusion_pair_1, pair_1, 15, 80, 80 );
		bilateralFilter ( block_fusion_pair_2, pair_2, 15, 80, 80 );

		//Write images to file to be converted to video
		cv::imwrite(getNextName(),pair_2);
		cv::imwrite(getNextName(),pair_1);
		cv::imwrite(getNextName(),pair_2);
		cv::imwrite(getNextName(),pair_1);



	}//end while
}

void runKaleidoRandom(){

		cout<<"runKaleidoRandom() running"<<endl;

		// open the video file for reading
		VideoCapture cap("out.mp4");
		//VideoCapture cap("slides.mov");

		if ( !cap.isOpened() )  // if not success, exit program
		{
			 cout << "Cannot open the video file" << endl;

		}else{
			cout<<"Video File Opened."<<endl;
		}

		//used to keep track of the number of frames
		int frame_number = 0;

		while(1)
		{

			/*
			 *  BGR is the original frame which will converted to Lab and split into 4 sub-frames
			 *  (two fusion pairs)
			 */
			Mat BGR;

			// read the next frame from video
			bool bSuccess = cap.read(BGR);



			//if not success or end of frames, break loop
			if (!bSuccess )
			{
				createVideo();
				break;
			}

			resize(BGR, BGR, Size(512, 512), 0, 0, INTER_CUBIC);

			//increase the number of frames to show how many original frames that have been processed
			frame_number++;

			/*
			 * CIE XYZ is a matrix with format (x,z) Y Luminance
			 */
			Mat XYZ;

			//convert BGR to XYZ
			cvtColor(BGR,XYZ, COLOR_BGR2XYZ);

			//create fusion pair for random color pollution
			Mat color_fusion_pair_1 = Mat::zeros( XYZ.size(), XYZ.type() );
			Mat color_fusion_pair_2 = Mat::zeros( XYZ.size(), XYZ.type() );


			for(int i = 0; i < XYZ.cols; i++)
			{
				for(int j = 0; j < XYZ.rows; j++)
				{

					bool in_RGB_range = false;

					Vec3b pixel = XYZ.at<Vec3b>(i,j);
					Vec3b pixels = BGR.at<Vec3b>(i,j);
					delta = .10;

					//store color in a struct
					XYZ_color.X = (float)pixel[0];
					XYZ_color.Y = (float)pixel[1];
					XYZ_color.Z = (float)pixel[2];

					// convert XYZ values to the Yxy values needed.
					YXYColors Yxy = convertToYxy(XYZ_color);



					//Problem with Black and RGB
					if (isnan(Yxy.x)==true || Yxy.y ==1 ||Yxy.y == 0 ){
						//cout<< "Yxy: "<<Yxy.Y<< " "<<Yxy.x<<" "<<Yxy.y<<endl;
						//break;
						//store color in a struct
						XYZ_color.X = 139.0;
						XYZ_color.Y = 137.0;
						XYZ_color.Z = 137.0;

						// convert XYZ values to the Yxy values needed.
						Yxy = convertToYxy(XYZ_color);

					}

					while(in_RGB_range == false ){

						//random number between 1 and 0 for random color
						float random_pointX = getRandomNumberInRange();
						float random_pointY = getRandomNumberInRange();

						// find out if random color is within the RGB range
						if (isInRGBRange(random_pointX, random_pointY)){in_RGB_range = true;}

						//get the distance between points
						float distance = getDistanceBetweenPoints(random_pointX, random_pointY, Yxy.x, Yxy.y);

						//get the angle between points
						float radians = getAngleInRadians(random_pointX, random_pointY,  Yxy.x, Yxy.y);

						//calculate new point based on distance and angle
						float newX = calculateNewXPoint(distance, radians, Yxy.x);
						float newY = calculateNewYPoint(distance, radians, Yxy.y);

						if (isInRGBRange(newX, newY) == true){

							// return two new coordinates to xyz format and add to Mat
							color_fusion_pair_1.at<Vec3b>(i,j)[0] = convertBackX(Yxy.Y, random_pointX, random_pointY );
							color_fusion_pair_1.at<Vec3b>(i,j)[1] = convertBackY(Yxy.Y );
							color_fusion_pair_1.at<Vec3b>(i,j)[2] =  convertBackZ(Yxy.Y, random_pointX, random_pointY );

							color_fusion_pair_2.at<Vec3b>(i,j)[0] = convertBackX(Yxy.Y, newX, newY );
							color_fusion_pair_2.at<Vec3b>(i,j)[1] = convertBackY(Yxy.Y );
							color_fusion_pair_2.at<Vec3b>(i,j)[2] = convertBackZ(Yxy.Y, newX, newY );

							in_RGB_range = true;

						}else {
							in_RGB_range = false;

						}
					}
				}
			}

			//conversion color pollution pair to BGR
			cvtColor(color_fusion_pair_1,color_fusion_pair_1,COLOR_XYZ2BGR);
			cvtColor(color_fusion_pair_2,color_fusion_pair_2,COLOR_XYZ2BGR);


			//Write images to file to be converted to video

			cv::imwrite(getNextName(),color_fusion_pair_2);
			cv::imwrite(getNextName(),color_fusion_pair_1);
			cv::imwrite(getNextName(),color_fusion_pair_2);
			cv::imwrite(getNextName(),color_fusion_pair_1);

	}
}

void runKaleidoMixedAndSmoothed(){

	cout<<"runKaleidoMixedAndSmoothed() running"<<endl;

			// open the video file for reading
			VideoCapture cap("out.mp4");
			//VideoCapture cap("slides.mov");

			if ( !cap.isOpened() )  // if not success, exit program
			{
				 cout << "Cannot open the video file" << endl;

			}else{
				cout<<"Video File Opened."<<endl;
			}

			//used to keep track of the number of frames
			int frame_number = 0;

			while(1)
			{

				/*
				 *  BGR is the original frame which will converted to Lab and split into 4 sub-frames
				 *  (two fusion pairs)
				 */
				Mat BGR;

				// read the next frame from video
				bool bSuccess = cap.read(BGR);



				//if not success or end of frames, break loop
				if (!bSuccess )
				{
					createVideo();
					break;
				}

				resize(BGR, BGR, Size(512, 512), 0, 0, INTER_CUBIC);

				//increase the number of frames to show how many original frames that have been processed
				frame_number++;

				/*
				 * CIE XYZ is a matrix with format (x,z) Y Luminance
				 */
				Mat XYZ;

				//convert BGR to XYZ
				cvtColor(BGR,XYZ, COLOR_BGR2XYZ);

				//create fusion pair for random color pollution
				Mat color_fusion_pair_1 = Mat::zeros( XYZ.size(), XYZ.type() );
				Mat color_fusion_pair_2 = Mat::zeros( XYZ.size(), XYZ.type() );


				for(int i = 0; i < XYZ.cols; i++)
				{
					for(int j = 0; j < XYZ.rows; j++)
					{

						bool in_RGB_range = false;

						Vec3b pixel = XYZ.at<Vec3b>(i,j);
						Vec3b pixels = BGR.at<Vec3b>(i,j);
						delta = .10;

						//store color in a struct
						XYZ_color.X = (float)pixel[0];
						XYZ_color.Y = (float)pixel[1];
						XYZ_color.Z = (float)pixel[2];

						// convert XYZ values to the Yxy values needed.
						YXYColors Yxy = convertToYxy(XYZ_color);



						//Problem with Black and RGB
						if (isnan(Yxy.x)==true || Yxy.y ==1 ||Yxy.y == 0 ){
							//cout<< "Yxy: "<<Yxy.Y<< " "<<Yxy.x<<" "<<Yxy.y<<endl;
							//break;
							//store color in a struct
							XYZ_color.X = 139.0;
							XYZ_color.Y = 137.0;
							XYZ_color.Z = 137.0;

							// convert XYZ values to the Yxy values needed.
							Yxy = convertToYxy(XYZ_color);

						}

						while(in_RGB_range == false ){

							//random number between 1 and 0 for random color
							float random_pointX = getRandomNumberInRange();
							float random_pointY = getRandomNumberInRange();

							// find out if random color is within the RGB range
							if (isInRGBRange(random_pointX, random_pointY)){in_RGB_range = true;}

							//get the distance between points
							float distance = getDistanceBetweenPoints(random_pointX, random_pointY, Yxy.x, Yxy.y);

							//get the angle between points
							float radians = getAngleInRadians(random_pointX, random_pointY,  Yxy.x, Yxy.y);

							//calculate new point based on distance and angle
							float newX = calculateNewXPoint(distance, radians, Yxy.x);
							float newY = calculateNewYPoint(distance, radians, Yxy.y);

							if (isInRGBRange(newX, newY) == true){

								// return two new coordinates to xyz format and add to Mat
								color_fusion_pair_1.at<Vec3b>(i,j)[0] = convertBackX(Yxy.Y, random_pointX, random_pointY );
								color_fusion_pair_1.at<Vec3b>(i,j)[1] = convertBackY(Yxy.Y );
								color_fusion_pair_1.at<Vec3b>(i,j)[2] =  convertBackZ(Yxy.Y, random_pointX, random_pointY );

								color_fusion_pair_2.at<Vec3b>(i,j)[0] = convertBackX(Yxy.Y, newX, newY );
								color_fusion_pair_2.at<Vec3b>(i,j)[1] = convertBackY(Yxy.Y );
								color_fusion_pair_2.at<Vec3b>(i,j)[2] = convertBackZ(Yxy.Y, newX, newY );

								in_RGB_range = true;

							}else {
								in_RGB_range = false;

							}
						}
					}
				}

				//conversion color pollution pair to BGR
				cvtColor(color_fusion_pair_1,color_fusion_pair_1,COLOR_XYZ2BGR);
				cvtColor(color_fusion_pair_2,color_fusion_pair_2,COLOR_XYZ2BGR);

				medianBlur(color_fusion_pair_1, color_fusion_pair_1, 5);
				medianBlur(color_fusion_pair_2, color_fusion_pair_2, 5);


				//Write images to file to be converted to video

				cv::imwrite(getNextName(),color_fusion_pair_2);
				cv::imwrite(getNextName(),color_fusion_pair_1);
				cv::imwrite(getNextName(),color_fusion_pair_2);
				cv::imwrite(getNextName(),color_fusion_pair_1);

		}



}
