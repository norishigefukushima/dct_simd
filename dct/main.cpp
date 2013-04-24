#include <opencv2/opencv.hpp>
#pragma comment(lib, "opencv_core240.lib")
#pragma comment(lib, "opencv_highgui240.lib")

using namespace cv;
using namespace std;
#include<iostream>

//simd LLM
void iDCT8x8_32f(const float* s, float* d, float* temp);//inv
void fDCT8x8_32f(const float* s, float* d, float* temp);//fwd
//c++ LLM
void iDCT2Dllm_32f(const float* s, float* d, float* temp);//inv
void fDCT2Dllm_32f(const float* s, float* d, float* temp);//fwd

void fDCT_Test(Mat& src, int iter=100000,bool isShowCoeff=false)
{
	cout<<"fwd DCT\n";
	Mat dst(Size(8,8),CV_32F);
	Mat dst1(Size(8,8),CV_32F);
	Mat dst2(Size(8,8),CV_32F);

	Mat tmp(Size(8,8),CV_32F);//swap buffer

	int64 pre = getTickCount();
	for(int i=0;i<iter;i++)
		cv::dct(src,dst);
	cout<<"ocv: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	pre = getTickCount();
	for(int i=0;i<iter;i++)
		fDCT2Dllm_32f(src.ptr<float>(0),dst1.ptr<float>(0),tmp.ptr<float>(0));
	cout<<"llm: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	pre = getTickCount();
	for(int i=0;i<iter;i++)
		fDCT8x8_32f(src.ptr<float>(0),dst2.ptr<float>(0),tmp.ptr<float>(0));
	cout<<"llm sse: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	if(isShowCoeff)
	{
		cout<<"ocv\n"<<dst<<endl<<endl;
		cout<<"llm\n"<<dst1<<endl<<endl;
		cout<<"llm sse\n"<<dst2<<endl<<endl;
	}
}

void iDCT_Test(Mat& src, int iter=100000, bool isShowCoeff=false)
{
	cout<<"inv DCT\n";
	Mat dst(Size(8,8),CV_32F);
	Mat dst1(Size(8,8),CV_32F);
	Mat dst2(Size(8,8),CV_32F);

	Mat tmp(Size(8,8),CV_32F);//swap buffer

	int64 pre = getTickCount();
	for(int i=0;i<iter;i++)
		cv::dct(src,dst,DCT_INVERSE);
	cout<<"ocv: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	pre = getTickCount();
	for(int i=0;i<iter;i++)
	{
		//src and dest pointers require 16byte alignment, and continuous field.
		//These pointer should are derived from OpenCV Mat or __declspec(align(16)) definition or _aligned_malloc and so on.
		iDCT2Dllm_32f(src.ptr<float>(0),dst1.ptr<float>(0),tmp.ptr<float>(0));
	}
	cout<<"llm: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	pre = getTickCount();
	for(int i=0;i<iter;i++)
		iDCT8x8_32f(src.ptr<float>(0),dst2.ptr<float>(0),tmp.ptr<float>(0));
	cout<<"llm sse: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	if(isShowCoeff)
	{
		cout<<"ocv\n"<<dst<<endl<<endl;
		cout<<"llm\n"<<dst1<<endl<<endl;
		cout<<"llm sse\n"<<dst2<<endl<<endl;
	}
}

void DCT_Quant_Test(Mat& src, Mat& dest,float threshold)
{
	Mat dst(Size(8,8),CV_32F);
	Mat swp(Size(8,8),CV_32F);
	Mat tmp(Size(8,8),CV_32F);

	Mat mblock;
	Mat mask;
	for(int j=0;j<src.rows/8;j++)
	{
		for(int i=0;i<src.cols/8;i++)
		{
			//make macro block
			src(Rect(8*i,8*j,8,8)).copyTo(mblock);
			//fwd DCT
			fDCT8x8_32f((float*)mblock.data,(float*)dst.data,(float*)tmp.data);

			//cut off coefficient like JPEG
			//bottleneck processes...///
			absdiff(dst,0,tmp);
			compare(tmp,threshold,mask,CMP_GT);
			swp.setTo(0.f);
			dst.copyTo(swp,mask);
			///////////////
			
			//inv DCT
			iDCT8x8_32f((float*)swp.data,(float*)dst.data,(float*)tmp.data);

			//copy micro block into dst image
			dst.copyTo(dest(Rect(8*i,8*j,8,8)));
		}
	}
}

int main()
{
	//src image: 32f bit gray for our function
	Mat b = imread("haze1.jpg",0);
	Mat a;b.convertTo(a,CV_32F);


	//make 8x8 macro block
	Mat src;
	a(Rect(8*24,8*16,8,8)).copyTo(src);

	//fwd DCT x 100000 interation, without showing coefficient
	fDCT_Test(src,100000,false);

	Mat swp;
	//fwd DCT for inv DCT
	cv::dct(src,swp);
	swp.copyTo(src);

	//inv DCT x 100000 interation, without showing coefficient
	iDCT_Test(src,100000,false);

	waitKey(1000);
	Mat aq(a.size(),CV_32F);
	for(int i=0;i<1000;i+=1)
	{
		int64 pre = getTickCount();
		DCT_Quant_Test(a,aq,i);
		
		cout<<"dct q: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

		Mat show;
		aq.convertTo(show,CV_8U);
		imshow("DCT:Q",show);
		waitKey(10);
	}
}