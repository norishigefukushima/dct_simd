#include <opencv2/opencv.hpp>

//for opencv2.4
//#pragma comment(lib, "opencv_core240.lib")
//#pragma comment(lib, "opencv_highgui240.lib")

//for opencv2.45 (may faster than 2.4 )
//#pragma comment(lib, "opencv_core245.lib")
//#pragma comment(lib, "opencv_highgui245.lib")

//for opencv2.46 
#pragma comment(lib, "opencv_core246.lib")
#pragma comment(lib, "opencv_highgui246.lib")

using namespace cv;
using namespace std;
#include<iostream>

//simd LLM
void iDCT8x8_llm_sse(const float* s, float* d, float* temp);//inv
void fDCT8x8_llm_sse(const float* s, float* d, float* temp);//fwd

//c++ LLM
void iDCT2D_llm(const float* s, float* d, float* temp);//inv
void fDCT2D_llm(const float* s, float* d, float* temp);//fwd

void dct4x4_llm_sse(float* a, float* b, float* temp ,int flag=0);//LLM SSE implimentation
void dct4x4_llm(float* a, float* b, float* temp ,int flag=0);//LLM C++ implimentation
void dct4x4_bf(Mat& a, Mat& b, int flag=0);//matmul: brute force implimentation


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
		fDCT2D_llm(src.ptr<float>(0),dst1.ptr<float>(0),tmp.ptr<float>(0));
	cout<<"llm: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	pre = getTickCount();
	for(int i=0;i<iter;i++)
		fDCT8x8_llm_sse(src.ptr<float>(0),dst2.ptr<float>(0),tmp.ptr<float>(0));
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
		iDCT2D_llm(src.ptr<float>(0),dst1.ptr<float>(0),tmp.ptr<float>(0));
	}
	cout<<"llm: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

	pre = getTickCount();
	for(int i=0;i<iter;i++)
		iDCT8x8_llm_sse(src.ptr<float>(0),dst2.ptr<float>(0),tmp.ptr<float>(0));
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
			fDCT8x8_llm_sse((float*)mblock.data,(float*)dst.data,(float*)tmp.data);

			//cut off coefficient like JPEG
			//bottleneck processes...///
			absdiff(dst,0,tmp);
			compare(tmp,threshold,mask,CMP_GT);
			swp.setTo(0.f);
			dst.copyTo(swp,mask);
			///////////////
			
			//inv DCT
			iDCT8x8_llm_sse((float*)swp.data,(float*)dst.data,(float*)tmp.data);

			//copy micro block into dst image
			dst.copyTo(dest(Rect(8*i,8*j,8,8)));
		}
	}
}


void fDCT4x4Test(Mat& a, int iter = 100000, bool isShowMat=false)
{
	Mat b;
	{
		int64 pre = getTickCount();
		for(int i=0;i<iter;i++)
			cv::dct(a,b);//DFT based implimentation
		cout<<"opencv:"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;


	b.setTo(0);
	{
		int64 pre = getTickCount();
		for(int i=0;i<iter;i++)
			dct4x4_bf(a,b);
		cout<<"BF:"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;

	b.setTo(0);
	Mat temp = Mat::zeros(4,4,CV_32F);
	//dest Mat should be allocated 
	{
		int64 pre = getTickCount();
		float* s = a.ptr<float>(0);
		float* d = b.ptr<float>(0);
		float* t = temp.ptr<float>(0);
		for(int i=0;i<iter;i++)
			dct4x4_llm(s,d,t);
		cout<<"LLM(c++):"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;

	b.setTo(0);
	{
		int64 pre = getTickCount();
		float* s = a.ptr<float>(0);
		float* d = b.ptr<float>(0);
		float* t = temp.ptr<float>(0);
		for(int i=0;i<iter;i++)
			dct4x4_llm_sse(s,d,t);
		cout<<"LLM(SSE):"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;
//	cout<<d*0.5<<endl;
}

void iDCT4x4Test(Mat& a, int iter = 100000, bool isShowMat=false)
{
	Mat b;
	{
		int64 pre = getTickCount();
		for(int i=0;i<iter;i++)
			cv::dct(a,b,DCT_INVERSE);//DFT based implimentation
		cout<<"opencv:"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;

	b.setTo(0);
	{
		int64 pre = getTickCount();
		for(int i=0;i<iter;i++)
			dct4x4_bf(a,b,DCT_INVERSE);
		cout<<"BF:"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;

	b.setTo(0);
	Mat temp = Mat::zeros(4,4,CV_32F);
	//dest Mat should be allocated 
	{
		int64 pre = getTickCount();
		float* s = a.ptr<float>(0);
		float* d = b.ptr<float>(0);
		float* t = temp.ptr<float>(0);
		for(int i=0;i<iter;i++)
			dct4x4_llm(s,d,t,DCT_INVERSE);
		cout<<"LLM(c++):"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;

	b.setTo(0);
	//dest Mat should be allocated 
	{
		int64 pre = getTickCount();
		float* s = a.ptr<float>(0);
		float* d = b.ptr<float>(0);
		float* t = temp.ptr<float>(0);
		for(int i=0;i<iter;i++)
			dct4x4_llm_sse(s,d,t,DCT_INVERSE);
		cout<<"LLM(SSE):"<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;
	}
	if(isShowMat) cout<<b<<endl;
//	cout<<d*0.5<<endl;
}

int main()
{
	//const bool isShow = true;
	const bool isShow = false;
	Mat s44 = Mat::zeros(4,4,CV_32F);
	s44.at<float>(0,0)=1.f;
	s44.at<float>(1,0)=1.f;
	s44.at<float>(2,0)=1.f;

	const int iter=100000;
	cout<<"fwd 4x4:"<<endl;
	fDCT4x4Test(s44,iter,isShow); cout<<endl;
	
	cout<<"inv 4x4:"<<endl;
	iDCT4x4Test(s44,iter,isShow); cout<<endl;
	

	//src image: 32f bit gray for our function
	Mat b = imread("haze1.jpg",0);
	Mat a;b.convertTo(a,CV_32F);

	//make 8x8 macro block
	Mat src;
	a(Rect(8*24,8*16,8,8)).copyTo(src);

	//fwd DCT x 100000 iteration, without showing coefficient
	fDCT_Test(src,iter,isShow);
	
	Mat swp;
	//fwd DCT for inv DCT
	cv::dct(src,swp);
	swp.copyTo(src);

	//inv DCT x 100000 iteration, without showing coefficient
	iDCT_Test(src,iter,isShow);

	return 0;

	waitKey(1000);
	Mat aq(a.size(),CV_32F);
	for(int i=0;i<300;i+=3)
	{
		int64 pre = getTickCount();
		DCT_Quant_Test(a,aq,i);
		
		cout<<"dct q: "<<1000.0*(getTickCount()-pre)/(getTickFrequency())<<" ms"<<endl;

		Mat show;
		aq.convertTo(show,CV_8U);
		imshow("DCT:Q",show);
		waitKey(1);
	}
}