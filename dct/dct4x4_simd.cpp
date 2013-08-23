//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.

#include <nmmintrin.h> //SSE4.2
#define  _USE_MATH_DEFINES
#include <math.h>

void transpose4x4(float* src, float* dest);
void transpose4x4(float* src);

#include <opencv2/opencv.hpp>
using namespace cv;

void dct4x4_1d_bf(Mat& a, Mat& b, int flag=0)
{
	const float c2 = 1.30656f;//cos(CV_PI*2/16.0)*sqrt(2);
	const float c6 = 0.541196;//cos(CV_PI*6/16.0)*sqrt(2);
	Mat dctf = Mat::ones(4,4,CV_32F);
	dctf.at<float>(1,0)=c2;	
	dctf.at<float>(1,1)=c6;	
	dctf.at<float>(1,2)=-c6;	
	dctf.at<float>(1,3)=-c2;	
	
	dctf.at<float>(2,1)= -1.f;	
	dctf.at<float>(2,2)= -1.f;	
	
	dctf.at<float>(3,0)= c6;	
	dctf.at<float>(3,1)=-c2;	
	dctf.at<float>(3,2)= c2;	
	dctf.at<float>(3,3)=-c6;

	if(flag == 0)
		b = 0.5*dctf*a;
	else
		b = 0.5*dctf.t()*a;
}

void dct4x4_bf(Mat& a, Mat& b, int flag=0)
{
	Mat temp;
	dct4x4_1d_bf(a,temp,flag);
	transpose4x4(temp.ptr<float>(0));
	dct4x4_1d_bf(temp,b);
	transpose4x4(b.ptr<float>(0));
}

void dct4x4_1d_llm_fwd(Mat& a, Mat& b)
{
	const float c2 = 1.30656f;//cos(CV_PI*2/16.0)*sqrt(2);
	const float c6 = 0.541196;//cos(CV_PI*6/16.0)*sqrt(2);

	float* s = a.ptr<float>(0);	
	float* d = b.ptr<float>(0);	
	for(int i=0; i<a.rows; i++,s+=4,d+=4)
	{
		float p03 = s[0] + s[3];
		float p12 = s[1] + s[2];
		float m03 = s[0] - s[3];
		float m12 = s[1] - s[2];

		d[0]=p03+p12;
		d[1]=c2*m03+c6*m12;
		d[2]=p03-p12;
		d[3]=c6*m03-c2*m12;
	}
	//b*=0.5;
}

void dct4x4_1d_llm_inv(Mat& a, Mat& b)
{
	const float c2 = 1.30656f;//cos(CV_PI*2/16.0)*sqrt(2);
	const float c6 = 0.541196;//cos(CV_PI*6/16.0)*sqrt(2);
	
	float* s = a.ptr<float>(0);	
	float* d = b.ptr<float>(0);	
	for(int i=0; i<a.rows; i++,s+=4,d+=4)
	{
		float t10 = s[0] + s[2];
		float t12 = s[0] - s[2];
		float t0 = c2*s[1] + c6*s[3];
		float t2 = c6*s[1] - c2*s[3];

		d[0]=t10+t0;
		d[1]=t12+t2;
		d[2]=t12-t2;
		d[3]=t10-t0;
	}
	//b*=0.5;
}

void dct4x4_llm(Mat& a, Mat& b, Mat& temp ,int flag)
{
	if(flag==0)
	{
		dct4x4_1d_llm_fwd(a,temp);
		transpose4x4(temp.ptr<float>(0));
		dct4x4_1d_llm_fwd(temp,b);
		transpose4x4(b.ptr<float>(0));
		b*=0.25;
	}
	else 
	{
		dct4x4_1d_llm_inv(a,temp);
		transpose4x4(temp.ptr<float>(0));
		dct4x4_1d_llm_inv(temp,b);
		transpose4x4(b.ptr<float>(0));
		b*=0.25;
	}
}