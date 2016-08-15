//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.


//paper TGLXLZW12
//Tong, Kai, Gu, Ying Ke, Li, Guo Lin, Xie, Xiang, Liu, Shou Hao, Zhao, Kai,Wang, Zhi Hua
//"A fast algorithm of 4-point floating DCT in image/video compression"
//Proc. International Conference on Audio, Language and Image Processing, 2012 .


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

void dct4x4_1d_tglxlzw_fwd(float* s, float* d)
{
	const float c2 = 1.30656f;//cos(CV_PI*2/16.0)*sqrt(2); printf("%f\n",sin(CV_PI*3/8.0)*sqrt(2));
	const float c6 = 0.541196;//cos(CV_PI*6/16.0)*sqrt(2); printf("%f\n",cos(CV_PI*3/8.0)*sqrt(2));
	
	for(int i=0; i<4; i++,s+=4,d+=4)
	{
		const float p03 = s[0] + s[3];
		const float p12 = s[1] + s[2];
		const float m12 = s[1] - s[2];

		const float i0 = s[0] - s[3]+m12;
		const float i1 = 1.414214f*m12;

		d[0]=p03+p12;
		d[1]=c2*(i1-i0);
		d[2]=p03-p12;
		d[3]=c6*(i0+i1);
	}
}

void dct4x4_1d_tglxlzw_fwd_sse(float* s, float* d)
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);
	const __m128 sq = _mm_set1_ps(1.414214f);//cos(CV_PI*6/16.0)*sqrt(2);
	
	__m128 s0 = _mm_load_ps(s);s+=4;
	__m128 s1 = _mm_load_ps(s);s+=4;
	__m128 s2 = _mm_load_ps(s);s+=4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0,s3);
	__m128 p12 = _mm_add_ps(s1,s2);
	__m128 m12 = _mm_sub_ps(s1,s2);

	__m128 i0 = _mm_add_ps(_mm_sub_ps(s0,s3),m12);	
	__m128 i1 = _mm_mul_ps(sq,m12);	

	_mm_store_ps(d, _mm_add_ps(p03,p12));
	_mm_store_ps(d+4,_mm_mul_ps(_mm_sub_ps(i1,i0),c2));
	_mm_store_ps(d+8, _mm_sub_ps(p03,p12));
	_mm_store_ps(d+12,_mm_mul_ps(_mm_add_ps(i1,i0),c6));
}
void dct4x4_tglxlzw(float* a, float* b, float* temp ,int flag)//9add, 3 mul
{
	if(flag==0)
	{
		dct4x4_1d_tglxlzw_fwd(a,temp);
		transpose4x4(temp);
		dct4x4_1d_tglxlzw_fwd(temp,b);
		transpose4x4(b);
		for(int i=0;i<16;i++)b[i]*=0.250f;
	}
	else 
	{
		//printf("not support\n");
		/*dct4x4_1d_llm_inv(a,temp);
		transpose4x4(temp);
		dct4x4_1d_llm_inv(temp,b);
		transpose4x4(b);
		for(int i=0;i<16;i++)b[i]*=0.250f;*/
	}
}

void dct4x4_tglxlzw_sse(float* a, float* b, float* temp ,int flag)//9add, 3 mul
{
	if(flag==0)
	{
		dct4x4_1d_tglxlzw_fwd_sse(a,temp);
		transpose4x4(temp);
		dct4x4_1d_tglxlzw_fwd_sse(temp,b);
		transpose4x4(b);
		__m128 c=_mm_set1_ps(0.250f);
		_mm_store_ps(b,_mm_mul_ps(_mm_load_ps(b),c));
		_mm_store_ps(b+4,_mm_mul_ps(_mm_load_ps(b+4),c));
		_mm_store_ps(b+8,_mm_mul_ps(_mm_load_ps(b+8),c));
		_mm_store_ps(b+12,_mm_mul_ps(_mm_load_ps(b+12),c));
	}
	else 
	{
		//printf("not support\n");
		/*dct4x4_1d_llm_inv(a,temp);
		transpose4x4(temp);
		dct4x4_1d_llm_inv(temp,b);
		transpose4x4(b);
		for(int i=0;i<16;i++)b[i]*=0.250f;*/
	}
}
void dct4x4_1d_llm_fwd(float* s, float* d)//8add, 4 mul
{
	const float c2 = 1.30656f;//cos(CV_PI*2/16.0)*sqrt(2);
	const float c6 = 0.541196;//cos(CV_PI*6/16.0)*sqrt(2);

	for(int i=0; i<4; i++,s+=4,d+=4)
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

void dct4x4_1d_llm_fwd_sse(float* s, float* d)//8add, 4 mul
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196);//cos(CV_PI*6/16.0)*sqrt(2);
	
	__m128 s0 = _mm_load_ps(s);s+=4;
	__m128 s1 = _mm_load_ps(s);s+=4;
	__m128 s2 = _mm_load_ps(s);s+=4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0,s3);
	__m128 p12 = _mm_add_ps(s1,s2);
	__m128 m03 = _mm_sub_ps(s0,s3);
	__m128 m12 = _mm_sub_ps(s1,s2);
		
	_mm_store_ps(d, _mm_add_ps(p03,p12));
	_mm_store_ps(d+4,_mm_add_ps(_mm_mul_ps(c2,m03), _mm_mul_ps(c6,m12)));
	_mm_store_ps(d+8, _mm_sub_ps(p03,p12));
	_mm_store_ps(d+12, _mm_sub_ps(_mm_mul_ps(c6,m03), _mm_mul_ps(c2,m12)));	
}

void dct4x4_1d_llm_inv(float* s, float* d)//8add, 4 mul
{
	const float c2 = 1.30656f;//cos(CV_PI*2/16.0)*sqrt(2);
	const float c6 = 0.541196;//cos(CV_PI*6/16.0)*sqrt(2);
	
	for(int i=0; i<4; i++,s+=4,d+=4)
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

void dct4x4_1d_llm_inv_sse(float* s, float* d)
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s);s+=4;
	__m128 s1 = _mm_load_ps(s);s+=4;
	__m128 s2 = _mm_load_ps(s);s+=4;
	__m128 s3 = _mm_load_ps(s);

	__m128 t10 = _mm_add_ps(s0,s2);
	__m128 t12 = _mm_sub_ps(s0,s2);

	__m128 t0 = _mm_add_ps(_mm_mul_ps(c2,s1),_mm_mul_ps(c6,s3));
	__m128 t2 = _mm_sub_ps(_mm_mul_ps(c6,s1),_mm_mul_ps(c2,s3));

	_mm_store_ps(d   ,_mm_add_ps(t10,t0));
	_mm_store_ps(d+4 ,_mm_add_ps(t12,t2));
	_mm_store_ps(d+8 ,_mm_sub_ps(t12,t2));
	_mm_store_ps(d+12,_mm_sub_ps(t10,t0));
}

void dct4x4_llm(float* a, float* b, float* temp ,int flag)
{
	if(flag==0)
	{
		dct4x4_1d_llm_fwd(a,temp);
		transpose4x4(temp);
		dct4x4_1d_llm_fwd(temp,b);
		transpose4x4(b);
		for(int i=0;i<16;i++)b[i]*=0.250f;
	}
	else 
	{
		dct4x4_1d_llm_inv(a,temp);
		transpose4x4(temp);
		dct4x4_1d_llm_inv(temp,b);
		transpose4x4(b);
		for(int i=0;i<16;i++)b[i]*=0.250f;
	}
}

void dct4x4_llm_sse(float* a, float* b, float* temp ,int flag)
{
	if(flag==0)
	{
		dct4x4_1d_llm_fwd_sse(a,temp);
		transpose4x4(temp);
		dct4x4_1d_llm_fwd_sse(temp,b);
		transpose4x4(b);
		__m128 c=_mm_set1_ps(0.250f);
		_mm_store_ps(b,_mm_mul_ps(_mm_load_ps(b),c));
		_mm_store_ps(b+4,_mm_mul_ps(_mm_load_ps(b+4),c));
		_mm_store_ps(b+8,_mm_mul_ps(_mm_load_ps(b+8),c));
		_mm_store_ps(b+12,_mm_mul_ps(_mm_load_ps(b+12),c));
	}
	else 
	{
		dct4x4_1d_llm_inv_sse(a,temp);
		transpose4x4(temp);
		dct4x4_1d_llm_inv_sse(temp,b);
		transpose4x4(b);
		__m128 c=_mm_set1_ps(0.250f);
		_mm_store_ps(b,_mm_mul_ps(_mm_load_ps(b),c));
		_mm_store_ps(b+4,_mm_mul_ps(_mm_load_ps(b+4),c));
		_mm_store_ps(b+8,_mm_mul_ps(_mm_load_ps(b+8),c));
		_mm_store_ps(b+12,_mm_mul_ps(_mm_load_ps(b+12),c));
	}
}
