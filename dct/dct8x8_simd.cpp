#include <nmmintrin.h> //SSE4.2
#define  _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>


//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.

void transpose8x8(float* src);
void transpose8x8(const float* src, float* dest);

void fDCT2D8x4_32f(const float* x, float* y)
{
	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x+56);
	__m128 t0 = _mm_add_ps(c0,c1);
	__m128 t7 = _mm_sub_ps(c0,c1);

	c1 = _mm_load_ps(x+48);
	c0 = _mm_load_ps(x+8);
	__m128 t1 = _mm_add_ps(c0,c1);
	__m128 t6 = _mm_sub_ps(c0,c1);

	c1 = _mm_load_ps(x+40);
	c0 = _mm_load_ps(x+16);
	__m128 t2 = _mm_add_ps(c0,c1);
	__m128 t5 = _mm_sub_ps(c0,c1);

	c0 = _mm_load_ps(x+24);
	c1 = _mm_load_ps(x+32);
	__m128 t3 = _mm_add_ps(c0,c1);
	__m128 t4 = _mm_sub_ps(c0,c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0,t3);
	__m128 c3 = _mm_sub_ps(t0,t3);
	c1 = _mm_add_ps(t1,t2);
	__m128 c2 = _mm_sub_ps(t1,t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	
	_mm_store_ps(y,_mm_add_ps(c0,c1));
	_mm_store_ps(y+32,_mm_sub_ps(c0,c1));

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);	
	_mm_store_ps(y+16,_mm_add_ps(_mm_mul_ps(w0,c2),_mm_mul_ps(w1,c3)));
	_mm_store_ps(y+48,_mm_sub_ps(_mm_mul_ps(w0,c3),_mm_mul_ps(w1,c2)));
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0,t4),_mm_mul_ps(w1,t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0,t7),_mm_mul_ps(w1,t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0,t5),_mm_mul_ps(w1,t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0,t6),_mm_mul_ps(w1,t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	_mm_store_ps(y+24,_mm_sub_ps(c0,c2));
	_mm_store_ps(y+40,_mm_sub_ps(c3,c1));
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0=_mm_mul_ps(_mm_add_ps(c0,c2), invsqrt2);
	c3=_mm_mul_ps(_mm_add_ps(c3,c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	_mm_store_ps(y+8,_mm_add_ps(c0,c3));
	_mm_store_ps(y+56,_mm_sub_ps(c0,c3));
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{ 
	y[i] *= invsqrt2h; 
	}*/
}
void fDCT8x8_llm_sse(const float* s, float* d, float* temp)
{
	transpose8x8(s,temp);

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp+4, d+4);

	transpose8x8(d,temp);
	
	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp+4, d+4);

	__m128 c=_mm_set1_ps(0.1250f);
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//0
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//1
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//2
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//3
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//4
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//5
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//6
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//7
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//8
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//9
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//10
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//11
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//12
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//13
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//14
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//15
}

void fDCT1Dllm_32f(const float* x, float* y)
{
	float t0,t1,t2,t3,t4,t5,t6,t7; float c0,c1,c2,c3; float r[8];

	//for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
	r[0]=1.414214f;
	r[1]=1.387040f;
	r[2]=1.306563f;
	r[3]=1.175876f;
	r[4]=1.000000f;
	r[5]=0.785695f;
	r[6]=0.541196f;
	r[7]=0.275899f;

	const float invsqrt2= 0.707107f;//(float)(1.0f / M_SQRT2);
	const float invsqrt2h=0.353554f;//invsqrt2*0.5f;

	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;

	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;

	y[0] = c0 + c1;
	y[4] = c0 - c1;
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];

	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];

	y[5] = c3 - c1; y[3] = c0 - c2;
	c0 = (c0 + c2) * invsqrt2;
	c3 = (c3 + c1) * invsqrt2;
	y[1] = c0 + c3; y[7] = c0 - c3;
}

void fDCT2D_llm(const float* s, float* d, float* temp)
{
	int j;
	for (j = 0; j < 8; j ++)
	{
		fDCT1Dllm_32f(s+j*8, temp+j*8);
	}
	transpose8x8(temp,d);

	for (j = 0; j < 8; j ++)
	{
		fDCT1Dllm_32f(d+j*8, temp+j*8);
	}
	transpose8x8(temp,d);

	for(j = 0;j < 64;j++)
	{ 
		d[j] *= 0.125; 
	}
}

void iDCT1Dllm_32f(const float* y, float* x)
{
	float a0,a1,a2,a3,b0,b1,b2,b3; float z0,z1,z2,z3,z4; float r[8];
	//for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2);printf("%f\n",r[i]); }
	r[0]=1.414214f;
	r[1]=1.387040f;
	r[2]=1.306563f;
	r[3]=1.175876f;
	r[4]=1.000000f;
	r[5]=0.785695f;
	r[6]=0.541196f;
	r[7]=0.275899f;
	
	z0 = y[1] + y[7]; z1 = y[3] + y[5]; z2 = y[3] + y[7]; z3 = y[1] + y[5];
	z4 = (z0 + z1) * r[3];

	z0 = z0 * (-r[3] + r[7]);
	z1 = z1 * (-r[3] - r[1]);
	z2 = z2 * (-r[3] - r[5]) + z4;
	z3 = z3 * (-r[3] + r[5]) + z4;

	b3 = y[7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2;
	b2 = y[5] * ( r[1] + r[3] - r[5] + r[7]) + z1 + z3;
	b1 = y[3] * ( r[1] + r[3] + r[5] - r[7]) + z1 + z2;
	b0 = y[1] * ( r[1] + r[3] - r[5] - r[7]) + z0 + z3;

	z4 = (y[2] + y[6]) * r[6];
	z0 = y[0] + y[4]; z1 = y[0] - y[4];
	z2 = z4 - y[6] * (r[2] + r[6]);
	z3 = z4 + y[2] * (r[2] - r[6]);
	a0 = z0 + z3; a3 = z0 - z3;
	a1 = z1 + z2; a2 = z1 - z2;

	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;
}

void iDCT2D_llm(const float* s, float* d, float* temp)
{
	int j;

	for (j = 0; j < 8; j ++)
	{
		iDCT1Dllm_32f(s+j*8, temp+j*8);
	}
	transpose8x8(temp,d);

	for (j = 0; j < 8; j ++)
	{
		iDCT1Dllm_32f(d+j*8, temp+j*8);
	}
	transpose8x8(temp,d);

	for(j = 0;j < 64;j++)
	{ 
		d[j] *= 0.125f; 
	}
}

void iDCT2D8x4_32f(const float* y, float* x)
{
	/*
	float a0,a1,a2,a3,b0,b1,b2,b3; float z0,z1,z2,z3,z4; float r[8]; int i;
	for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
	*/
	/*
	0: 1.414214
	1: 1.387040
	2: 1.306563
	3: 
	4: 1.000000
	5: 0.785695
	6: 
	7: 0.275899
	*/
	__m128 my1 = _mm_load_ps(y+8);
	__m128 my7 = _mm_load_ps(y+56);
	__m128 mz0 = _mm_add_ps(my1,my7);

	__m128 my3 = _mm_load_ps(y+24);
	__m128 mz2 = _mm_add_ps(my3,my7);
	__m128 my5 = _mm_load_ps(y+40);
	__m128 mz1 = _mm_add_ps(my3,my5);
	__m128 mz3 = _mm_add_ps(my1,my5);

	__m128 w = _mm_set1_ps(1.175876f);
	__m128 mz4 = _mm_mul_ps(_mm_add_ps(mz0,mz1),w);
	//z0 = y[1] + y[7]; z1 = y[3] + y[5]; z2 = y[3] + y[7]; z3 = y[1] + y[5];
	//z4 = (z0 + z1) * r[3];

	w = _mm_set1_ps(-1.961571f);
	mz2 =_mm_add_ps(_mm_mul_ps(mz2,w),mz4);
	w = _mm_set1_ps(-0.390181f);
	mz3 =_mm_add_ps(_mm_mul_ps(mz3,w),mz4);
	w = _mm_set1_ps(-0.899976f);
	mz0 =_mm_mul_ps(mz0,w);
	w = _mm_set1_ps(-2.562915f);
	mz1 =_mm_mul_ps(mz1,w);

	
	/*
	-0.899976
	-2.562915
	-1.961571
	-0.390181
	z0 = z0 * (-r[3] + r[7]);
	z1 = z1 * (-r[3] - r[1]);
	z2 = z2 * (-r[3] - r[5]) + z4;
	z3 = z3 * (-r[3] + r[5]) + z4;*/

	w = _mm_set1_ps(0.298631f);
	__m128 mb3 =_mm_add_ps(_mm_add_ps(_mm_mul_ps(my7,w),mz0),mz2);
	w = _mm_set1_ps(2.053120f);
	__m128 mb2 =_mm_add_ps(_mm_add_ps(_mm_mul_ps(my5,w),mz1),mz3);
	w = _mm_set1_ps(3.072711f);
	__m128 mb1 =_mm_add_ps(_mm_add_ps(_mm_mul_ps(my3,w),mz1),mz2);
	w = _mm_set1_ps(1.501321f);
	__m128 mb0 =_mm_add_ps(_mm_add_ps(_mm_mul_ps(my1,w),mz0),mz3);
	/*
	0.298631
	2.053120
	3.072711
	1.501321
	b3 = y[7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2;
	b2 = y[5] * ( r[1] + r[3] - r[5] + r[7]) + z1 + z3;
	b1 = y[3] * ( r[1] + r[3] + r[5] - r[7]) + z1 + z2;
	b0 = y[1] * ( r[1] + r[3] - r[5] - r[7]) + z0 + z3;
	*/

	__m128 my2 = _mm_load_ps(y+16);
	__m128 my6 = _mm_load_ps(y+48);
	w = _mm_set1_ps(0.541196f);
	mz4 = _mm_mul_ps(_mm_add_ps(my2,my6),w);
	__m128 my0 = _mm_load_ps(y);
	__m128 my4 = _mm_load_ps(y+32);
	mz0=_mm_add_ps(my0,my4);
	mz1=_mm_sub_ps(my0,my4);


	w = _mm_set1_ps(-1.847759f);
	mz2=_mm_add_ps(mz4,_mm_mul_ps(my6,w));
	w = _mm_set1_ps(0.765367f);
	mz3=_mm_add_ps(mz4,_mm_mul_ps(my2,w));

	my0 = _mm_add_ps(mz0,mz3);
	my3 = _mm_sub_ps(mz0,mz3);
	my1 = _mm_add_ps(mz1,mz2);
	my2 = _mm_sub_ps(mz1,mz2);
	/*
	1.847759
	0.765367
	z4 = (y[2] + y[6]) * r[6];
	z0 = y[0] + y[4]; z1 = y[0] - y[4];
	z2 = z4 - y[6] * (r[2] + r[6]);
	z3 = z4 + y[2] * (r[2] - r[6]);
	a0 = z0 + z3; a3 = z0 - z3;
	a1 = z1 + z2; a2 = z1 - z2;
	*/

	_mm_store_ps(x   ,_mm_add_ps(my0,mb0));
	_mm_store_ps(x+56,_mm_sub_ps(my0,mb0));
	_mm_store_ps(x+ 8,_mm_add_ps(my1,mb1));
	_mm_store_ps(x+48,_mm_sub_ps(my1,mb1));
	_mm_store_ps(x+16,_mm_add_ps(my2,mb2));
	_mm_store_ps(x+40,_mm_sub_ps(my2,mb2));
	_mm_store_ps(x+24,_mm_add_ps(my3,mb3));
	_mm_store_ps(x+32,_mm_sub_ps(my3,mb3));
	/*
	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;
	for(i = 0;i < 8;i++){ x[i] *= 0.353554f; }
	*/
}

void iDCT8x8_llm_sse(const float* s, float* d, float* temp)
{
	transpose8x8(s,temp);

	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp+4, d+4);

	transpose8x8(d,temp);
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp+4, d+4);

	__m128 c=_mm_set1_ps(0.1250f);
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//0
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//1
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//2
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//3
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//4
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//5
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//6
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//7
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//8
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//9
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//10
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//11
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//12
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//13
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//14
	_mm_store_ps(d,_mm_mul_ps(_mm_load_ps(d),c));d+=4;//15
}

//Plonka, Gerlind, and Manfred Tasche. "Fast and numerically stable algorithms for discrete cosine transforms." Linear algebra and its applications 394 (2005) : 309 - 345.

static void fdct81d_GT(const float *src, float *dst)
{
	for (int i = 0; i < 8; i++)
	{

		const float mx00 = src[0] + src[7];
		const float mx01 = src[1] + src[6];
		const float mx02 = src[2] + src[5];
		const float mx03 = src[3] + src[4];
		const float mx04 = src[0] - src[7];
		const float mx05 = src[1] - src[6];
		const float mx06 = src[2] - src[5];
		const float mx07 = src[3] - src[4];
		const float mx08 = mx00 + mx03;
		const float mx09 = mx01 + mx02;
		const float mx0a = mx00 - mx03;
		const float mx0b = mx01 - mx02;
		const float mx0c = 1.38703984532215f*mx04 + 0.275899379282943f*mx07;
		const float mx0d = 1.17587560241936f*mx05 + 0.785694958387102f*mx06;
		const float mx0e = -0.785694958387102f*mx05 + 1.17587560241936f*mx06;
		const float mx0f = 0.275899379282943f*mx04 - 1.38703984532215f*mx07;
		const float mx10 = 0.353553390593274f * (mx0c - mx0d);
		const float mx11 = 0.353553390593274f * (mx0e - mx0f);
		dst[0] = 0.353553390593274f * (mx08 + mx09);
		dst[1] = 0.353553390593274f * (mx0c + mx0d);
		dst[2] = 0.461939766255643f*mx0a + 0.191341716182545f*mx0b;
		dst[3] = 0.707106781186547f * (mx10 - mx11);
		dst[4] = 0.353553390593274f * (mx08 - mx09);
		dst[5] = 0.707106781186547f * (mx10 + mx11);
		dst[6] = 0.191341716182545f*mx0a - 0.461939766255643f*mx0b;
		dst[7] = 0.353553390593274f * (mx0e + mx0f);
		dst += 8;
		src += 8;
	}
}

static void idct81d_GT(const float *src, float *dst)
{
	for (int i = 0; i < 8; i++)
	{
		const float mx00 = 1.4142135623731f  *src[0];
		const float mx01 = 1.38703984532215f *src[1] + 0.275899379282943f*src[7];
		const float mx02 = 1.30656296487638f *src[2] + 0.541196100146197f*src[6];
		const float mx03 = 1.17587560241936f *src[3] + 0.785694958387102f*src[5];
		const float mx04 = 1.4142135623731f  *src[4];
		const float mx05 = -0.785694958387102f*src[3] + 1.17587560241936f*src[5];
		const float mx06 = 0.541196100146197f*src[2] - 1.30656296487638f*src[6];
		const float mx07 = -0.275899379282943f*src[1] + 1.38703984532215f*src[7];
		const float mx09 = mx00 + mx04;
		const float mx0a = mx01 + mx03;
		const float mx0b = 1.4142135623731f*mx02;
		const float mx0c = mx00 - mx04;
		const float mx0d = mx01 - mx03;
		const float mx0e = 0.353553390593274f * (mx09 - mx0b);
		const float mx0f = 0.353553390593274f * (mx0c + mx0d);
		const float mx10 = 0.353553390593274f * (mx0c - mx0d);
		const float mx11 = 1.4142135623731f*mx06;
		const float mx12 = mx05 + mx07;
		const float mx13 = mx05 - mx07;
		const float mx14 = 0.353553390593274f * (mx11 + mx12);
		const float mx15 = 0.353553390593274f * (mx11 - mx12);
		const float mx16 = 0.5f*mx13;
		dst[0] = 0.25f * (mx09 + mx0b) + 0.353553390593274f*mx0a;
		dst[1] = 0.707106781186547f * (mx0f + mx15);
		dst[2] = 0.707106781186547f * (mx0f - mx15);
		dst[3] = 0.707106781186547f * (mx0e + mx16);
		dst[4] = 0.707106781186547f * (mx0e - mx16);
		dst[5] = 0.707106781186547f * (mx10 - mx14);
		dst[6] = 0.707106781186547f * (mx10 + mx14);
		dst[7] = 0.25f * (mx09 + mx0b) - 0.353553390593274f*mx0a;
		dst += 8;
		src += 8;
	}
}

static void fdct81d_sse_GT(const float *src, float *dst)
{
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);
	for (int i = 0; i < 2; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 8);
		__m128 ms2 = _mm_load_ps(src + 16);
		__m128 ms3 = _mm_load_ps(src + 24);
		__m128 ms4 = _mm_load_ps(src + 32);
		__m128 ms5 = _mm_load_ps(src + 40);
		__m128 ms6 = _mm_load_ps(src + 48);
		__m128 ms7 = _mm_load_ps(src + 56);

		__m128 mx00 = _mm_add_ps(ms0, ms7);
		__m128 mx01 = _mm_add_ps(ms1, ms6);
		__m128 mx02 = _mm_add_ps(ms2, ms5);
		__m128 mx03 = _mm_add_ps(ms3, ms4);
		__m128 mx04 = _mm_sub_ps(ms0, ms7);
		__m128 mx05 = _mm_sub_ps(ms1, ms6);
		__m128 mx06 = _mm_sub_ps(ms2, ms5);
		__m128 mx07 = _mm_sub_ps(ms3, ms4);
		__m128 mx08 = _mm_add_ps(mx00, mx03);
		__m128 mx09 = _mm_add_ps(mx01, mx02);
		__m128 mx0a = _mm_sub_ps(mx00, mx03);
		__m128 mx0b = _mm_sub_ps(mx01, mx02);

		__m128 mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
		__m128 mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
		__m128 mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
		__m128 mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
		__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		__m128 mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

		_mm_store_ps(dst + 0, _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09)));
		_mm_store_ps(dst + 8, _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d)));
		_mm_store_ps(dst + 16, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b)));
		_mm_store_ps(dst + 24, _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11)));
		_mm_store_ps(dst + 32, _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09)));
		_mm_store_ps(dst + 40, _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11)));
		_mm_store_ps(dst + 48, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b)));
		_mm_store_ps(dst + 56, _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f)));
		dst += 4;
		src += 4;
	}
}

static void fdct88_sse_GT(const float *src, float *dst)
{
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);

	__m128 ms0 = _mm_load_ps(src);
	__m128 ms1 = _mm_load_ps(src + 8);
	__m128 ms2 = _mm_load_ps(src + 16);
	__m128 ms3 = _mm_load_ps(src + 24);
	__m128 ms4 = _mm_load_ps(src + 32);
	__m128 ms5 = _mm_load_ps(src + 40);
	__m128 ms6 = _mm_load_ps(src + 48);
	__m128 ms7 = _mm_load_ps(src + 56);

	__m128 mx00 = _mm_add_ps(ms0, ms7);
	__m128 mx01 = _mm_add_ps(ms1, ms6);
	__m128 mx02 = _mm_add_ps(ms2, ms5);
	__m128 mx03 = _mm_add_ps(ms3, ms4);
	__m128 mx04 = _mm_sub_ps(ms0, ms7);
	__m128 mx05 = _mm_sub_ps(ms1, ms6);
	__m128 mx06 = _mm_sub_ps(ms2, ms5);
	__m128 mx07 = _mm_sub_ps(ms3, ms4);
	__m128 mx08 = _mm_add_ps(mx00, mx03);
	__m128 mx09 = _mm_add_ps(mx01, mx02);
	__m128 mx0a = _mm_sub_ps(mx00, mx03);
	__m128 mx0b = _mm_sub_ps(mx01, mx02);

	__m128 mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	__m128 mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	__m128 mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	__m128 mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	__m128 mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	__m128 md00 = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	__m128 md01 = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	__m128 md02 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	__m128 md03 = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));

	_MM_TRANSPOSE4_PS(md00, md01, md02, md03);

	__m128 md10 = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	__m128 md11 = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	__m128 md12 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	__m128 md13 = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	_MM_TRANSPOSE4_PS(md10, md11, md12, md13);

	src += 4;
	ms0 = _mm_load_ps(src);
	ms1 = _mm_load_ps(src + 8);
	ms2 = _mm_load_ps(src + 16);
	ms3 = _mm_load_ps(src + 24);
	ms4 = _mm_load_ps(src + 32);
	ms5 = _mm_load_ps(src + 40);
	ms6 = _mm_load_ps(src + 48);
	ms7 = _mm_load_ps(src + 56);

	mx00 = _mm_add_ps(ms0, ms7);
	mx01 = _mm_add_ps(ms1, ms6);
	mx02 = _mm_add_ps(ms2, ms5);
	mx03 = _mm_add_ps(ms3, ms4);
	mx04 = _mm_sub_ps(ms0, ms7);
	mx05 = _mm_sub_ps(ms1, ms6);
	mx06 = _mm_sub_ps(ms2, ms5);
	mx07 = _mm_sub_ps(ms3, ms4);
	mx08 = _mm_add_ps(mx00, mx03);
	mx09 = _mm_add_ps(mx01, mx02);
	mx0a = _mm_sub_ps(mx00, mx03);
	mx0b = _mm_sub_ps(mx01, mx02);

	mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	__m128 md04 = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	__m128 md05 = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	__m128 md06 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	__m128 md07 = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
	_MM_TRANSPOSE4_PS(md04, md05, md06, md07);

	__m128 md14 = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	__m128 md15 = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	__m128 md16 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	__m128 md17 = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	_MM_TRANSPOSE4_PS(md14, md15, md16, md17);

	src -= 4;
	ms0 = md00;
	ms1 = md01;
	ms2 = md02;
	ms3 = md03;
	ms4 = md04;
	ms5 = md05;
	ms6 = md06;
	ms7 = md07;

	mx00 = _mm_add_ps(ms0, ms7);
	mx01 = _mm_add_ps(ms1, ms6);
	mx02 = _mm_add_ps(ms2, ms5);
	mx03 = _mm_add_ps(ms3, ms4);
	mx04 = _mm_sub_ps(ms0, ms7);
	mx05 = _mm_sub_ps(ms1, ms6);
	mx06 = _mm_sub_ps(ms2, ms5);
	mx07 = _mm_sub_ps(ms3, ms4);
	mx08 = _mm_add_ps(mx00, mx03);
	mx09 = _mm_add_ps(mx01, mx02);
	mx0a = _mm_sub_ps(mx00, mx03);
	mx0b = _mm_sub_ps(mx01, mx02);

	mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	__m128 a = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	__m128 b = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	__m128 c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	__m128 d = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));

	_mm_store_ps(dst + 0, a);
	_mm_store_ps(dst + 8, b);
	_mm_store_ps(dst + 16, c);
	_mm_store_ps(dst + 24, d);

	a = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	b = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	d = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	_MM_TRANSPOSE4_PS(a, b, c, d);
	dst += 4;
	_mm_store_ps(dst + 0, a);
	_mm_store_ps(dst + 8, b);
	_mm_store_ps(dst + 16, c);
	_mm_store_ps(dst + 24, d);

	ms0 = md10;
	ms1 = md11;
	ms2 = md12;
	ms3 = md13;
	ms4 = md14;
	ms5 = md15;
	ms6 = md16;
	ms7 = md17;

	mx00 = _mm_add_ps(ms0, ms7);
	mx01 = _mm_add_ps(ms1, ms6);
	mx02 = _mm_add_ps(ms2, ms5);
	mx03 = _mm_add_ps(ms3, ms4);
	mx04 = _mm_sub_ps(ms0, ms7);
	mx05 = _mm_sub_ps(ms1, ms6);
	mx06 = _mm_sub_ps(ms2, ms5);
	mx07 = _mm_sub_ps(ms3, ms4);
	mx08 = _mm_add_ps(mx00, mx03);
	mx09 = _mm_add_ps(mx01, mx02);
	mx0a = _mm_sub_ps(mx00, mx03);
	mx0b = _mm_sub_ps(mx01, mx02);

	mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	a = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	b = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	d = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
	_MM_TRANSPOSE4_PS(a, b, c, d);
	_mm_store_ps(dst + 28, a);
	_mm_store_ps(dst + 36, b);
	_mm_store_ps(dst + 44, c);
	_mm_store_ps(dst + 52, d);

	a = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	b = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	d = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	_MM_TRANSPOSE4_PS(a, b, c, d);

	_mm_store_ps(dst + 32, a);
	_mm_store_ps(dst + 40, b);
	_mm_store_ps(dst + 48, c);
	_mm_store_ps(dst + 56, d);
}

static void idct81d_sse_GT(const float *s, float *d)
{
	float* dst = d;
	float* src = (float*)s;
	const __m128 c1414 = _mm_set1_ps(1.4142135623731f);
	const __m128 c0250 = _mm_set1_ps(0.25f);
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);

	for (int i = 0; i < 2; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 8);
		__m128 ms2 = _mm_load_ps(src + 16);
		__m128 ms3 = _mm_load_ps(src + 24);
		__m128 ms4 = _mm_load_ps(src + 32);
		__m128 ms5 = _mm_load_ps(src + 40);
		__m128 ms6 = _mm_load_ps(src + 48);
		__m128 ms7 = _mm_load_ps(src + 56);

		__m128 mx00 = _mm_mul_ps(c1414, ms0);
		__m128 mx01 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), ms1), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), ms7));
		__m128 mx02 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.30656296487638f), ms2), _mm_mul_ps(_mm_set1_ps(0.541196100146197f), ms6));
		__m128 mx03 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), ms3), _mm_mul_ps(_mm_set1_ps(0.785694958387102f), ms5));
		__m128 mx04 = _mm_mul_ps(c1414, ms4);
		__m128 mx05 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), ms3), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), ms5));
		__m128 mx06 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.541196100146197f), ms2), _mm_mul_ps(_mm_set1_ps(-1.30656296487638f), ms6));
		__m128 mx07 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.275899379282943f), ms1), _mm_mul_ps(_mm_set1_ps(1.38703984532215f), ms7));
		__m128 mx09 = _mm_add_ps(mx00, mx04);
		__m128 mx0a = _mm_add_ps(mx01, mx03);
		__m128 mx0b = _mm_mul_ps(c1414, mx02);
		__m128 mx0c = _mm_sub_ps(mx00, mx04);
		__m128 mx0d = _mm_sub_ps(mx01, mx03);
		__m128 mx0e = _mm_mul_ps(c0353, _mm_sub_ps(mx09, mx0b));
		__m128 mx0f = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
		__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		__m128 mx11 = _mm_mul_ps(c1414, mx06);
		__m128 mx12 = _mm_add_ps(mx05, mx07);
		__m128 mx13 = _mm_sub_ps(mx05, mx07);
		__m128 mx14 = _mm_mul_ps(c0353, _mm_add_ps(mx11, mx12));
		__m128 mx15 = _mm_mul_ps(c0353, _mm_sub_ps(mx11, mx12));
		__m128 mx16 = _mm_mul_ps(_mm_set1_ps(0.5f), mx13);

		_mm_store_ps(dst + 0, _mm_add_ps(_mm_mul_ps(c0250, _mm_add_ps(mx09, mx0b)), _mm_mul_ps(c0353, mx0a)));
		_mm_store_ps(dst + 8, _mm_mul_ps(c0707, _mm_add_ps(mx0f, mx15)));
		_mm_store_ps(dst + 16, _mm_mul_ps(c0707, _mm_sub_ps(mx0f, mx15)));
		_mm_store_ps(dst + 24, _mm_mul_ps(c0707, _mm_add_ps(mx0e, mx16)));
		_mm_store_ps(dst + 32, _mm_mul_ps(c0707, _mm_sub_ps(mx0e, mx16)));
		_mm_store_ps(dst + 40, _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx14)));
		_mm_store_ps(dst + 48, _mm_mul_ps(c0707, _mm_add_ps(mx10, mx14)));
		_mm_store_ps(dst + 56, _mm_sub_ps(_mm_mul_ps(c0250, _mm_add_ps(mx09, mx0b)), _mm_mul_ps(c0353, mx0a)));
		dst += 4;
		src += 4;
	}
}

void fDCT8x8GT(const float* s, float* d)
{
	fdct88_sse_GT(s, d);
	/*fdct81d_sse_GT(s, d);
	transpose8x8(d);
	fdct81d_sse_GT(d, d);
	transpose8x8(d);*/
}

void iDCT8x8GT(const float* s, float* d)
{
	idct81d_sse_GT(s, d);
	transpose8x8(d);
	idct81d_sse_GT(d, d);
	transpose8x8(d);
}


#ifdef UNDERCONSTRUCTION_____
//internal simd using sse3
void LLMDCTOpt(const float* x, float* y)
{
	float t4,t5,t6,t7; float c0,c1,c2,c3; 
	float* r = dct_tbl;

	const float invsqrt2= 0.707107f;//(float)(1.0f / M_SQRT2);
	const float invsqrt2h=0.353554f;//invsqrt2*0.5f;

	{
		__m128 mc1 = _mm_load_ps(x);
		__m128 mc2 = _mm_loadr_ps(x+4);

		__m128 mt1 = _mm_add_ps(mc1,mc2);
		__m128 mt2 = _mm_sub_ps(mc1,mc2);//rev

		mc1 = _mm_addsub_ps(_mm_shuffle_ps(mt1,mt1,_MM_SHUFFLE(1,1,0,0)),_mm_shuffle_ps(mt1,mt1,_MM_SHUFFLE(2,2,3,3)));
		mc1 = _mm_shuffle_ps(mc1,mc1,_MM_SHUFFLE(0,2,3,1));

		_mm_store_ps(y,mc1);
		_mm_store_ps(y+4,mt2);

	}
	c0=y[0];
	c1=y[1];
	c2=y[2];
	c3=y[3];
	/*c3=y[0];
	c0=y[1];
	c2=y[2];
	c1=y[3];*/

	t7=y[4];
	t6=y[5];
	t5=y[6];
	t4=y[7];

	y[0] = c0 + c1;
	y[4] = c0 - c1;
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];

	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];

	y[5] = c3 - c1; y[3] = c0 - c2;
	c0 = (c0 + c2) * invsqrt2;
	c3 = (c3 + c1) * invsqrt2;
	y[1] = c0 + c3; y[7] = c0 - c3;

	const __m128 invsqh = _mm_set_ps1(invsqrt2h);
	__m128 my = _mm_load_ps(y);
	_mm_store_ps(y,_mm_mul_ps(my,invsqh));

	my = _mm_load_ps(y+4);
	_mm_store_ps(y+4,_mm_mul_ps(my,invsqh));
}
#endif

/*	
void print(__m128 src)
{
printf_s("%5.3f %5.3f %5.3f %5.3f\n",
src.m128_f32[0], src.m128_f32[1],
src.m128_f32[2], src.m128_f32[3] );
}*/
