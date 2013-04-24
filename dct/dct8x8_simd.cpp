#include <nmmintrin.h> //SSE4.2
#define  _USE_MATH_DEFINES
#include <math.h>

//info: code
//http://d.hatena.ne.jp/shiku_otomiya/20100902/p1 (in japanese)

//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.

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

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);
	_mm_store_ps(y,_mm_mul_ps(_mm_add_ps(c0,c1),invsqrt2h));
	_mm_store_ps(y+32,_mm_mul_ps(_mm_sub_ps(c0,c1),invsqrt2h));

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);	
	_mm_store_ps(y+16,_mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0,c2),_mm_mul_ps(w1,c3)),  invsqrt2h));
	_mm_store_ps(y+48,_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0,c3),_mm_mul_ps(w1,c2)),  invsqrt2h));
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

	_mm_store_ps(y+24,_mm_mul_ps(_mm_sub_ps(c0,c2),  invsqrt2h));
	_mm_store_ps(y+40,_mm_mul_ps(_mm_sub_ps(c3,c1),  invsqrt2h));
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0=_mm_mul_ps(_mm_add_ps(c0,c2), invsqrt2);
	c3=_mm_mul_ps(_mm_add_ps(c3,c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	_mm_store_ps(y+8,_mm_mul_ps(_mm_add_ps(c0,c3),  invsqrt2h));
	_mm_store_ps(y+56,_mm_mul_ps(_mm_sub_ps(c0,c3),  invsqrt2h));
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{ 
	y[i] *= invsqrt2h; 
	}*/
}
void fDCT8x8_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			temp[8*i+j] =s[8*j+i];
		}
	}

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp+4, d+4);

	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			temp[8*i+j] =d[8*j+i];
		}
	}
	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp+4, d+4);
}

void fDCT1Dllm_32f(const float* x, float* y)
{
	float t0,t1,t2,t3,t4,t5,t6,t7; float c0,c1,c2,c3; float r[8];int i;

	for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
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

	for(i = 0;i < 8;i++)
	{ 
		y[i] *= invsqrt2h; 
	}
}

void fDCT2Dllm_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j ++)
	{
		fDCT1Dllm_32f(s+j*8, temp+j*8);
	}

	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			d[8*i+j] =temp[8*j+i];
		}
	}
	for (int j = 0; j < 8; j ++)
	{
		fDCT1Dllm_32f(d+j*8, temp+j*8);
	}

	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			d[8*i+j] =temp[8*j+i];
		}
	}
}

void iDCT1Dllm_32f(const float* y, float* x)
{
	float a0,a1,a2,a3,b0,b1,b2,b3; float z0,z1,z2,z3,z4; float r[8]; int i;

	for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }

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

	for(i = 0;i < 8;i++){ x[i] *= 0.353554f; }
}

void iDCT2Dllm_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j ++)
	{
		iDCT1Dllm_32f(s+j*8, temp+j*8);
	}

	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			d[8*i+j] =temp[8*j+i];
		}
	}
	for (int j = 0; j < 8; j ++)
	{
		iDCT1Dllm_32f(d+j*8, temp+j*8);
	}

	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			d[8*i+j] =temp[8*j+i];
		}
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

	w = _mm_set1_ps(0.353554f);
	_mm_store_ps(x   ,_mm_mul_ps(w,_mm_add_ps(my0,mb0)));
	_mm_store_ps(x+56,_mm_mul_ps(w,_mm_sub_ps(my0,mb0)));
	_mm_store_ps(x+ 8,_mm_mul_ps(w,_mm_add_ps(my1,mb1)));
	_mm_store_ps(x+48,_mm_mul_ps(w,_mm_sub_ps(my1,mb1)));
	_mm_store_ps(x+16,_mm_mul_ps(w,_mm_add_ps(my2,mb2)));
	_mm_store_ps(x+40,_mm_mul_ps(w,_mm_sub_ps(my2,mb2)));
	_mm_store_ps(x+24,_mm_mul_ps(w,_mm_add_ps(my3,mb3)));
	_mm_store_ps(x+32,_mm_mul_ps(w,_mm_sub_ps(my3,mb3)));
	/*
	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;
	for(i = 0;i < 8;i++){ x[i] *= 0.353554f; }
	*/
}

void iDCT8x8_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			temp[8*i+j] =s[8*j+i];
		}
	}
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp+4, d+4);

	for (int j = 0; j < 8; j ++)
	{
		for (int i = 0; i < 8; i ++)
		{
			temp[8*i+j] =d[8*j+i];
		}
	}
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp+4, d+4);
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