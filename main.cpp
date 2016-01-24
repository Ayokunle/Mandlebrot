/*
 * CS3014 Mandelbrot Project
 * 
 * Using techniques we've covered in class, accelerate the rendering of
 * the M set.
 * 
 * Hints
 * 
 * 1) Vectorize
 * 2) Use threads
 * 3) Load Balance
 * 4) Profile and Optimise
 * 
 * Potential FAQ.
 * 
 * Q1) Why when I zoom in far while palying with the code, why does the image begin to render all blocky?
 * A1) In order to render at increasing depths we must use increasingly higher precision floats
 * 	   We quickly run out of precision with 32 bits floats. Change all floats to doubles if you want
 * 	   dive deeper. Eventually you will however run out of precision again and need to integrate an
 * 	   infinite precision math library or use other techniques.
 * 
 * Q2) Why do some frames render much faster than others?
 * A2) Frames with a lot of black, i.e, frames showing a lot of set M, show pixels that ran until the 
 *     maximum number of iterations was reached before bailout. This means more CPU time was consumed
 */



#include <iostream>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
 #include <xmmintrin.h>

// header file for sleep system call
#include <unistd.h> 

#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif



#include "Screen.h"


/*
 * You can't change these values to accelerate the rendering.
 * Feel free to play with them to render different images though.
 */
const int 	MAX_ITS = 1000;			//Max Iterations before we assume the point will not escape
const float HXRES = 700; 			// horizontal resolution	
const float HYRES = 700;			// vertical resolution
const int 	MAX_DEPTH = 40;		// max depth of zoom
const float ZOOM_FACTOR = 1.02;		// zoom between each frame

/* Change these to zoom into different parts of the image */
const float PX = -0.702295281061;	// Centre point we'll zoom on - Real component
const float PY = +0.350220783400;	// Imaginary component


/*
 * The palette. Modifying this can produce some really interesting renders.
 * The colours are arranged R1,G1,B1, R2, G2, B2, R3.... etc.
 * RGB values are 0 to 255 with 0 being darkest and 255 brightest
 * 0,0,0 is black
 * 255,255,255 is white
 * 255,0,0 is bright red
 */
unsigned char pal[]={
	219,112,147,
	218,112,214,
	0,255,127,
	255,215,0,
	222,184,135,
	139,69,19,
	128,30,250,
	180,149,250,
	250,250,250,
	100,28,250,
	191,82,250,
	47,5,250,
	138,39,250,
	81,27,4,
	192,89,250,
	61,27,250,
	216,148,250,
	71,14,250,
	142,48,250,
	196,102,250,
	250,9,250,
	132,192,250,
	194,15,250,
	92,250,250,
	166,59,4,
	244,178,4,
	194,121,4,
	120,41,219,
	25,125,219,
	80,250,219,
	250,3,219,
	249,204,219,
	250,25,219,
	250,30,250,
	250,192,250,
	104,192,250,
	239,171,250,
	130,57,250,
	111,192,250,
	250,192,250};
const int PAL_SIZE = 40;  //Number of entries in the palette 



/* 
 * Return true if the point cx,cy is a member of set M.
 * iterations is set to the number of iterations until escape.
 */
bool member(__m128 cx_, __m128 cy_, int& iterations)
{

	iterations = 0;

	__m128 x = _mm_set1_ps(0.0);
	__m128 y = _mm_set1_ps(0.0);

	__m128 x_sqrt = _mm_mul_ps(x, x);
	__m128 y_sqrt = _mm_mul_ps(y, y);
	__m128 two = _mm_set1_ps(2);
	__m128 two_sqrt = _mm_mul_ps(two, two);

	__m128 iterations_ = _mm_set1_ps(iterations);
	__m128 max_its_ = _mm_set1_ps(MAX_ITS);

	__m128 xy_sqrt = _mm_add_ps(x_sqrt, y_sqrt);
	__m128 temp1 = _mm_cmplt_ps(xy_sqrt, two_sqrt);

	__m128 temp2 = _mm_cmplt_ps(iterations_, max_its_);
	__m128 temp3 = _mm_and_ps(temp1, temp2);

	while( _mm_movemask_ps(temp3) == 15){

		__m128 xtemp = _mm_sub_ps(x_sqrt, _mm_add_ps(y_sqrt, cx_));
		y = _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(x,y)), cy_);
		x = xtemp;
		__m128 one = _mm_set1_ps(1);
		iterations_ = _mm_add_ps(iterations_, one);

		x_sqrt = _mm_mul_ps(x, x);
		y_sqrt = _mm_mul_ps(y, y);

		xy_sqrt = _mm_add_ps(x_sqrt, y_sqrt);
		temp1 = _mm_cmplt_ps(xy_sqrt, two_sqrt);

		temp2 = _mm_cmplt_ps(iterations_, max_its_);

		temp3 = _mm_and_ps(temp1, temp2);
	}
	
	float it[4] ;
	_mm_storeu_ps(it, iterations_);
	iterations = it[0];

	temp2 = _mm_cmpeq_ps(iterations_, max_its_);

	return (_mm_movemask_ps(temp2) == 15);
}


#define SIZE 4096

float vals[SIZE];

float a, b;
int main(){	

	int hx, hy = 0;

	float m=1.0; /* initial  magnification */

	/*  */
	__m128 vHXRES = _mm_set1_ps(HXRES);
	__m128 vHYRES = _mm_set1_ps(HYRES);
			
	__m128 vPX = _mm_set1_ps(PX);
	__m128 vPY = _mm_set1_ps(PY);

	__m128 constant = _mm_set1_ps(0.5);
	__m128 constant_four = _mm_set1_ps(4.0);
	__m128 constant_four_f = _mm_set1_ps(4.0f);
	
	/* Create a screen to render to */
	Screen *screen;
	screen = new Screen((int)HXRES, (int) HYRES);

	int depth=0;

#ifdef TIMING
  	struct timeval start_time;
  	struct timeval stop_time;
  	long long total_time = 0;
#endif
  	//#pragma omp parallel
	while (depth < MAX_DEPTH) {
#ifdef TIMING
	 /* record starting time */
	gettimeofday(&start_time, NULL);
#endif
	    #pragma omp parallel for private(hy, hx) shared(constant_four_f, constant_four, vHXRES, vHYRES) schedule(dynamic)
	    
		for (hy=0; hy< (int) HYRES; hy++) {
			for (hx=0; hx<  (int) HXRES; hx++){
				int iterations;

				/* 
				 * Translate pixel coordinates to complex plane coordinates centred
				 * on PX, PY
				 */	

				//float cx = ((((float)hx / (float)HXRES) -0.5 + (PX/(4.0/m))) *(4.0f/m));
				//float cy = ((((float)hy / (float)HYRES) -0.5 + (PY/(4.0/m)))*(4.0f/m));


				__m128 vm = _mm_set1_ps(m);*
				__m128 temp3 = _mm_div_ps( constant_four,  vm); /* 4.0/m */
				__m128 temp6 = _mm_div_ps(constant_four_f, vm); /* 4.0f/m */
				
				__m128 vhx = _mm_set1_ps(hx); /* vhx contains 4 copies of hx */
				__m128 temp1 = _mm_div_ps( vhx,  vHXRES); /* hx/HXRES */
				__m128 temp2 = _mm_sub_ps( temp1,  constant); /*(hx/ HXRES)- 0.5 */
				__m128 temp4 = _mm_div_ps( vPX,  temp3); /* PX/(4.0/m) */
				__m128 temp5 = _mm_add_ps( temp2,  temp4); /*((hx/HXRES) -0.5) + PX/(4.0/m) */
				__m128 vcx = _mm_mul_ps( temp5,  temp6); /* ((hx/HXRES) -0.5) + PX/(4.0/m))*(4.0f/m) */
				
//---------------------------------------------------------------------------------------------------------------------

				__m128 vhy = _mm_set1_ps(hy); /* vhy contains 4 copies of hy */
				temp1 = _mm_div_ps( vhy,  vHYRES);
				temp2 = _mm_sub_ps( temp1,  constant);
				temp4 = _mm_div_ps( vPY,  temp3);
				temp5 = _mm_add_ps( temp2,  temp4);
				__m128 vcy = _mm_mul_ps( temp5,  temp6);

				if (!member(vcx, vcy, iterations)){
					/* Point is not a member, colour based on number of iterations before escape */
					int i=(iterations%40) - 1;
					int b = i*3;
					screen->putpixel(hx, hy, pal[b], pal[b+1], pal[b+2]);
				} else {
					/* Point is a member, colour it black */
					screen->putpixel(hx, hy, 0, 0, 0);
				}
			}
		}
		
#ifdef TIMING
		gettimeofday(&stop_time, NULL);
		total_time += (stop_time.tv_sec - start_time.tv_sec) * 1000000L + (stop_time.tv_usec - start_time.tv_usec);
#endif
		/* Show the rendered image on the screen */
		screen->flip();
		std::cout << "Render done " << depth++ << " " << m << std::endl;
		
		/* Zoom in */
		m *= ZOOM_FACTOR;
	}
	
	//sleep(2);
#ifdef TIMING
	std::cout << "Total executing time " << total_time << " microseconds\n";
	std::cout << "Total executing time " << (float) total_time/1000000 << " seconds\n";
#endif
	std::cout << "Clean Exit"<< std::endl;

}