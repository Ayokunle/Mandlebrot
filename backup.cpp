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
const float 	HXRES = 700; 			// horizontal resolution	
const float 	HYRES = 700;			// vertical resolution
const int 	MAX_DEPTH = 40;		// max depth of zoom
const float ZOOM_FACTOR = 1.1;		// zoom between each frame

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
bool member(float cx, float cy, int& iterations)
{
	float x = 0.0;
	float y = 0.0;
	iterations = 0;
	//#pragma omp parallel
	while ((x*x + y*y < (2*2)) && (iterations < MAX_ITS)) {
		float xtemp = x*x - y*y + cx;
		y = 2*x*y + cy;
		x = xtemp;
		iterations++;
	}

	return (iterations == MAX_ITS);
}


#include <xmmintrin.h>

#define SIZE 4096

float vals[SIZE];

float a, b;
int main(){	

	int hx, hy = 0;

	float m=1.0; /* initial  magnification		*/


	/*     */
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

	while (depth < MAX_DEPTH) {
#ifdef TIMING
	 /* record starting time */
	gettimeofday(&start_time, NULL);
#endif
	    #pragma omp parallel for private(hy, hx) shared(constant_four_f, constant_four, vHXRES, vHYRES)
	    
		for (hy=0; hy< (int) HYRES; hy++) {
			for (hx=0; hx<  (int) HXRES; hx++){
				int iterations;

				/* 
				 * Translate pixel coordinates to complex plane coordinates centred
				 * on PX, PY
				 */	
				
				//float cx = ((((float)hx / (float)HXRES) -0.5 + (PX/(4.0/m))) *(4.0f/m));
				//float cy = ((((float)hy / (float)HYRES) -0.5 + (PY/(4.0/m)))*(4.0f/m));
				
				__m128 vhx = _mm_set1_ps(hx); /* vhx contains 4 copies of hx */

				// float xy[4] ;

				// _mm_storeu_ps(xy, vhx);
				// std::cout << "float hx[0] : " << xy[0] << std::endl;
				// std::cout << "float hx : " << hx << std::endl;

				//__m128 vhy = _mm_set1_ps(hy); /* vhy contains 4 copies of hy */

				__m128 temp1 = _mm_div_ps( vhx,  vHXRES);
				// xy[4] ;

				// _mm_storeu_ps(xy, temp1);
				// std::cout << "float temp1 : " << xy[0] << std::endl;
				// std::cout << "float (hx/HXRES) : " << ((float)hx/(float)HXRES) << std::endl;

				__m128 temp2 = _mm_sub_ps( temp1,  constant);
				// xy[4] ;

				// _mm_storeu_ps(xy, temp2);
				// std::cout << "float temp2 : " << xy[0] << std::endl;
				// std::cout << "float (hx/HXRES)-0.5 : " << (((float)hx/(float)HXRES)-0.5) << std::endl;

				__m128 vm = _mm_set1_ps(m);
				// xy[4] ;

				// _mm_storeu_ps(xy, vm);
				// std::cout << "float vm : " << xy[0] << std::endl;
				// std::cout << "float (m) : " << (m) << std::endl;

				// xy[4] ;

				// _mm_storeu_ps(xy, constant_four);
				// std::cout << "float constant_four : " << xy[0] << std::endl;
				// std::cout << "float constant_four - real : " << 4.0 << std::endl;


				__m128 temp3 = _mm_div_ps( constant_four,  vm);
				// xy[4] ;

				// _mm_storeu_ps(xy, temp3);
				// std::cout << "float temp3 : " << xy[0] << std::endl;
				// std::cout << "float (4.0/m) : " << (4.0/m) << std::endl;

				__m128 temp4 = _mm_div_ps( vPX,  temp3);

				__m128 temp5 = _mm_add_ps( temp2,  temp4);
				__m128 temp6 = _mm_div_ps(constant_four_f, vm);

				__m128 vcx = _mm_mul_ps( temp5,  temp6);
				float xy[4] ;

				_mm_storeu_ps(xy, vcx);
				// std::cout << "float xy : " << xy[0] << std::endl;
				// std::cout << "float cx : " << cx << std::endl << std::endl;
//---------------------------------------------------------------------------------------------------------------------

				__m128 vhy = _mm_set1_ps(hy); /* vhx contains 4 copies of hx */

				// float xy[4] ;

				// _mm_storeu_ps(xy, vhx);
				// std::cout << "float hx[0] : " << xy[0] << std::endl;
				// std::cout << "float hx : " << hx << std::endl;

				//__m128 vhy = _mm_set1_ps(hy); /* vhy contains 4 copies of hy */

				temp1 = _mm_div_ps( vhy,  vHYRES);
				// xy[4] ;

				// _mm_storeu_ps(xy, temp1);
				// std::cout << "float temp1 : " << xy[0] << std::endl;
				// std::cout << "float (hx/HXRES) : " << ((float)hx/(float)HXRES) << std::endl;

				temp2 = _mm_sub_ps( temp1,  constant);
				// xy[4] ;

				// _mm_storeu_ps(xy, temp2);
				// std::cout << "float temp2 : " << xy[0] << std::endl;
				// std::cout << "float (hx/HXRES)-0.5 : " << (((float)hx/(float)HXRES)-0.5) << std::endl;

				//__m128 vm = _mm_set1_ps(m);
				// xy[4] ;

				// _mm_storeu_ps(xy, vm);
				// std::cout << "float vm : " << xy[0] << std::endl;
				// std::cout << "float (m) : " << (m) << std::endl;

				// xy[4] ;

				// _mm_storeu_ps(xy, constant_four);
				// std::cout << "float constant_four : " << xy[0] << std::endl;
				// std::cout << "float constant_four - real : " << 4.0 << std::endl;


				//temp3 = _mm_div_ps( constant_four,  vm);
				// xy[4] ;

				// _mm_storeu_ps(xy, temp3);
				// std::cout << "float temp3 : " << xy[0] << std::endl;
				// std::cout << "float (4.0/m) : " << (4.0/m) << std::endl;

				temp4 = _mm_div_ps( vPY,  temp3);

				temp5 = _mm_add_ps( temp2,  temp4);
				//temp6 = _mm_div_ps(constant_four_f, vm);

				__m128 vcy = _mm_mul_ps( temp5,  temp6);
				float yy[4] ;

				_mm_storeu_ps(yy, vcy);
				// std::cout << "float xy : " << xy[0] << std::endl;
				// std::cout << "float cx : " << cx << std::endl << std::endl;

				if (!member(xy[2], yy[2], iterations)){
					/* Point is not a member, colour based on number of iterations before escape */
					int i=(iterations%40) - 1;
					int b = i*3;
					screen->putpixel(hx, hy, pal[b], pal[b+1], pal[b+2]);
					//std::cout << "-";
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
	
	sleep(2);
#ifdef TIMING
	std::cout << "Total executing time " << total_time << " microseconds\n";
#endif
	std::cout << "Clean Exit"<< std::endl;

}
