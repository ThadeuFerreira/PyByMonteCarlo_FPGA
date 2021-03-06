// ConsoleApplication1.cpp : define o ponto de entrada para o aplicativo do console.
//

#include "stdafx.h"
#include <cmath>
#include <iostream>
#include <iostream>
#include <list>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <functional>

static std::list<double> globalUniformDistributionList; // lista com distribuição uniforme - usando Mersenne Twister
static std::list<double> globalNormalDistributionList;
static std::list<double> globalAssetPricesList; // lista com os preços das ações
static std::list<double> globalPayoffs; // lista com os preços das ações

#define SIZE 1024*1024*64

using namespace std;

#define W 32
#define R 16
#define P 0
#define M1 13
#define M2 9
#define M3 5

#define MAT0POS(t,v) (v^(v>>t))
#define MAT0NEG(t,v) (v^(v<<(-(t))))
#define MAT3NEG(t,v) (v<<(-(t)))
#define MAT4NEG(t,b,v) (v ^ ((v<<(-(t))) & b))

#define V0            STATE[state_i                   ]
#define VM1           STATE[(state_i+M1) & 0x0000000fU]
#define VM2           STATE[(state_i+M2) & 0x0000000fU]
#define VM3           STATE[(state_i+M3) & 0x0000000fU]
#define VRm1          STATE[(state_i+15) & 0x0000000fU]
#define VRm2          STATE[(state_i+14) & 0x0000000fU]
#define newV0         STATE[(state_i+15) & 0x0000000fU]
#define newV1         STATE[state_i                 ]
#define newVRm1       STATE[(state_i+14) & 0x0000000fU]

#define FACT 2.32830643653869628906e-10

static unsigned int state_i = 0;
static unsigned int STATE[R];
static unsigned int z0, z1, z2;

void InitWELLRNG512a(unsigned int seed) {
	STATE[0] = seed + 72852922; // Numeros "aleatorios" arbritários para inicialização 
	STATE[1] = seed + 41699578;
	STATE[2] = seed + 56707026;
	STATE[3] = seed + 33717249;
	STATE[4] = seed + 18306974;
	STATE[5] = seed + 30824004;
	STATE[6] = seed + 42901955;
	STATE[7] = seed + 80465302;
	STATE[8] = 94968136;
	STATE[9] = 41480876;
	STATE[10] = 57870066;
	STATE[11] = 37220400;
	STATE[12] = 14597146;
	STATE[13] = 1165159;
	STATE[14] = 99349121;
	STATE[15] = 68083911;
}

double WELLRNG512a(void) {
	z0 = VRm1;
	z1 = MAT0NEG(-16, V0) ^ MAT0NEG(-15, VM1);
	z2 = MAT0POS(11, VM2);
	newV1 = z1 ^ z2;
	newV0 = MAT0NEG(-2, z0) ^ MAT0NEG(-18, z1) ^ MAT3NEG(-28, z2) ^ MAT4NEG(-5, 0xda442d24U, newV1);
	state_i = (state_i + 15) & 0x0000000fU;
	return ((double)STATE[state_i]) * FACT;
}

static unsigned long x = 123456789, y = 362436069, z = 521288629;

unsigned long xorshf96(void) {          //period 2^96-1
	unsigned long t;
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;

	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;

	return z;
}

int main()
{
	clock_t beginClock = clock();
	mt19937::result_type seed = time(0);
	auto dice_rand = std::bind(std::uniform_real_distribution<double>(0, 1),
		std::mt19937(seed));
	int accum = 0;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < SIZE -1; j+=2)
		{
			double x1 = dice_rand();
			double y1 = dice_rand();
			

			if (sqrt((x1*x1) + (y1*y1))<1.0f) {
				accum = accum + 1;				
			}
		}

		cout << std::setprecision(9) << 8*(double)accum / (SIZE -1) << endl;
		accum = 0;
	}
	clock_t endClock = clock();
	double time_spent = (double)(endClock - beginClock) / CLOCKS_PER_SEC;
	cout << std::setprecision(9) << "Time Mersenne = " << time_spent << endl;
	beginClock = clock();
	InitWELLRNG512a(0);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < SIZE - 1; j += 2)
		{
			double x1 = xorshf96()*FACT;
			double y1 = xorshf96()*FACT;


			if (sqrt((x1*x1) + (y1*y1))<1.0f) {
				accum = accum + 1;
			}
		}

		cout << std::setprecision(9)<< 8 * (double)accum / (SIZE - 1) << endl;
		accum = 0;
	}
	endClock = clock();
	time_spent = (double)(endClock - beginClock) / CLOCKS_PER_SEC;
	cout << std::setprecision(9) << "Time XOR Shift = " << time_spent << endl;

	beginClock = clock();
	InitWELLRNG512a(0);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < SIZE - 1; j += 2)
		{
			double x1 = WELLRNG512a();
			double y1 = WELLRNG512a();


			if (sqrt((x1*x1) + (y1*y1))<1.0f) {
				accum = accum + 1;
			}
		}

		cout << std::setprecision(9) << 8 * (double)accum / (SIZE - 1) << endl;
		accum = 0;
	}
	endClock = clock();
	time_spent = (double)(endClock - beginClock) / CLOCKS_PER_SEC;
	cout << std::setprecision(9) << "Time WELL = " << time_spent << endl;
	cin >> accum;
    return 0;
}

