#include <omp.h>
#include <stdio.h>

#define EXTLARGE

#ifdef SMALL
#define MIMAX 65
#define MJMAX 33
#define MKMAX 33
#endif

#ifdef MIDDLE
#define MIMAX 129
#define MJMAX 65
#define MKMAX 65
#endif

#ifdef LARGE
#define MIMAX 257
#define MJMAX 129
#define MKMAX 129
#endif

#ifdef EXTLARGE
#define MIMAX 513
#define MJMAX 257
#define MKMAX 257
#endif

#define NN 200

static float a[MIMAX][MJMAX][MKMAX][4],
             b[MIMAX][MJMAX][MKMAX][3], c[MIMAX][MJMAX][MKMAX][3];
static float p[MIMAX][MJMAX][MKMAX];
static float wrk1[MIMAX][MJMAX][MKMAX], wrk2[MIMAX][MJMAX][MKMAX];
static float bnd[MIMAX][MJMAX][MKMAX];

static int imax, jmax, kmax;
static float omega;


void
initmt()
{
    int i, j, k;
    
    for (i = 0; i < imax; ++i) {
#pragma omp task firstprivate(i) private(j, k) shared(a, b, c, p, wrk1, bnd)
        {
            for (j = 0; j < jmax; ++j) {
                for (k = 0; k < kmax; ++k) {
                    a[i][j][k][0] = 0.0;
                    a[i][j][k][1] = 0.0;
                    a[i][j][k][2] = 0.0;
                    a[i][j][k][3] = 0.0;
                    b[i][j][k][0] = 0.0;
                    b[i][j][k][1] = 0.0;
                    b[i][j][k][2] = 0.0;
                    c[i][j][k][0] = 0.0;
                    c[i][j][k][1] = 0.0;
                    c[i][j][k][2] = 0.0;
                    p[i][j][k] = 0.0;
                    wrk1[i][j][k] = 0.0;
                    bnd[i][j][k] = 0.0;
                }
            }
        }
#pragma omp taskwait
    }

    for (i = 0; i < imax; ++i) {
#pragma omp task firstprivate(i) private(j, k) shared(a, b, c, p, wrk1, bnd)
        {
            for (j = 0; j < jmax; ++j) {
                for (k = 0; k < kmax; ++k) {
                    a[i][j][k][0] = 1.0;
                    a[i][j][k][1] = 1.0;
                    a[i][j][k][2] = 1.0;
                    a[i][j][k][3] = 1.0 / 6.0;
                    c[i][j][k][0] = 1.0;
                    c[i][j][k][1] = 1.0;
                    c[i][j][k][2] = 1.0;
                    p[i][j][k] = (float) (k * k) /
                                 (float) ((kmax - 1) * (kmax - 1));
                    wrk1[i][j][k] = 0.0;
                    bnd[i][j][k] = 1.0;
                }
            }
        }
    }
}

float
jacobi(int nn)
{
    int i, j, k, n;
    float gosa;
    float s0, ss;
    float local_gosa = 0.0;

#pragma omp parallel shared(a, b, c, p, wrk1, wrk2, bnd, omega, gosa)
    {
#pragma omp single
        {
            for (n = 0; n < nn; ++n) {
                gosa = 0.0;
                for (i = 1; i < imax - 1; ++i) {
#pragma omp task firstprivate(i) private(s0, ss, local_gosa)
                    {
                        local_gosa = 0.0;

                        for (j = 1; j < jmax - 1; ++j) {
                            for (k = 1; k < kmax - 1; ++k) {
                                s0 = a[i][j][k][0] * p[i + 1][j][k] +
                                     a[i][j][k][1] * p[i][j + 1][k] +
                                     a[i][j][k][2] * p[i][j][k + 1] +
                                     b[i][j][k][0] * (p[i + 1][j + 1][k] -
                                                      p[i + 1][j - 1][k] -
                                                      p[i - 1][j + 1][k] +
                                                      p[i - 1][j - 1][k]) +
                                     b[i][j][k][1] * (p[i][j + 1][k + 1] -
                                                      p[i][j - 1][k + 1] -
                                                      p[i][j + 1][k - 1] +
                                                      p[i][j - 1][k - 1]) +
                                     b[i][j][k][2] * (p[i + 1][j][k + 1] -
                                                      p[i - 1][j][k + 1] -
                                                      p[i + 1][j][k - 1] +
                                                      p[i - 1][j][k - 1]) +
                                     c[i][j][k][0] * p[i - 1][j][k] +
                                     c[i][j][k][1] * p[i][j - 1][k] +
                                     c[i][j][k][2] * p[i][j][k - 1] +
                                     wrk1[i][j][k];


                                ss = (s0 * a[i][j][k][3] - p[i][j][k]) *
                                      bnd[i][j][k];
                                local_gosa += ss * ss;

                                wrk2[i][j][k] = p[i][j][k] + omega * ss;
                            }
                        }
#pragma omp atomic
                    gosa += local_gosa;
#pragma omp taskwait
                    }
               }

                for (i = 1; i < imax - 1; ++i) {
#pragma omp task firstprivate(i)
                    for (j = 1; j < jmax - 1; ++j) {
                        for (k = 1; k < kmax - 1; ++k)
                            p[i][j][k] = wrk2[i][j][k];
                    }
#pragma omp taskwait
                }
            }
        }
    }

    return gosa;
}

int
main()
{
    int i, j, k;
    float gosa;
    double cpu0, cpu1, cpu2, cpu3, nflop, xmflops2, score;

    omega = 0.8;
    imax = MIMAX - 1;
    jmax = MJMAX - 1;
    kmax = MKMAX - 1;

    cpu2 = omp_get_wtime();
    
    initmt();
    
    cpu3 = omp_get_wtime();
    
//    printf("mimax = %d mjmax = %d mkmax = %d\n", MIMAX, MJMAX, MKMAX);
//    printf("imax = %d jmax = %d kmax =%d\n", imax, jmax, kmax);


    cpu0 = omp_get_wtime();

    gosa = jacobi(NN);

    cpu1 = omp_get_wtime();

    nflop = (kmax - 2) * (jmax - 2) * (imax - 2) * 34;

    if (cpu1 != 0.0) {
        xmflops2 = nflop / cpu1 * 1.0e-6 * (float) NN;
    }

    score = xmflops2 / 32.27;

      printf("cpu : %f sec.\n", cpu1 - cpu0);
//    printf("init : %f sec.\n", cpu3 - cpu2);
//    printf("Loop executed for %d times\n", NN);
//    printf("Gosa : %e \n", gosa);
//    printf("MFLOPS measured : %f\n", xmflops2);
//    printf("Score based on MMX Pentium 200MHz : %f\n", score);
//    printf("\n");

    return (0);
}
