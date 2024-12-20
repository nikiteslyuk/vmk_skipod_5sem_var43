#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>

#define SMALL

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

float jacobi(int, int, int, int);
void initmt(int, int, int, int);

static float p[MIMAX][MJMAX][MKMAX];
static float a[MIMAX][MJMAX][MKMAX][4], b[MIMAX][MJMAX][MKMAX][3],
    c[MIMAX][MJMAX][MKMAX][3];
static float bnd[MIMAX][MJMAX][MKMAX];
static float wrk1[MIMAX][MJMAX][MKMAX], wrk2[MIMAX][MJMAX][MKMAX];

static int imax, jmax, kmax;
static float omega;

void initmt(int i_start, int i_end, int jmax, int kmax) {
    int i, j, k;
    for (i = i_start; i <= i_end; i++)
        for (j = 0; j < jmax; ++j)
            for (k = 0; k < kmax; ++k) {
                a[i][j][k][0] = 1.0;
                a[i][j][k][1] = 1.0;
                a[i][j][k][2] = 1.0;
                a[i][j][k][3] = 1.0 / 6.0;
                b[i][j][k][0] = 0.0;
                b[i][j][k][1] = 0.0;
                b[i][j][k][2] = 0.0;
                c[i][j][k][0] = 1.0;
                c[i][j][k][1] = 1.0;
                c[i][j][k][2] = 1.0;
                p[i][j][k] =
                    (float)(k * k) / (float)((kmax - 1) * (kmax - 1));
                wrk1[i][j][k] = 0.0;
                bnd[i][j][k] = 1.0;
                wrk2[i][j][k] = 0.0;
            }
}

float jacobi(int nn, int i_start, int i_end, int size) {
    int i, j, k, n;
    float gosa_local, gosa_global;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    static float send_left[MJMAX][MKMAX], send_right[MJMAX][MKMAX];
    static float recv_left[MJMAX][MKMAX], recv_right[MJMAX][MKMAX];

    for (n = 0; n < nn; ++n) {
        if (i_start > 0) {
            for (j = 0; j < jmax; j++)
                for (k = 0; k < kmax; k++)
                    send_left[j][k] = p[i_start][j][k];
        }
        if (i_end < imax - 1) {
            for (j = 0; j < jmax; j++)
                for (k = 0; k < kmax; k++)
                    send_right[j][k] = p[i_end][j][k];
        }

        MPI_Request reqs[4];
        int req_count = 0;

        if (left != MPI_PROC_NULL) {
            MPI_Isend(&send_left[0][0], jmax * kmax, MPI_FLOAT, left, 0,
                      MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&recv_left[0][0], jmax * kmax, MPI_FLOAT, left, 1,
                      MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Isend(&send_right[0][0], jmax * kmax, MPI_FLOAT, right, 1,
                      MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Irecv(&recv_right[0][0], jmax * kmax, MPI_FLOAT, right, 0,
                      MPI_COMM_WORLD, &reqs[req_count++]);
        }

        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        if (left != MPI_PROC_NULL) {
            for (j = 0; j < jmax; j++)
                for (k = 0; k < kmax; k++)
                    p[i_start - 1][j][k] = recv_left[j][k];
        }
        if (right != MPI_PROC_NULL) {
            for (j = 0; j < jmax; j++)
                for (k = 0; k < kmax; k++)
                    p[i_end + 1][j][k] = recv_right[j][k];
        }

        gosa_local = 0.0f;

        for (i = (i_start > 1 ? i_start : 1);
             i <= (i_end < imax - 2 ? i_end : imax - 2); i++)
            for (j = 1; j < jmax - 1; ++j)
                for (k = 1; k < kmax - 1; ++k) {
                    float s0 = a[i][j][k][0] * p[i + 1][j][k] +
                               a[i][j][k][1] * p[i][j + 1][k] +
                               a[i][j][k][2] * p[i][j][k + 1] +
                               b[i][j][k][0] *
                                   (p[i + 1][j + 1][k] - p[i + 1][j - 1][k] -
                                    p[i - 1][j + 1][k] + p[i - 1][j - 1][k]) +
                               b[i][j][k][1] *
                                   (p[i][j + 1][k + 1] - p[i][j - 1][k + 1] -
                                    p[i][j + 1][k - 1] + p[i][j - 1][k - 1]) +
                               b[i][j][k][2] *
                                   (p[i + 1][j][k + 1] - p[i - 1][j][k + 1] -
                                    p[i + 1][j][k - 1] + p[i - 1][j][k - 1]) +
                               c[i][j][k][0] * p[i - 1][j][k] +
                               c[i][j][k][1] * p[i][j - 1][k] +
                               c[i][j][k][2] * p[i][j][k - 1] + wrk1[i][j][k];

                    float ss =
                        (s0 * a[i][j][k][3] - p[i][j][k]) * bnd[i][j][k];
                    gosa_local += ss * ss;
                    wrk2[i][j][k] = p[i][j][k] + omega * ss;
                }

        for (i = (i_start > 1 ? i_start : 1);
             i <= (i_end < imax - 2 ? i_end : imax - 2); i++)
            for (j = 1; j < jmax - 1; ++j)
                for (k = 1; k < kmax - 1; ++k)
                    p[i][j][k] = wrk2[i][j][k];

        MPI_Allreduce(&gosa_local, &gosa_global, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
    }

    return gosa_global;
}

int main(int argc, char* argv[]) {
    int i, j, k;
    int rank, size;
    double cpu0, cpu1, nflop, xmflops2, score;
    float gosa_local, gosa;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omega = 0.8;
    imax = MIMAX - 1;
    jmax = MJMAX - 1;
    kmax = MKMAX - 1;

    int i_block = imax / size;
    int rest = imax % size;
    int i_start, i_end;
    if (rank < rest) {
        i_block = imax / size + 1;
        i_start = rank * i_block;
        i_end = i_start + i_block - 1;
    } else {
        i_block = imax / size;
        i_start = rest * ((imax / size) + 1) + (rank - rest) * i_block;
        i_end = i_start + i_block - 1;
    }

    initmt(i_start, i_end, jmax, kmax);

    if (rank == 0) {
        printf("mimax = %d mjmax = %d mkmax = %d\n", MIMAX, MJMAX, MKMAX);
        printf("imax = %d jmax = %d kmax = %d\n", imax, jmax, kmax);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cpu0 = MPI_Wtime();

    gosa = jacobi(NN, i_start, i_end, size);

    MPI_Barrier(MPI_COMM_WORLD);
    cpu1 = MPI_Wtime();

    nflop = (kmax - 2) * (jmax - 2) * (imax - 2) * 34;

    if (rank == 0) {
        if (cpu1 != 0.0)
            xmflops2 = nflop / cpu1 * 1.0e-6 * (float)NN;
        else
            xmflops2 = 0.0;

        score = xmflops2 / 32.27;

        printf("%f,\n", cpu1 - cpu0);
        printf("cpu : %f sec.\n", cpu1 - cpu0);
        printf("Loop executed for %d times\n", NN);
        printf("Gosa : %e \n", gosa);
        printf("MFLOPS measured : %f\n", xmflops2);
        printf("Score based on MMX Pentium 200MHz : %f\n", score);
    }

    MPI_Finalize();
    return (0);
}
