#include <omp.h>
#include <cstdio>
#include <cstdlib>

int main()
{
  /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel
  {
    /* Obtain thread number */
    int tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);

    /* Only master thread does this */
    if (tid == 0)
      printf("Number of threads = %d\n", omp_get_num_threads());
  }  /* All threads join master thread and disband */
}
