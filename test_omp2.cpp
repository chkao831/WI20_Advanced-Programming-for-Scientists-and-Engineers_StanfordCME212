#include <omp.h>
#include <cstdio>
#include <cstdlib>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

int main()
{
  auto printer = [](int i) {
    printf("Hello on element %2d from thread %2d\n", i, omp_get_thread_num());
  };
  using enumerator = thrust::counting_iterator<int>;
  thrust::for_each(thrust::omp::par, enumerator(0), enumerator(100), printer);
}
