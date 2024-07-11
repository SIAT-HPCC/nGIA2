#include <iostream>
#include <vector>
#include <omp.h>

int main() {
  std::cout << "hello\n";
  std::vector<uint32_t> data(100, 0);
  #pragma omp parallel for
  for (uint32_t i=0; i<16; i++) {
    if (omp_get_thread_num() == i)
      data[i] = i;
  }
  for (uint32_t i=0; i<32; i++)
    if (data[i] != 0)
      std::cout << data[i] << "\n";
}
