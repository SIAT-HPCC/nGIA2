#ifndef __FUNCH__
#define __FUNCH__

#include <string> // std::string
#include <vector> // vector

//--------数据结构--------//
struct Option {           // 输入参数
  std::string packedFile; // packed文件
  std::string resultFile; // result文件
  uint32_t identity;      // identity
  uint32_t loopCount;     // 循环次数
};

//--------声明函数--------//
void init(int argc, char **argv, Option &option);
void clustering(const Option &option, std::vector<uint32_t> &results);
void saveResult(const Option &option, const std::vector<uint32_t> &results);
#endif // __FUNCH__
