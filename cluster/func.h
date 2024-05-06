#ifndef __FUNCH__
#define __FUNCH__

#include <string>  // std::string
#include <vector>  // vector

//--------数据结构--------//
struct Option {  // 输入参数
  std::string packedFile;  // packed文件
  std::string resultFile;  // result文件
  int32_t identity;  // 相似度
};

//--------声明函数--------//
void init(int argc, char **argv, Option &option);  // 读输入
void clustering(const Option &option, std::vector<int32_t> &result);
#endif  // __FUNCH__
