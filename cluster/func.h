#ifndef __FUNCH__
#define __FUNCH__

#include <string>  // std::string
#include <vector>  // vector

//--------数据结构--------//
struct Option {  // 输入参数
  std::string packedFile;  // packed文件
  std::string resultFile;  // result文件
  uint32_t identity;  // 相似度
  uint32_t mode;  // 聚类模式
};

//--------声明函数--------//
void init(int argc, char **argv, Option &option);  // 读输入
void clusteringPrecise(const Option &option, std::vector<uint32_t> &results);
void clusteringFast(const Option &option, std::vector<uint32_t> &results);
void conutResult(const Option &option, const std::vector<uint32_t> &results);
#endif  // __FUNCH__
