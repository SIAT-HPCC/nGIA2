#include "func.h"  // 数据结构与函数
#include "timer.h"  // 计时器

int main(int argc, char **argv) {
  Timer::Timer timer;  // 计时
  Option option;  // 输入参数
  init(argc, argv, option);  // 读输入 初始化显卡
  std::vector<int32_t> result;
  clustering(option, result);  // 聚类
  timer.getDuration();  // 耗时
}
