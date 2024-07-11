#include "func.h"  // 数据结构与函数
#include "timer.h"  // 计时器

int main(int argc, char **argv) {
  Timer::Timer timer;  // 计时
  Option option = {packedFile:"", resultFile:"", identity:0, loopCount:0};
  init(argc, argv, option);  // 初始化
  std::vector<uint32_t> results(0);  // 聚类结果
  clustering(option, results);  // 聚类
  saveResult(option, results);  // 保存结果
  timer.getDuration();  // 耗时
  timer.getTimeNow();  // 时间戳
}
