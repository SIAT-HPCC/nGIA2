#include "func.h"  // 数据结构与函数
#include "timer.h"  // 计时器

int main(int argc, char **argv) {
  Timer::Timer timer;  // 计时
  Option option = {packedFile:"", resultFile:"", identity:0, mode:0};  // 选项
  init(argc, argv, option);  // 读输入 初始化显卡
  std::vector<uint32_t> results(0);  // 聚类结果
  if (option.mode == 0) clusteringPrecise(option, results);  // 精确聚类
  if (option.mode == 1) clusteringFast(option, results);  // 快速聚类
  conutResult(option, results);  // 统计结果
  timer.getDuration();  // 耗时
  timer.getTimeNow();  // 时间戳
}
