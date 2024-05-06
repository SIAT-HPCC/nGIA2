/*
  timer.h
  chrono库实现的计时器 支持时间戳 中断计时 用法如下
  Timer::Timer timer;  // 声明
  timer.getTimeNow();  // 打时间戳
  timer.start();  // 开始计时
  timer.pause();  // 暂停计时
  timer.resume();  // 恢复计时
  timer.getDuration();  // 统计耗时
  2024/03/09 by 鞠震
*/

#include <iostream>  // cout
#include <chrono>  // system_clock
#ifndef __TIMERH__  // 防御声明 pragma once 通用性有问题
#define __TIMERH__
//--------类声明--------//
namespace Timer {  // 命名空间
class Timer {  // 计时器
  private:
    std::chrono::steady_clock::time_point stamp;  // 时间戳
    std::chrono::duration<double> duration;  // 耗时 秒
    bool running;  // 正在计时
  public:
    Timer() {  // 构造
      stamp = std::chrono::steady_clock::now();
      duration = std::chrono::duration<double>(0);
      running = true;
    }
    ~Timer() {}  // 析构
    void getTimeNow() {  // 输出当前时间戳
      auto now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(now);
      std::cout << ctime(&time_t);
    }
    void start() {  // 开始计时
      stamp = std::chrono::steady_clock::now();
      duration = std::chrono::duration<double>(0);
      running = true;
    }
    void pause() {  // 暂停计时
      duration += std::chrono::steady_clock::now()-stamp;
      running = false;
    }
    void resume() {  // 恢复计时
      stamp = std::chrono::steady_clock::now();
      running = true;
    }
    void getDuration() {  // 输出耗时
      auto duration_cast = duration;
      if (running) duration_cast += std::chrono::steady_clock::now()-stamp;
      std::cout << duration_cast.count() << " s\n";
    }
};
}
#endif  // __TIMERH__
