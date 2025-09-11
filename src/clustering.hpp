/*
clustering.hpp
输入文件数据内容:
  uint32_t         序列数         1
  vector<size_t>   packed数据偏移 readsCount
  vector<size_t>   fasta数据偏移  readsCount
  分组数据 (128*readsCount条记录)
    vector<uint32_t> 序列对应的代表序列 readsCount
    ... (共128组)
    vector<uint32_t> 序列对应的代表序列 readsCount
  packed数据 (readsCount条记录)
    uint32_t 数据长度 1
    uint32_t 压缩数据 (数据长度+31)/32*5
  fasta数据 (readsCount条记录)
    string 序列名 1
    string 序列 1
长度<0xFFFF 数量<0x7FFFFFFF
用法:
ngia2 clustering -p packed文件 -r result文件 -i identity值
2025-08-04 by 鞠震
*/

#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include <iostream>  // cout
#include <vector>  // vector
#include <cstdint>  // int32_t
#include <fstream>  // ifstream
#include <cstring>  // memset
#include <cuda.h>  // cuda
#include "timer.hpp"  // timer
#include "parser.hpp"  // parser

namespace Clustering {  // 命名空间

//--------数据--------//
struct Option {  // 输入参数
  std::string packedFile;  // packed文件
  std::string resultFile;  // result文件
  uint32_t identity;  // identity
};

//--------函数--------//
// init 初始化 ok
void init(int argc, char **argv, Option &option) {
  {  // 解析命令行
    Parser::Parser parser;  // 解析器
    parser.add(true, "-p", "packed file (string)", "");  // packed
    parser.add(true, "-r", "result file (string)", "");  // result
    parser.add(true, "-i", "identity (1-99)", "");  // identity
    parser.parse(argc, argv);  // 解析命令行
    option.packedFile = parser.getValue<std::string>("-p");  // packed文件
    option.resultFile = parser.getValue<std::string>("-r");  // result文件
    option.identity = parser.getValue<uint32_t>("-i");  // 相似度
  }
  bool checkPassed = true;  // 校验是否通过
  {  // 校验参数
    std::ifstream packedFile(option.packedFile);  // packed文件
    if (!packedFile.good()) {  // 没有输入文件
      std::cout << option.packedFile << " does not exists.\n";
      checkPassed = false;
    }
    if (option.identity<1 || option.identity>99) {  // 相似度溢出
      std::cout << "identity should be 1-99\n";
      checkPassed = false;
    }
  }
  cudaDeviceProp prop;  // 显卡属性
  {  // 配置显卡 export CUDA_VISIBLE_DEVICES=0 指定GPU
    cudaSetDevice(0);  // 默认用第一个GPU
    cudaGetDeviceProperties(&prop, 0);  // GPU属性
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 共享内存 缓存优先
    cudaDeviceSynchronize();  // 激活GPU
    if (cudaGetLastError() != cudaSuccess) {  // 校验失败
      std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
      checkPassed = false;
    }
  }
  if (!checkPassed) { exit(0); }  // 校验失败
  std::cout << "use gpu:\t" << prop.name << "\n";
  std::cout << "packed file:\t" << option.packedFile << "\n";
  std::cout << "result file:\t" << option.resultFile << "\n";
  std::cout << "identity:\t" << option.identity << "\n";
}

// 不要改内外循环 寄存器使用会变少
// 不要数据预取 或操作指针 用线程数掩盖延迟
// kernel_dynamic 动态规划
template <uint32_t entropy, uint32_t tabsize>  // 熵 字母表大小
__global__ void __launch_bounds__(64, 1)  // maxThread/block, minBlock/SM
    kernel_dynamic(uint32_t *reads, size_t *offsets, uint32_t *jobs,
                   const uint32_t jobCount, uint32_t *cluster,
                   const float threshold) {
  uint32_t index = blockDim.x * blockIdx.x + threadIdx.x; // 线程编号
  if (index >= jobCount) {                                // 超出范围
    return;
  }
  __restrict__ uint32_t *represent = &reads[offsets[jobs[index * 2 + 0]]];
  __restrict__ uint32_t *read = &reads[offsets[jobs[index * 2 + 1]]];
  uint32_t length1 = represent[0];    // 代表序列长度
  uint32_t length2 = read[0];         // 剩余序列长度
  uint32_t netLength1 = represent[1]; // 代表序列净长度
  uint32_t netLength2 = read[1];      // 剩余序列净长度
  uint32_t lines[2048];               // 每行结果 别赋初值 开销太大
  memset(lines, 0xFF, (netLength1 + 31) / 32 * sizeof(uint32_t)); // 1:不匹配
  uint32_t Rows[entropy] = {0}; // 从行取的32个碱基/氨基酸
  uint32_t Cols[entropy] = {0}; // 从列取的32个碱基/氨基酸
  uint32_t matchs[tabsize] = {0}; // 匹配的碱基/氨基酸 寄存器 1匹配 0不匹配
  uint32_t lsft = ceil((float)length2 - (float)length2 * threshold); // 左偏移
  lsft = (lsft + 31) / 32 * 32;                                      // 32对齐
  uint32_t rsft = ceil((float)length1 - (float)length2 * threshold); // 右偏移
  rsft = rsft + 33;                                                  // 32补全
  // 计算
  for (uint32_t i = 0; i < netLength2; i += 32) { // 遍历列
    uint32_t carrys = 0;                          // 进位
    for (uint32_t e = 0; e < entropy; e++)
      Cols[e] = read[2 + (i >> 5) * entropy + e];
    uint32_t jstart = max((int32_t)i - (int32_t)lsft, (int32_t)0); // 开始
    uint32_t jend =
        min((int32_t)i + (int32_t)rsft, (int32_t)netLength1); // 结束
    for (uint32_t j = jstart; j < jend; j += 32) {            // 遍历行
      for (uint32_t e = 0; e < entropy; e++)
        Rows[e] = represent[2 + (j >> 5) * entropy + e];
      for (uint32_t k = 0; k < tabsize; k++) { // 预生成match
        uint32_t match = 0xFFFFFFFF;
        for (uint32_t e = 0; e < entropy; e++)
          match &= Rows[e] ^ 0xFFFFFFFF + (k >> e & 1);
        matchs[k] = match;
      }
      uint32_t row = lines[j >> 5];       // 上一行结果
      for (uint32_t k = 0; k < 32; k++) { // 32*32的核心
        uint32_t order = 0;
        for (uint32_t e = 0; e < entropy; e++)
          order += (Cols[e] >> k & 1) << e;
        uint32_t match = matchs[order]; // 匹配上的碱基/氨基酸
        uint32_t carry = carrys & 1;    // 进位
        uint32_t term0 = row & match;
        uint32_t term1 = row & (~match);
        uint32_t carryRow = row + carry;
        carry = carryRow < row; // 是否发生进位
        carryRow += term0;
        carry |= carryRow < term0; // 是否发生进位
        row = carryRow | term1;
        carrys = (carrys >> 1) + (carry << 31); // 写回进位
      }
      lines[j >> 5] = row;
    }
  }
  { // 统计结果
    uint32_t sum = 0;
    for (uint32_t i = 0; i < netLength1; i += 32)
      sum += 32 - __popc(lines[i >> 5]);
    sum -= min((netLength1 + 31) / 32 * 32 - netLength1,
               (netLength2 + 31) / 32 * 32 - netLength2);
    uint32_t cutoff = ceil((float)length1 * threshold);
    if (sum >= cutoff) { // 不用优化 没第二个分支 要返回了
      cluster[jobs[index * 2 + 1]] = jobs[index * 2 + 0]; // 写入聚类结果
    }
  }
}

// clustering 利用局部敏感哈希快速聚类
void clustering(const Option &option, std::vector<uint32_t> &results) {
  // uint32_t readsCount = 0;  // 序列数
  // std::vector<uint32_t> preGroup(0);  // 预分组
  // size_t *offsets = NULL;  // 序列偏移
  // uint32_t *reads = NULL;  // 序列数据
  // {  // 读数据
  //   std::ifstream packedFile(option.packedFile);  // packed文件
  //   packedFile.read((char *)&entropy, sizeof(uint32_t));  // 序列的熵
  //   packedFile.read((char *)&readsCount, sizeof(uint32_t));  // 序列数
  //   packedFile.read((char *)&signedCount, sizeof(uint32_t));  // 签名尺寸
  //   readLengths.assign(readsCount, 0);  // 初始化
  //   packedFile.seekg(sizeof(uint32_t)*readsCount, std::ios::cur);  // 跳序列名
  //   packedFile.read((char *)readLengths.data(), sizeof(uint32_t)*readsCount);
  //   cudaMallocManaged(&offsets, sizeof(size_t)*(readsCount+1));  // 偏移
  //   cudaMemAdvise(offsets, sizeof(size_t)*(readsCount+1), cudaMemAdviseSetReadMostly, 0);
  //   packedFile.read((char *)offsets, sizeof(size_t)*(readsCount+1));  // 偏移
  //   preGroup.assign((size_t)readsCount * signedCount, 0);  // 预聚类
  //   packedFile.seekg(sizeof(size_t)*(readsCount-1), std::ios::cur);  // 移hash
  //   packedFile.read((char *)preGroup.data(), sizeof(uint32_t)*preGroup.size());
  //   cudaMallocManaged  #pragma omp for schedule(dynamic, 1)  // 每个线程负责一组签名(&reads, offsets[readsCount] - offsets[0]); // 打包数据
  //   cudaMemAdvise(reads, offsets[readsCount]-offsets[0], cudaMemAdviseSetReadMostly, 0);
  //   packedFile.read((char *)reads, offsets[readsCount]-offsets[0]); // 读打包
  //   packedFile.close();  // 读文件完成
  //   size_t position = offsets[0];  // 偏移的起始位置
  //   for (uint32_t i=0; i<readsCount; i++) {  // 字节位置转为uint32_t偏移
  //     offsets[i] = (offsets[i]-position)/sizeof(uint32_t);
  //   }
  //   std::cout << "reads count:\t" << readsCount << "\n";  // 序列数
  //   std::cout <  #pragma omp for schedule(dynamic, 1)  // 每个线程负责一组签名< "longest:\t" << readLengths.front() << "\n";  // 长
  //   std::cout << "shortest:\t" << readLengths.back() << "\n";  // 短
  // }
  // const float threshold = (float)option.identity/100.0f;  // 相似度阈值
  // const uint32_t loopCount = signedCount;  // 循环分组次数
  // uint32_t *cluster = NULL;  // 聚类结果
  // uint32_t *jobs = NULL;  // 比对任务
  // {  // 聚类过程start
  //   cudaMallocManaged(&cluster, sizeof(uint32_t)*readsCount);  // 聚类结果
  //   memset(cluster, 0xFF, sizeof(uint32_t)*readsCount);  // 最大值是未聚类
  //   cudaMallocManaged(&jobs, sizeof(uint32_t)*readsCount*2);  // 剩余序列
  //   memset(jobs, 0, sizeof(uint32_t)*readsCount*2);  // 最初没有任务
  //   // minHash算法的核心
  //   uint32_t jobCount = 0;                              // 任务数是0
  //   std::cout << "clustering:\n";                       // 开始聚类
  //   for (uint32_t loop = 0; loop < loopCount; loop++) { // 重复分组比对过程
  //     std::cout << "\r" << loop + 1 << "/" << loopCount << std::flush;
  //     // 序列分组1
  //     jobCount = 0;
  //     uint32_t *preGroupLoop = preGroup.data() + (size_t)loop * readsCount;
  //     uint32_t rep = 0, job = 0;  // 代表序列 任务序列
  //     for (uint32_t i = 0; i < readsCount; i++) { // 遍历所有序列的签名
  //       if (preGroupLoop[i] > 0x7FFFFFFF) {
  //         rep = preGroupLoop[i] & 0x7FFFFFFF;
  //         continue;
  //       }
  //       job = preGroupLoop[i];
  //       if (cluster[job] == 0xFFFFFFFF) {  // 还没聚类就聚一下
  //         jobs[jobCount * 2 + 0] = rep;  // 代表序列
  //         jobs[jobCount * 2 + 1] = job;  // 任务序列
  //         jobCount += readLengths[rep] * threshold < readLengths[job];
  //       }
  //       rep = job;  // 长度与相似度同时最接近的序列
  //     }
  //     // 序列比对
  //     cudaMemPrefetchAsync(cluster, sizeof(uint32_t) * jobCount, 0);  // toGPU
  //     cudaMemPrefetchAsync(jobs, sizeof(uint32_t) * jobCount * 2, 0); // toGPU
  //     if (entropy == 3)
  //       kernel_dynamic<3, 5><<<(jobCount + 63) / 64, 64>>>(
  //           reads, offsets, jobs, jobCount, cluster, threshold); // 基因
  //     if (entropy == 5)
  //       kernel_dynamic<5, 23><<<(jobCount + 63) / 64, 64>>>(
  //           reads, offsets, jobs, jobCount, cluster, threshold); // 蛋白
  //     cudaMemPrefetchAsync(cluster, sizeof(uint32_t) * readsCount,
  //                          cudaCpuDeviceId, 0); // toHost
  //     cudaStreamSynchronize(0);                 // 等数据传输完成
  //   }
  //   std::cout << "\n";
  // }  // 聚类过程end


// 这里就截止了-------------------------------------------------------------------
// return;
// 这里就截止了-------------------------------------------------------------------


  // {                                             // 生成结果start
  //   for (uint32_t i = 0; i < readsCount; i++) { // 边缘节点 追溯代表序列
  //     uint32_t rep = cluster[i];
  //     if (rep != 0xFFFFFFFF) { // 如果不是代表序列 就追溯代表序列
  //       while (cluster[rep] != 0xFFFFFFFF)
  //         rep = cluster[rep];
  //     }
  //     cluster[i] = rep;
  //   }
  //   results.assign(readsCount, 0); // 聚类结果
  //   cudaMemcpy(results.data(), cluster, sizeof(uint32_t) * readsCount,
  //              cudaMemcpyDeviceToHost); // 拷贝结果回内存
  // }                                     // 生成结果end
  // cudaFree(offsets);
  // cudaFree(reads);
  // cudaFree(cluster);
  // cudaFree(jobs);
}

//--------主函数--------//
void clustering(int argc, char **argv) {
  Timer::Timer timer;  // 计时器
  timer.printStamp();  // 时间戳

  Option option = { packedFile:"", resultFile:"", identity:0 };  // 选项
  init(argc, argv, option);  // 初始化
  std::vector<uint32_t> results(0);  // 聚类结果
  clustering(option, results);  // 聚类
  // saveResult(option, results);  // 保存结果

  timer.printStamp();  // 时间戳
  timer.printDuration();  // 耗时
}

}  // namespace Clustering
#endif  // CLUSTERING_HPP

// 优化
// 比对算法优化
// cudaMemAdvise(reads, position-offsets[0], cudaMemAdviseSetReadMostly, 0);
// 数据预取
// -maxrregcount 56 --resource-usage
// cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount, cudaCpuDeviceId,
// 0); cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount, 0);
// cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 共享内存变缓存
// __restrict__
// 常量内存
// 显存内计算remains
// 少量序列用batch

// 4090的SM参数:
// 1536个线程
// 48个warp
// 24个blockhttps://172.16.0.75:809/
// 64K个寄存器
// 八个warp就能隐藏延迟了，四发射，64线程足够
