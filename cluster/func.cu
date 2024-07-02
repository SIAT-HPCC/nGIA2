// func.cu
// Endian:little 默认小端
// 为了方便 统一用uint32_t 避免有符号溢出
// 很多冗余优化 是为减少不可预期的编译行为 减小耗时波动
#include "func.h"        // 数据结构与函数
#include "parser.h"      // 解析器
#include <algorithm>     // stable_sort
#include <fstream>       // fstream
#include <iostream>      // cout
#include <omp.h>         // openmp
#include <unordered_map> // unordered_map
#include <vector>        // vector

// init 初始化 ok
void init(int argc, char **argv, Option &option) {
  {                        // 解析命令行
    Parser::Parser parser; // 解析器
    parser.add("packed", "-p", "packed file", "string", "", true);
    parser.add("result", "-r", "result file", "string", "", true);
    parser.add("identity", "-i", "identity 1-99", "int32_t", "", true);
    parser.add("loop", "-l", "loop count 64-128", "int32_t", "64", false);
    if (!parser.parse(argc, argv)) { // 解析失败 退出
      exit(0);
    }
    option.packedFile = parser.getString("packed");  // packed文件
    option.resultFile = parser.getString("result");  // result文件
    option.identity = parser.getInt32_t("identity"); // 相似度
    option.loopCount = parser.getInt32_t("loop");    // 循环次数
  }
  {                                              // 校验参数
    std::ifstream packedFile(option.packedFile); // packed文件
    if (!packedFile.is_open()) {                 // 没有输入文件 退出
      std::cout << option.packedFile << " not exists\n";
      exit(0);
    }
    packedFile.close();
    std::cout << "packed:\t\t" << option.packedFile << "\n"; // 打印信息
    std::cout << "result:\t\t" << option.resultFile << "\n"; // 打印信息
    if (option.identity < 1 || option.identity > 99) { // 相似度溢出 退出
      std::cout << "identity should be 1-99\n";
      exit(0);
    }
    std::cout << "identity:\t" << option.identity << "\n"; // 打印信息
    if (option.loopCount < 64 || 128 < option.loopCount) {
      std::cout << "loop count should be 64-128\n";
      exit(0);
    }
    std::cout << "loop count:\t" << option.loopCount << "\n"; // 打印信息
  }
  {                      // 配置显卡 export CUDA_VISIBLE_DEVICES=0 指定GPU
    cudaDeviceProp prop; // 显卡属性
    if (cudaGetDeviceProperties(&prop, 1) != cudaSuccess) { // 找不到显卡 退出
      std::cout << "find no GPU \n";
      auto err = cudaGetLastError(); // 报错信息
      std::cout << cudaGetErrorString(err) << "\n";
      exit(0);
    }
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); // 共享内存 缓存优先
    cudaDeviceSynchronize();                         // 激活GPU
    std::cout << "use GPU:\t" << prop.name << "\n";
  }
}

// 不要改内外循环 寄存器使用会变少
// 不要数据预取 或操作指针 用线程数掩盖延迟
// kernel_dynamic 动态规划
template <uint32_t entropy, uint32_t tabsize> // 熵 字母表大小
__global__ void __launch_bounds__(64, 1)      // maxThread/block, minBlock/SM
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
  uint32_t entropy = 0;                                      // 数据的熵
  uint32_t readsCount = 0;                                   // 序列数
  uint32_t signedCount = 0;                                  // 签名尺寸
  std::vector<uint32_t> readLengths(0);                      // 序列长度
  std::vector<uint32_t> preGroup(0);                         // 预分组
  size_t *offsets = NULL;                                    // 序列偏移
  uint32_t *reads = NULL;                                    // 序列数据
  {                                                          // 读数据
    std::ifstream packedFile(option.packedFile);             // packed文件
    packedFile.read((char *)&entropy, sizeof(uint32_t));     // 序列的熵
    packedFile.read((char *)&readsCount, sizeof(uint32_t));  // 序列数
    packedFile.read((char *)&signedCount, sizeof(uint32_t)); // 签名尺寸
    readLengths.assign(readsCount, 0);                       // 初始化
    packedFile.seekg(sizeof(uint32_t) * readsCount, std::ios::cur); // 跳序列名
    packedFile.read((char *)readLengths.data(), sizeof(uint32_t) * readsCount);
    cudaMallocManaged(&offsets, sizeof(size_t) * (readsCount + 1)); // 偏移
    cudaMemAdvise(offsets, sizeof(size_t) * (readsCount + 1),
                  cudaMemAdviseSetReadMostly, 0); // 只读不写
    packedFile.read((char *)offsets, sizeof(size_t) * (readsCount + 1)); // 偏移
    preGroup.assign(readsCount * signedCount, 0);                 // 预聚类
    size_t groupOffset = sizeof(uint32_t) * (3 + readsCount * 2); // hash位置
    groupOffset += sizeof(size_t) * readsCount * 2;
    packedFile.seekg(groupOffset, std::ios::beg); // 移到hashTable处
    packedFile.read((char *)preGroup.data(),
                    sizeof(uint32_t) * preGroup.size());         // 预聚类
    cudaMallocManaged(&reads, offsets[readsCount] - offsets[0]); // 打包数据
    cudaMemAdvise(reads, offsets[readsCount] - offsets[0],
                  cudaMemAdviseSetReadMostly, 0); // 建议 只读不写
    packedFile.seekg(offsets[0], std::ios::beg);  // 移位
    packedFile.read((char *)reads, offsets[readsCount] - offsets[0]); // 打包
    packedFile.close();                         // 读文件完成
    size_t position = offsets[0];               // 偏移的起始位置
    for (uint32_t i = 0; i < readsCount; i++) { // 字节位置转为uint32_t偏移
      offsets[i] = (offsets[i] - position) / sizeof(uint32_t);
    }
    const std::string types[6] = {"", "", "", "gene", "", "protein"};
    std::cout << "data type:\t" << types[entropy] << "\n"; // 文件类型
    std::cout << "reads count:\t" << readsCount << "\n";   // 序列数
    std::cout << "longest:\t" << reads[0] << "\n";         // 长
    std::cout << "shortest:\t" << reads[offsets[readsCount - 1]] << "\n"; // 短
  }
  float threshold = (float)option.identity / 100.0f; // 相似度阈值
  uint32_t loopCount = option.loopCount;             // 循环分组次数
  uint32_t *cluster = NULL;                          // 聚类结果
  uint32_t *jobs = NULL;                             // 比对任务
  {                                                  // 聚类过程start
    cudaMallocManaged(&cluster, sizeof(uint32_t) * readsCount); // 聚类结果
    memset(cluster, 0xFF, sizeof(uint32_t) * readsCount); // 最大值未聚类
    cudaMallocManaged(&jobs, sizeof(uint32_t) * readsCount * 2); // 剩余序列
    memset(jobs, 0, sizeof(uint32_t) * readsCount * 2); // 最初没有任务
    // minHash算法的核心
    uint32_t jobCount = 0;                              // 任务数是0
    std::cout << "clustering:\n";                       // 开始聚类
    bool jobActive = false;                             // 激活任务
    for (uint32_t loop = 0; loop < loopCount; loop++) { // 重复分组比对过程
      std::cout << "\r" << loop + 1 << "/" << loopCount << std::flush;
      jobCount = 0;
      uint32_t *preGroupLoop = preGroup.data() + loop * readsCount; // 当前循环
      uint32_t rep = 0, job = 0;                  // 代表序列 任务序列
      for (uint32_t i = 0; i < readsCount; i++) { // 遍历所有序列的签名
        if (preGroupLoop[i] >> 31 == 1) {         // 找到代表序列
          rep = preGroupLoop[i] & 0x7FFFFFFF;
          jobActive = true;
        } else if (jobActive) {
          job = preGroupLoop[i];
          if (readLengths[rep] * threshold < readLengths[job]) { // 长度靠谱
            jobs[jobCount * 2 + 0] = rep;                        // 代表序列
            jobs[jobCount * 2 + 1] = job;                        // 任务序列
            jobCount += cluster[job] == 0xFFFFFFFF; // 没聚类 任务成功
          } else {
            jobActive = false;
          }
        }
      }
      // 序列比对
      cudaMemPrefetchAsync(cluster, sizeof(uint32_t) * jobCount, 0);  // toGPU
      cudaMemPrefetchAsync(jobs, sizeof(uint32_t) * jobCount * 2, 0); // toGPU
      if (entropy == 3)
        kernel_dynamic<3, 5><<<(jobCount + 63) / 64, 64>>>(
            reads, offsets, jobs, jobCount, cluster, threshold); // 基因
      if (entropy == 5)
        kernel_dynamic<5, 23><<<(jobCount + 63) / 64, 64>>>(
            reads, offsets, jobs, jobCount, cluster, threshold); // 蛋白
      cudaMemPrefetchAsync(cluster, sizeof(uint32_t) * readsCount,
                           cudaCpuDeviceId, 0); // toHost
      cudaStreamSynchronize(0);                 // 等数据传输完成
    }
    std::cout << "\n";
  }                                             // 聚类过程end
  {                                             // 生成结果start
    for (uint32_t i = 0; i < readsCount; i++) { // 边缘节点 追溯代表序列
      uint32_t rep = cluster[i];
      if (rep != 0xFFFFFFFF) { // 如果不是代表序列 就追溯代表序列
        while (cluster[rep] != 0xFFFFFFFF)
          rep = cluster[rep];
      }
      cluster[i] = rep;
    }
    results.assign(readsCount, 0); // 聚类结果
    cudaMemcpy(results.data(), cluster, sizeof(uint32_t) * readsCount,
               cudaMemcpyDeviceToHost); // 拷贝结果回内存
  }                                     // 生成结果end
  cudaFree(offsets);
  cudaFree(reads);
  cudaFree(cluster);
  cudaFree(jobs);
}

// countResult 统计结果
void saveResult(const Option &option, const std::vector<uint32_t> &results) {
  uint32_t readsCount = results.size();        // 序列数
  std::vector<uint64_t> orders(readsCount, 0); // 32bit:32bit 代表序列:任务序列
  { // 计算结果文件的写入顺序
    for (uint32_t i = 0; i < readsCount; i++) { // 遍历结果
      uint32_t rep = results[i];                // 记录了代表序列
      if (rep == 0xFFFFFFFF)
        rep = i; // 自己就是代表序列
      orders[i] = (((uint64_t)rep) << 32) + (uint64_t)i;
    }
    std::stable_sort(orders.begin(), orders.end()); // 排序
  }
  std::vector<size_t> fastaOffsets(readsCount, 0);  // 输入文件的偏移
  std::vector<size_t> resultOffsets(readsCount, 0); // 结果文件的偏移
  { // 计算结果文件的偏移
    std::vector<uint32_t> nameLengths(readsCount, 0);     // 序列名长度
    std::vector<uint32_t> readLengths(readsCount, 0);     // 序列长度
    std::ifstream fastaFile(option.packedFile);           // 输入文件
    fastaFile.seekg(sizeof(uint32_t) * 3, std::ios::beg); // 跳到长度位置
    fastaFile.read((char *)nameLengths.data(), sizeof(uint32_t) * readsCount);
    fastaFile.read((char *)readLengths.data(), sizeof(uint32_t) * readsCount);
    fastaFile.seekg(sizeof(size_t) * readsCount, std::ios::cur); // 跳到偏移位置
    fastaFile.read((char *)fastaOffsets.data(), sizeof(size_t) * readsCount);
    size_t offset = 0;
    uint32_t count = 0; // 代表序列的数量
    for (uint32_t i = 0; i < readsCount; i++) {
      uint32_t rep = (orders[i] >> 32) & 0xFFFFFFFF; // 代表序列
      uint32_t job = orders[i] & 0xFFFFFFFF;         // 任务序列
      resultOffsets[job] = offset;
      if (job == rep) { // 代表序列 顶格
        offset += nameLengths[job] + readLengths[job] + 2;
        count += 1;
      } else { // 非代表序列 前方加两个空格
        offset += nameLengths[job] + 3;
      }
    }
    fastaFile.close();
    std::cout << "cluster:\t" << count << "\n";
  }
  std::ofstream(option.resultFile).close(); // 先清空输出文件
#pragma omp parallel
  {                                             // 写入结果文件
    std::ifstream fastaFile(option.packedFile); // 输入
    std::ofstream resultFile(option.resultFile, std::ios::in); // 输出
    std::string name = "", read = ""; // 序列名 序列数据
#pragma omp master
    { std::cout << "save:\t." << std::flush; } // 打印进度
#pragma omp for
    for (uint32_t i = 0; i < readsCount; i++) {          // 写入结果
      uint32_t rep = (orders[i] >> 32) & 0xFFFFFFFF;     // 代表序列
      uint32_t job = orders[i] & 0xFFFFFFFF;             // 任务序列
      fastaFile.seekg(fastaOffsets[job], std::ios::beg); // 跳到输入开始
      getline(fastaFile, name);
      name += "\n"; // 读序列名
      getline(fastaFile, read);
      read += "\n";                                        // 读序列数据
      resultFile.seekp(resultOffsets[job], std::ios::beg); // 跳到输出开始
      if (rep == job) {                                    // 代表序列
        resultFile.write((char *)name.data(), name.size());
        resultFile.write((char *)read.data(), read.size());
      } else { // 任务序列
        name = "  " + name;
        resultFile.write((char *)name.data(), name.size());
      }
      if ((i + 1) % (1024 * 1024) == 0)
        std::cout << "." << std::flush; // 打印进度
    }
#pragma omp master
    { std::cout << " finish\n"; }
    fastaFile.close();
    resultFile.close();
  }
}

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
// 24个block
// 64K个寄存器
// 八个warp就能隐藏延迟了，四发射，64线程足够
