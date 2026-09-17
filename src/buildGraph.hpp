/*
buildGraph.hpp
输入文件数据内容:
  uint32_t  seqCount     1
  uint32_t[seqCount]  nameLengths  序列名字长度
  uint32_t[seqCount]  readLengths  序列数据长度
  vector<size_t>  packed数据偏移 readsCount
  vector<size_t>  fasta数据偏移  readsCount
  分组数据  (GROUP_BANDS * readsCount条记录 每条uint32_t=4字节)
    band-major: band b的代表索引行在 signBase+b*seqCount
    rep[band][i]=同签名桶内索引更小的最近序列 组头=自己
    ... (共GROUP_BANDS组)
  packed数据 (readsCount条记录)
    uint32_t 数据长度 1
    uint32_t 压缩数据 (数据长度+31)/32*4
  fasta数据 (readsCount条记录)
    string 序列名 1
    string 序列 1
长度<0xFFFF 数量<0x7FFFFFFF
用法:
ngia3 clustering -p packed文件 -r result文件 -i identity值
2026-07-13 by 鞠震
*/

#ifndef BUILDGRAPH_HPP
#define BUILDGRAPH_HPP

#include <iostream>  // cout
#include <vector>  // vector
#include <cstdint>  // uint32_t
#include <cmath>  // powf, fabsf
#include <fstream>  // ifstream
#include <cstring>  // memset
#include <algorithm>  // sort, unique
#include <parallel/algorithm>  // __gnu_parallel::sort
#include <tuple>  // tuple
#include <utility>  // pair
#include <omp.h>  // openmp
#include <cuda.h>  // cuda
#include "parser.hpp"  // parser
#include "timer.hpp"  // Timer

namespace BuildGraph {  // 命名空间

//--------数据--------//
struct Data{  // 数据
  uint32_t seqCount;  // 序列数(≤2^31-2 打包时已校验)
  std::vector<uint32_t> readLengths;  // 序列数据长度
  size_t *packedOffsets;  // packed数据偏移
  size_t fastaFirst;  // fasta区起点(=打包区终点 原fastaOffsets[0])
  std::vector<uint32_t> repIdx;  // 代表序列索引(每band: rep[i]=同组最近,组内第一条=自己)
  uint32_t *packedReads;  // packed数据
};

//--------函数--------//
//-------- CUDA错误检查辅助: 失败打印并返回false --------//
template <typename T>
inline bool allocManaged(T*& p, size_t bytes, const char* what) {  // 分配
  cudaError_t err = cudaMallocManaged(&p, bytes);  // 分配失败检查
  if (err != cudaSuccess) {  // 分配失败
    std::cout << what << " failed: " << cudaGetErrorString(err) << "\n";
    return false;  // 返回失败
  }
  return true;  // 返回成功
}
inline bool cudaOk(cudaError_t err, const char* what) {  // CUDA错误检查
  if (err != cudaSuccess) {  // 调用失败
    std::cout << what << " failed: " << cudaGetErrorString(err) << "\n";
    return false;  // 返回失败
  }
  return true;  // 返回成功
}
// check 自检
bool check(const std::string& packedFile, const std::string& resultFile,
const uint32_t identity, const uint32_t bands) {
  bool checkPassed = true;  // 每一项校验都通过才进行下一步
  {  // 校验参数
    std::ifstream file(packedFile);
    if (!file.good()) {  // 没有输入文件
      std::cout << packedFile << " does not exists.\n";
      checkPassed = false;
    }
    if (identity<1 || identity>99) {  // 相似度溢出
      std::cout << "identity should be 1-99\n";
      checkPassed = false;
    }
    if (bands<1 || bands>GROUP_BANDS) {  // band数溢出
      std::cout << "bands should be 1-" << GROUP_BANDS << "\n";
      checkPassed = false;
    }
    if (checkPassed) {  // 校验packed头: seqCount必须>0
      uint32_t fileSeqCount = 0;  // 文件中的序列数
      file.read((char*)&fileSeqCount, sizeof(uint32_t));  // 序列数(uint32_t)
      if (!file || fileSeqCount == 0) {  // 文件损坏或0序列
        std::cout << packedFile << " contains 0 sequences.\n";
        checkPassed = false;
      }
    }
  }
  cudaDeviceProp prop;  // 显卡属性
  {  // 配置显卡 export CUDA_VISIBLE_DEVICES=0 指定GPU1
    Timer::Timer t1; cudaSetDevice(0); t1.pause();
    std::cout << "  [cudaSetDevice]\t"; t1.printDuration();
    Timer::Timer t2; cudaGetDeviceProperties(&prop, 0); t2.pause();
    std::cout << "  [cudaGetDeviceProperties]\t"; t2.printDuration();
    Timer::Timer t3;
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 共享内存 缓存优先
    t3.pause();
    std::cout << "  [cudaDeviceSetCacheConfig]\t"; t3.printDuration();
    Timer::Timer t4; cudaDeviceSynchronize(); t4.pause();
    std::cout << "  [cudaDeviceSynchronize]\t"; t4.printDuration();
    if (cudaGetLastError() != cudaSuccess) {  // GPU状态校验
      std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
      checkPassed = false;
    }
  }
  std::cout << "use gpu:\t" << prop.name << "\n";
  std::cout << "packed file:\t" << packedFile << "\n";
  std::cout << "result file:\t" << resultFile << "\n";
  std::cout << "identity:\t" << identity << "\n";
  return checkPassed;
}

// readData 读取数据 bands: 需要用到的band行数
Data readData(const std::string& packedFile, const uint32_t bands) {
  Data inputData{
    .seqCount = 0,  // 序列数
    .readLengths = {},  // 序列数据长度
    .packedOffsets = nullptr,  // packed数据偏移
    .fastaFirst = 0,  // fasta区起点
    .repIdx = {},  // 代表序列索引
    .packedReads = nullptr  // packed数据
  };  // 输入数据
  // 读序列数
  std::ifstream file(packedFile);  // 输入文件
  file.read((char*)&inputData.seqCount, sizeof(uint32_t));  // 序列数(uint32_t)
  const size_t seqCount = inputData.seqCount;  // 用的太多 存一下
  // 跳过nameLengths(聚类流程不使用) 读readLengths
  file.seekg((std::streamoff)seqCount * sizeof(uint32_t), std::ios::cur);  // 跳过
  inputData.readLengths.resize(seqCount);
  file.read((char*)inputData.readLengths.data(), sizeof(uint32_t)*seqCount);
  // 读偏移
  if (!allocManaged(inputData.packedOffsets, sizeof(size_t)*seqCount,
    "cudaMallocManaged(packedOffsets)")) {  // 分配失败
    inputData.seqCount = 0;  // 失败标记 上层据此中止
    return inputData;  // 提前返回
  }
  file.read((char*)inputData.packedOffsets, sizeof(size_t)*seqCount);
  // fasta偏移只需首元素(=打包区终点) 其余跳过
  inputData.fastaFirst = 0;
  file.read((char*)&inputData.fastaFirst, sizeof(size_t));  // 读fastaOffsets[0]
  file.seekg((std::streamoff)(seqCount - 1) * sizeof(size_t), std::ios::cur);  // 跳过其余
  // 读分组信息(代表索引): 只读前bands行(band-major连续) 防uint32溢出
  uint32_t useBands = bands;  // 防御: 越界钳制到[1,GROUP_BANDS](正常check已挡)
  if (useBands < 1) { useBands = 1; }
  if (useBands > GROUP_BANDS) { useBands = GROUP_BANDS; }
  inputData.repIdx.resize((size_t)useBands * seqCount);  // 只分配要用的行
  file.read((char*)inputData.repIdx.data(), sizeof(uint32_t)*useBands*seqCount);
  file.seekg((std::streamoff)(GROUP_BANDS - useBands) * seqCount
    * sizeof(uint32_t), std::ios::cur);  // 跳过未用的band行 到packedReads起点
  // 读打包数据
  size_t tempLength = 0;  // 打包数据长度
  tempLength = inputData.fastaFirst - inputData.packedOffsets[0];
  if (!allocManaged(inputData.packedReads, tempLength,
    "cudaMallocManaged(packedReads)")) {  // 分配失败
    cudaFree(inputData.packedOffsets);  // 释放已分配
    inputData.packedOffsets = nullptr;
    inputData.seqCount = 0;  // 失败标记 上层据此中止
    return inputData;  // 提前返回
  }
  file.read((char*)inputData.packedReads, tempLength);  // 读打包数据
  // 把文件偏移转为内存数据偏移
  tempLength = inputData.packedOffsets[0];  // 把文件偏移转为内存数据偏移
  for (size_t i=0; i<seqCount; i++) {  // 字节位置转为uint32_t偏移
    inputData.packedOffsets[i] -= tempLength;
    inputData.packedOffsets[i] /= sizeof(uint32_t);  // char转uint32_t
  }
  // 日志
  std::cout << "reads count:\t" << seqCount << "\n";  // 序列数
  std::cout << "longest:\t" << inputData.readLengths[0] << "\n";  // 最长
  std::cout << "shortest:\t" << inputData.readLengths[seqCount-1] << "\n";
  return inputData;  // 返回输入数据
}

//==========================================================================//
// 注意! 本内核与 fastClustering.hpp 中的 kernel_dynamic 必须保持逐字一致!
// 任何调优/修复都要同步改两份, 否则 clustering 与 fast 两条路径的结果会分叉。
//==========================================================================//
// 不要改内外循环 寄存器使用会变少
// 不要数据预取 或操作指针 用线程数掩盖延迟
// kernel_dynamic 动态规划
__global__ void __launch_bounds__(64, 1)  // maxThreadblock, minBlock/SM
kernel_dynamic(const uint32_t* reads, const size_t* offsets,
uint32_t* jobs, const uint32_t jobCount, const float threshold) {
  uint32_t index = blockDim.x*blockIdx.x+threadIdx.x; // 线程编号
  if (index >= jobCount) { return; }  // 超出范围
  const uint32_t* __restrict__ represent = reads+offsets[jobs[index*3+0]];//代表
  const uint32_t* __restrict__ target = reads+offsets[jobs[index*3+1]];  // 目标
  const uint32_t lengthRep = represent[0];  // 代表序列长度
  const uint32_t lengthTar = target[0];  // 目标序列长度
  if (lengthTar < lengthRep*threshold) { return; }  // 长度过滤(与候选收集同规则)
  uint32_t lines[2048];  // 每行结果 别全赋初值 也别动态申请 开销太大
  memset(lines, 0xFF, (lengthRep+31)/32*sizeof(uint32_t));  // 0是匹配 1是不匹配
  uint32_t Rows[4] = {0};  // 从代表取的32个碱基/氨基酸 行
  uint32_t Cols[4] = {0};  // 从目标取的32个碱基/氨基酸 列
  uint32_t matchs[16] = {0};  // 匹配的碱基/氨基酸 1是匹配 0是不匹配
  // 左偏移要对齐32 否则+32以后，会超出长度范围，导致漏比对
  const uint32_t lsft = ((uint32_t)ceil(lengthTar-lengthTar*threshold)+31)&~31;
  // 右偏移要加32 因为是分块计算，一块长是32，按右下角算右偏移
  const uint32_t rsft = (uint32_t)ceil(lengthRep-lengthTar*threshold)+32;
  for (uint32_t i=0; i<lengthTar; i+=32) {  // 遍历列
    uint32_t carrys = 0;  // 进位
    #pragma unroll  // 展开循环得到列
    for (uint32_t e=0; e<4; e++) { Cols[e] = target[1+i/32*4+e]; }  // 取列
    const uint32_t jstart = max((int32_t)i-(int32_t)lsft, (int32_t)0);  // 开始
    const uint32_t jend = min((int32_t)i+(int32_t)rsft, (int32_t)lengthRep);
    for (uint32_t j=jstart; j<jend; j+=32) {  // 遍历行
      #pragma unroll  // 展开循环得到行
      for(uint32_t e=0; e<4; e++) { Rows[e] = represent[1+j/32*4+e]; }
      for (uint32_t k=1; k<16; k++) {  // 预生成match 0跟谁都不匹配
        uint32_t match = 0xFFFFFFFF;
        #pragma unroll  // 展开循环得到单个match
        for (uint32_t e=0; e<4; e++) { match &= Rows[e]^0xFFFFFFFF+(k>>e&1); }
        matchs[k] = match;
      }
      uint32_t row = lines[j/32];  // 上一行结果
      for (uint32_t k=0; k<32; k++) {  // 32*32的核心
        uint32_t order = 0;
        for (uint32_t e=0; e<4; e++) { order += (Cols[e]>>k&1)<<e; }
        uint32_t match = matchs[order];  // 匹配上的碱基/氨基酸
        uint32_t carry = carrys & 1;  // 进位
        uint32_t term0 = row & match;
        uint32_t term1 = row & (~match);
        uint32_t carryRow = row + carry;
        carry = carryRow < row;  // 是否发生进位
        carryRow += term0;
        carry |= carryRow < term0;  // 是否发生进位
        row = carryRow | term1;
        carrys = (carrys>>1) + (carry<<31);  // 写回进位
      }
      lines[j/32] = row;  // 写回结果
    }
  }
  // 统计结果
  uint32_t lcs = 0;  // 最长公共子序列长度
  for (uint32_t i=0; i<lengthRep; i+=32) { lcs += 32-__popc(lines[i/32]); }
  // 相似度 = lcs / 比对长度（lenRep+lenTar-lcs）
  if (lcs > (lengthRep+lengthTar-lcs)*threshold) {  // 不用优化 没第二个分支了
    jobs[index*3+2] = lcs*100/(lengthRep+lengthTar-lcs);  // 写入比对结果
  } else {
    jobs[index*3+2] = 0;  // 写入比对结果
  }
}

// buildGraph 构建邻接图
std::vector<std::tuple<int, int, float>> buildGraph
(const Data& inputData, const uint32_t identity, const uint32_t bands, bool& ok) {  // ok 是否成功
  const size_t seqCount = inputData.seqCount;  // 序列数
  const float threshold = identity / 100.0f;  // 相似度阈值
  const uint32_t b = bands;  // band数
  // 候选判定: 非组头(rep!=自己) 且 目标长度满足阈值(与kernel长度过滤一致)
  auto isJob = [&](const uint32_t* repRow, size_t i) {
    uint32_t repSeq = repRow[i];  // 最近代表
    return repSeq != i &&
      inputData.readLengths[i] >= inputData.readLengths[repSeq] * threshold;
  };
  std::vector<std::tuple<int, int, float>> graph;  // 邻接图
  std::vector<uint64_t> cand;  // band级候选缓冲 跨band复用(G1)
  std::vector<size_t> chunkCnt;  // 分块候选计数(跨band复用)
  uint32_t *jobs = nullptr;  // 比对任务 跨band复用(G2)
  size_t jobsCap = 0;  // jobs已分配容量(uint32数)
  Timer::Timer tLoop;  // 计时: bands轮 收集+比对
  size_t totalPairs = 0;  // 全部band候选总数(含跨band重复)
  for (uint32_t band = 0; band < b; band++) {  // 循环每band一行候选
    // 1. 取本band的代表索引行 (makedb已按band-major存好)
    const uint32_t* rep = inputData.repIdx.data() + (size_t)band * seqCount;
    // 2. 并行收集本band候选: 分块计数->前缀和->并行填充(顺序与串行一致)
    const size_t CHUNK = 1 << 16;  // 每块序列数
    const size_t nChunk = (seqCount + CHUNK - 1) / CHUNK;  // 块数
    chunkCnt.assign(nChunk, 0);  // 各块候选数
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < nChunk; c++) {  // 并行计数
      const size_t beg = c * CHUNK;  // 块起点
      const size_t end = std::min(beg + CHUNK, seqCount);  // 块终点
      size_t cnt = 0;  // 本块候选数
      for (size_t i = beg; i < end; i++) { if (isJob(rep, i)) { cnt++; } }
      chunkCnt[c] = cnt;
    }
    size_t total = 0;  // 前缀和->各块写起点
    for (size_t c = 0; c < nChunk; c++) { size_t t = chunkCnt[c]; chunkCnt[c] = total; total += t; }
    cand.resize(total);  // 一次分配(只扩容不越界)
    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < nChunk; c++) {  // 并行填充
      const size_t beg = c * CHUNK;  // 块起点
      const size_t end = std::min(beg + CHUNK, seqCount);  // 块终点
      size_t k = chunkCnt[c];  // 本块写位置
      for (size_t i = beg; i < end; i++) {
        if (isJob(rep, i)) { cand[k++] = ((uint64_t)rep[i] << 32) | i; }
      }
    }
    const uint32_t count = (uint32_t)cand.size();  // 本band候选数
    if (count == 0) { continue; }  // 本band无候选 直接下一行
    totalPairs += count;  // 累计候选数
    // 3. 准备本band任务缓冲 (rep, tar, 0) 容量不足才重新分配
    if ((size_t)count * 3 > jobsCap) {  // 需要扩容
      if (jobs) { cudaFree(jobs); }  // 释放旧缓冲
      jobsCap = (size_t)count * 3;  // 新容量(3×count个uint32)
      if (!allocManaged(jobs, jobsCap * sizeof(uint32_t),
        "cudaMallocManaged(jobs)")) {  // 分配失败
        jobs = nullptr;  // 指针置空
        jobsCap = 0;  // 容量归零
        ok = false;  // 失败标记
        return {};  // 提前返回
      }
    }
    #pragma omp parallel for schedule(static)  // 并行填 各元素独立
    for (uint32_t j = 0; j < count; j++) {
      jobs[j * 3 + 0] = (uint32_t)(cand[j] >> 32);  // 代表序列
      jobs[j * 3 + 1] = (uint32_t)cand[j];  // 目标序列
      jobs[j * 3 + 2] = 0;  // 结果初始0
    }
    // 4. 启动kernel比对本band
    kernel_dynamic<<<(count + 63) / 64, 64>>>(
      inputData.packedReads, inputData.packedOffsets,
      jobs, count, threshold
    );
    if (!cudaOk(cudaGetLastError(), "kernel launch")) {  // launch配置错误
      cudaFree(jobs);  // 释放复用缓冲
      jobs = nullptr;
      ok = false;  // 失败标记
      return {};  // 提前返回
    }
    if (!cudaOk(cudaDeviceSynchronize(), "kernel execution")) {  // kernel运行错误
      cudaFree(jobs);  // 释放复用缓冲
      jobs = nullptr;
      ok = false;  // 失败标记
      return {};  // 提前返回
    }
    // 5. 比对结果加入图 (允许跨band重复边 最后统一去重)
    graph.reserve(graph.size() + count);  // 按本band候选数渐进预留 减少realloc
    for (uint32_t j = 0; j < count; j++) {
      uint32_t score = jobs[j * 3 + 2];  // 比对结果
      if (score > 0) {
        graph.emplace_back(
          (int)jobs[j * 3 + 0], (int)jobs[j * 3 + 1],
          (float)score / 100.0f
        );
      }
    }
  }
  cudaFree(jobs);  // 释放跨band复用缓冲(G2)
  tLoop.pause();
  std::cout << "compare pairs:\t" << totalPairs << "\n";  // 含跨band重复
  std::cout << "[align " << b << " bands]\t"; tLoop.printDuration();
  // 6. 边表排序去重(去掉跨band重复边 同对序列比对结果确定 权重相同)
  Timer::Timer tGraph;  // 计时：构图
  __gnu_parallel::sort(graph.begin(), graph.end());  // OpenMP并行排序(小数据自动退化)
  graph.erase(std::unique(graph.begin(), graph.end()), graph.end());  // 去重
  graph.shrink_to_fit();  // 释放内存
  tGraph.pause();
  std::cout << "[buildGraph]\t"; tGraph.printDuration();
  return graph;
}

void freeMemory(Data& inputData) {
  inputData.readLengths.clear();
  inputData.readLengths.shrink_to_fit();
  inputData.repIdx.clear();
  inputData.repIdx.shrink_to_fit();
  cudaFree(inputData.packedOffsets);
  inputData.packedOffsets = nullptr;
  cudaFree(inputData.packedReads);
  inputData.packedReads = nullptr;
}

}  // namespace BuildGraph

#endif  // BUILDGRAPH_HPP

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
