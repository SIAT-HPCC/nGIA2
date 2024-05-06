// func.cu
// Endian:little 默认小端
#include <iostream>  // cout
#include <fstream>  // fstream
#include <vector>  // vector
#include "parser.h"  // 解析器
#include "func.h"  // 数据结构与函数

// init 初始化 ok
void init(int argc, char **argv, Option &option) {
  {  // 解析命令行
    Parser::Parser parser;  // 解析器
    parser.add("packed", "-p", "packed file", "string", "", true);
    parser.add("result", "-r", "result file", "string", "", true);
    parser.add("identity", "-i", "identity 1-99", "int32_t", "", true);
    if (!parser.parse(argc, argv)) exit(0);  // 解析
    option.packedFile = parser.getString("packed");  // packed文件
    option.resultFile = parser.getString("result");  // result文件
    option.identity = parser.getInt32_t("identity");  // 相似度
  }
  {  // 校验参数
    std::ifstream packedFile(option.packedFile);  // packed文件
    if (!packedFile.is_open()) {  // 没有输入文件
      std::cout << option.packedFile << " not exists\n";
      exit(0);
    }
    packedFile.close();
    if (option.identity < 1 || option.identity > 99) {  // 数据类型不对
      std::cout << "identity should be 1-99\n";
      exit(0);
    }
  }
  // 打印信息
  std::cout << "packed:\t\t" << option.packedFile << "\n";
  std::cout << "result:\t\t" << option.resultFile << "\n";
  std::cout << "identity:\t" << option.identity << "\n";
}

// k32 32*32的计算核心
template<int32_t entropy>
__device__ inline void k32(const uint32_t *Rows, const uint32_t *Cols,
uint32_t *carrys, uint32_t *line, const int32_t colCount) {
  uint32_t matchs[1<<entropy] = {0};  // 匹配碱基/氨基酸 寄存器 1匹配 0不匹配
  {  // 预生成match
    for (int32_t i=0; i<1<<entropy; i++) {
      uint32_t match = 0xFFFFFFFF;
      for (int32_t e=0; e<entropy; e++) match &= Rows[e]^0xFFFFFFFF+(i>>e&1);
      matchs[i] = match;
    }
  }
  uint32_t row = *line;  // 上一行结果
  for (int32_t k=0; k<colCount; k++) {  // 32*32的核心
    int32_t order = 0;
    for (int32_t e=0; e<entropy; e++) order += (Cols[e]>>k&1)<<e;
    uint32_t match = matchs[order];  // 匹配上的碱基/氨基酸
    uint32_t carry = *carrys>>k&1;  // 进位
    uint32_t term0 = row & match;
    uint32_t term1 = row & (~match);
    uint32_t carryRow = row+carry;
    carry = carryRow < row;  // 是否发生进位
    carryRow += term0;
    carry |= carryRow < term0;  // 是否发生进位
    row = carryRow | term1;
    *carrys &= ~(1<<k); *carrys += carry<<k;  // 写回进位
  }
  *line = row;
}
// kernel_dynamicGen 动态规划
template<int32_t entropy>
__global__ void kernel_dynamic(uint32_t *reads, size_t *offsets,
uint32_t *remains, int32_t remainCount, uint32_t *cluster, float threshold) {
  int32_t index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  if (index+1 >= remainCount) return;  // 超出范围
  uint32_t *represent = &reads[offsets[remains[0]]];  // 代表序列起始位置
  uint32_t *read = &reads[offsets[remains[index+1]]];  // 任务序列起始位置
  uint32_t length1 = represent[0];  // 代表序列长度
  uint32_t length2 = read[0];  // 剩余序列长度
  uint32_t netLength1 = represent[1];  // 代表序列净长度
  uint32_t netLength2 = read[1];  // 剩余序列净长度

  uint32_t line[2048];  // 每行结果
  memset(line, 0xFF, (netLength1+31)/32*sizeof(uint32_t));  // 0:匹配 1:不匹配
  uint32_t Rows[entropy] = {0};  // 从行取的32个碱基/氨基酸
  uint32_t Cols[entropy] = {0};  // 从列取的32个碱基/氨基酸
  int32_t lshift = (length2-ceil(length2*threshold)+31)/32;  // 左偏移
  int32_t rshift = (length1-ceil(length2*threshold)+31)/32;  // 右偏移

  // 计算
  for (int32_t i=0; i<netLength2/32*32; i+=32) {  // 遍历列
    int32_t colCount = 32;  // 列向剩余
    uint32_t carrys = 0;  // 进位
    for (int32_t e=0; e<entropy; e++) Cols[e] = read[2+i/32*entropy+e];
    int32_t jstart = max(i/32-lshift, 0);
    int32_t jend = min(i/32+rshift, (netLength1+31)/32-1);
    for (int32_t j=jstart; j<=jend; j++) {  // 遍历行
      for (int32_t e=0; e<entropy; e++) Rows[e] = represent[2+j*entropy+e];
      k32<entropy>(Rows, Cols, &carrys, &line[j], colCount);
    }
  }
  for (int32_t i=netLength2/32*32; i<netLength2; i+=32) {  // 补齐
    int32_t colCount = netLength2-i;  // 列向剩余
    uint32_t carrys = 0;  // 进位
    for (int32_t e=0; e<entropy; e++) Cols[e] = read[2+i/32*entropy+e];
    int32_t jstart = max(i/32-lshift, 0);
    int32_t jend = min(i/32+rshift, (netLength1+31)/32-1);
    for (int32_t j=jstart; j<=jend; j++) {  // 遍历行
      for (int32_t e=0; e<entropy; e++) Rows[e] = represent[2+j*entropy+e];
      k32<entropy>(Rows, Cols, &carrys, &line[j], colCount);
    }
  }
  {  // 统计结果
    int32_t sum = 0;
    for (int32_t i=0; i<netLength1/32*32; i+=32) {
      sum += 32 - __popc(line[i/32]);
    }
    if (netLength1%32 != 0) {
      uint32_t mask = (1<<netLength1%32)-1;
      sum += netLength1%32 - __popc(line[netLength1/32]&mask);
    }
    int32_t cutoff = ceil((float)length2*threshold);
    if (sum >= cutoff) {
      cluster[remains[index+1]] = remains[0];
      remains[index+1] = 0xFFFFFFFF;  // 已经聚类了
    }
  }
}

// clustering 聚类
void clustering(const Option &option, std::vector<int32_t> &result) {
  int32_t entropy = 0;  // 数据的熵
  int32_t readsCount = 0;  // 序列数
  size_t *offsets = NULL;  // 序列偏移
  uint32_t *reads = NULL;  // 序列数据
  {  // 读数据
    std::ifstream packedFile(option.packedFile);  // packed文件
    packedFile.read((char*)&entropy, sizeof(int32_t));  // 读序列类型
    packedFile.read((char*)&readsCount, sizeof(int32_t));  // 读序列数
    cudaMallocManaged(&offsets, sizeof(size_t)*(readsCount+1));  // 序列偏移
    packedFile.read((char*)offsets, sizeof(size_t)*(readsCount+1));  // 序列偏移
    cudaMallocManaged(&reads, offsets[readsCount]-offsets[0]);  // 打包数据
    packedFile.seekg(offsets[0], std::ios::beg);  // 移位
    packedFile.read((char*)reads, offsets[readsCount]-offsets[0]);  // 打包数据
    packedFile.close();  // 读文件完成
    size_t position = sizeof(int32_t)*2+sizeof(size_t)*readsCount*2;
    for (int32_t i=0; i<readsCount; i++) {  // 字节位置转为uint32_t位移
      offsets[i] = (offsets[i]-position)/sizeof(uint32_t);
    }
    if (entropy == 2) {  // 基因
      std::cout << "data type:\tgene" << "\n";
    } else {  // 蛋白
      std::cout << "data type:\tprotein" << "\n";
    }
    std::cout << "reads count:\t" << readsCount << "\n";  // 序列数
    std::cout << "longest:\t" << reads[0] << "\n";  // 最长
    std::cout << "shortest:\t" << reads[offsets[readsCount-1]] << "\n";  // 最短
  }
  uint32_t *cluster = NULL;  // 聚类结果
  uint32_t *remains = NULL;  // 剩余序列
  float threshold = (float)option.identity/100.0f;  // 相似度阈值
  {  // 聚类过程
    cudaMallocManaged(&cluster, sizeof(uint32_t)*readsCount);  // 聚类结果
    cudaMallocManaged(&remains, sizeof(uint32_t)*readsCount);  // 剩余序列
    for (int32_t i=0; i<readsCount; i++) {  // 初始化
      cluster[i] = 0xFFFFFFFF;  // 最大值就是没聚类 是代表序列
      remains[i] = i;  // 需要比对的序列
    }
    int32_t remainCount = readsCount;  // 剩余序列数
    while (remainCount > 0) {  // 直到剩余序列为0
      if (entropy == 2) {  // 基因 32*8=256
        kernel_dynamic<2><<<(remainCount+255)/256, 256>>>
          (reads, offsets, remains, remainCount, cluster, threshold);
      } else {  // 蛋白 32*8=256
        kernel_dynamic<5><<<(remainCount+255)/256, 256>>>
          (reads, offsets, remains, remainCount, cluster, threshold);
      }
      cudaDeviceSynchronize();
      int32_t j = 0;
      for (int32_t i=1; i<remainCount; i++) {  // 更新剩余序列
        if (remains[i] != 0xFFFFFFFF) {  // 还没聚类
          remains[j] = remains[i];
          j += 1;
        }
      }
      remainCount = j;
      std::cout << "\r" << remains[0] << "/" << readsCount << std::flush;
    }
    std::cout << "\r" << readsCount << "/" << readsCount << "\n";
  }
  {  // 统计类个数
    int32_t count = 0;
    for (int32_t i=0; i<readsCount; i++) {
      if (cluster[i] == 0xFFFFFFFF) {
        count += 1;
      }
    }
    std::cout << count << "\n";
  }
  cudaFree(offsets);
  cudaFree(reads);
  cudaFree(cluster);
  cudaFree(remains);
}

// 优化
// 比对算法优化
// cudaMemAdvise(reads, position-offsets[0], cudaMemAdviseSetReadMostly, 0);
// 数据预取
// -maxrregcount 56 --resource-usage
// cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount, cudaCpuDeviceId, 0);
// cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount, 0);
// __restrict__
// 常量内存
// 显存内计算remains
// 少量序列用batch
