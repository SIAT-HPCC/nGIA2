// func.cu
// Endian:little 默认小端
// 很多优化看起来冗余 不提升性能 是为减少不可预期的编译行为 让性能稳定 让耗时无波动
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
    if (option.identity < 1 || option.identity > 99) {  // 数据范围不对
      std::cout << "identity should be 1-99\n";
      exit(0);
    }
  }
  cudaDeviceProp prop;  // 显卡属性
  int32_t count = 0;  // 显卡个数
  {  // 配置显卡
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) {
      std::cout << "find no GPU\n";
      exit(0);
    }
    cudaGetDeviceProperties(&prop, 0);  // 获取属性
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 共享内存变缓存
    cudaDeviceSynchronize();  // 激活GPU
  }
  // 打印信息
  std::cout << "packed:\t\t" << option.packedFile << "\n";
  std::cout << "result:\t\t" << option.resultFile << "\n";
  std::cout << "identity:\t" << option.identity << "\n";
  std::cout << "use GPU:\t" << prop.name << " (1/" << count << ")\n";
}

__constant__ uint32_t represent[10242];  // 代表序列 常量内存 约40KB

// 不要改内外循环 寄存器使用会变少
// 不要数据预取 或操作指针 用线程数掩盖延迟
// kernel_dynamic 动态规划
template<int32_t entropy, int32_t tabsize>  // 熵 字母表
__global__ void kernel_dynamic(uint32_t *reads, size_t *offsets,
uint32_t *remains, const int32_t remainCount, uint32_t *cluster,
const float threshold) {
  int32_t index = blockDim.x*blockIdx.x+threadIdx.x+1;  // 线程编号
  if (index >= remainCount) return;  // 超出范围
  uint32_t *read = &reads[offsets[remains[index]]];  // 任务序列起始位置
  uint32_t length1 = represent[0];  // 代表序列长度
  uint32_t length2 = read[0];  // 剩余序列长度
  uint32_t netLength1 = represent[1];  // 代表序列净长度
  uint32_t netLength2 = read[1];  // 剩余序列净长度
  uint32_t lines[2048];  // 每行结果 别赋初值 开销太大
  memset(lines, 0xFF, (netLength1+31)/32*sizeof(uint32_t));  // 0:匹配 1:不匹配
  uint32_t Rows[entropy] = {0};  // 从行取的32个碱基/氨基酸
  uint32_t Cols[entropy] = {0};  // 从列取的32个碱基/氨基酸
  uint32_t matchs[tabsize] = {0};  // 匹配的碱基/氨基酸 寄存器 1匹配 0不匹配
  int32_t lsft = ceil((float)length2-(float)length2*threshold);  // 左偏移
  lsft = (lsft+31)/32*32;  // 32对齐
  int32_t rsft = ceil((float)length1-(float)length2*threshold);  // 右偏移
  rsft = rsft+33;  // 32补全
  // 计算
  for (int32_t i=0; i<netLength2; i+=32) {  // 遍历列
    uint32_t carrys = 0;  // 进位
    for (int32_t e=0; e<entropy; e++) Cols[e] = read[2+i/32*entropy+e];
    for (int32_t j=max(i-lsft,0); j<min(i+rsft,netLength1); j+=32) {  // 遍历行
      for (int32_t e=0; e<entropy; e++) Rows[e] = represent[2+j/32*entropy+e];
      for (int32_t k=0; k<tabsize; k++) {  // 预生成match
        uint32_t match = 0xFFFFFFFF;
        for (int32_t e=0; e<entropy; e++) match &= Rows[e]^0xFFFFFFFF+(k>>e&1);
        matchs[k] = match;
      }
      uint32_t row = lines[j/32];  // 上一行结果
      for (int32_t k=0; k<32; k++) {  // 32*32的核心
        int32_t order = 0;
        for (int32_t e=0; e<entropy; e++) order += (Cols[e]>>k&1)<<e;
        uint32_t match = matchs[order];  // 匹配上的碱基/氨基酸
        uint32_t carry = carrys&1;  // 进位
        uint32_t term0 = row & match;
        uint32_t term1 = row & (~match);
        uint32_t carryRow = row+carry;
        carry = carryRow < row;  // 是否发生进位
        carryRow += term0;
        carry |= carryRow < term0;  // 是否发生进位
        row = carryRow | term1;
        carrys = (carrys>>1)+(carry<<31);  // 写回进位
      }
      lines[j/32] = row;
    }
  }
  {  // 统计结果
    int32_t sum = 0;
    for (int32_t i=0; i<netLength1; i+=32) sum += 32-__popc(lines[i/32]);
    sum-=min((netLength1+31)/32*32-netLength1,(netLength2+31)/32*32-netLength2);
    int32_t cutoff = ceil((float)length2*threshold);
    if (sum >= cutoff) {  // 不用优化 没第二个分支 要返回了
      cluster[remains[index]] = remains[0];
      remains[index] = 0xFFFFFFFF;  // 已经聚类了
    }
  }
}

// clustering 聚类
void clustering(const Option &option, std::vector<int32_t> &result) {
  uint32_t entropy = 0;  // 数据的熵
  uint32_t readsCount = 0;  // 序列数
  size_t hashOffset = 0;  // hash偏移
  size_t *offsets = NULL;  // 序列偏移
  uint32_t *reads = NULL;  // 序列数据
  {  // 读数据
    std::ifstream packedFile(option.packedFile);  // packed文件
    packedFile.read((char*)&entropy, sizeof(uint32_t));  // 读序列类型
    packedFile.read((char*)&readsCount, sizeof(uint32_t));  // 读序列数
    packedFile.read((char*)&hashOffset, sizeof(size_t));  // hash偏移
    cudaMallocManaged(&offsets, sizeof(size_t)*(readsCount+1));  // packed偏移
    cudaMemAdvise(offsets, sizeof(size_t)*(readsCount+1),
      cudaMemAdviseSetReadMostly, 0);  // 告诉编译器 只读不写
    packedFile.read((char*)offsets, sizeof(size_t)*(readsCount+1));  // 序列偏移
    cudaMallocManaged(&reads, offsets[readsCount]-offsets[0]);  // 打包数据
    cudaMemAdvise(reads, offsets[readsCount]-offsets[0],
      cudaMemAdviseSetReadMostly, 0);  // 告诉编译器 只读不写
    packedFile.seekg(offsets[0], std::ios::beg);  // 移位
    packedFile.read((char*)reads, offsets[readsCount]-offsets[0]);  // 打包数据
    packedFile.close();  // 读文件完成
    size_t position = offsets[0];  // 偏移的起始位置
    for (int32_t i=0; i<readsCount; i++) {  // 字节位置转为uint32_t偏移
      offsets[i] = (offsets[i]-position)/sizeof(uint32_t);
    }
    if (entropy == 3) std::cout << "data type:\tgene" << "\n";  // 基因
    if (entropy == 5) std::cout << "data type:\tprotein" << "\n";  // 蛋白
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
      // 准备代表序列
      std::cout << "\r" << remains[0]+1 << "/" << readsCount << std::flush;
      size_t repOff = offsets[remains[0]];
      size_t length = ((reads[repOff+1]+31)/32*entropy+2)*sizeof(uint32_t);
      cudaMemcpyToSymbol(represent, &reads[repOff], length);
      // 序列比对
      if (entropy == 3) kernel_dynamic<3, 5><<<(remainCount+63)/64, 64>>>
        (reads, offsets, remains, remainCount, cluster, threshold);  // 基因
      if (entropy == 5) kernel_dynamic<5, 23><<<(remainCount+63)/64, 64>>>
        (reads, offsets, remains, remainCount, cluster, threshold);  // 蛋白
      // 计算剩余任务
      cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount,
        cudaCpuDeviceId, 0);  // toHost
      cudaStreamSynchronize(0);  // 等数据传输完成
      int32_t count = 0;
      for (int32_t i=1; i<remainCount; i++) {  // 计算剩余的序列
        if (remains[i] != 0xFFFFFFFF) {
          remains[count] = remains[i];
          count += 1;
        }
      }
      remainCount = count;  // 剩余序列数就是任务数
      cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount, 0);  // toGPU
    }
    cudaDeviceSynchronize();  // 整体退出
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
// cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 共享内存变缓存
// __restrict__
// 常量内存
// 显存内计算remains
// 少量序列用batch

// gene   : 90 9392 6.43686s
// protein: 80 5399 5.07424s

// 4090的SM参数:
// 1536个线程
// 48个warp
// 24个block
// 64K个寄存器
// 八个warp就能隐藏延迟了，四发射，64线程足够




// {  // 预聚类
//   std::vector<std::vector<uint32_t>> pairs(5, std::vector<uint32_t>());
//   std::unordered_map<std::string, uint32_t> represents;  // 代表序列
//   for (uint32_t i=0; i<5; i++) {  // 遍历r
//     uint32_t r = pow(2, i);
//     uint32_t b = 64/r;
//     represents.clear();  // 清空代表序列
//     for (uint32_t j=0; j<b; j++) {  // 遍历b
//       for (uint32_t k=0; k<readsCount; k++) {  // 遍历签名矩阵
//         std::string signedName = "";
//         for (int32_t l=r*j; l<r*j+r; l++) {
//           signedName += std::to_string(signedMatrix[k][l])+" ";
//         }
//         auto iterator = represents.find(signedName);
//         if (iterator == represents.end()) {  // 没找到代表序列
//           represents[signedName] = k;
//         } else {  // 找到了代表序列
//           pairs[i].push_back(iterator->second);
//           pairs[i].push_back(k);
//         }
//       }
//     }
//   }
//   // 写入结果
//   std::ofstream clusterFile(option.packedFile, std::ios::in);  // preCluster
//   clusterFile.seekp(offset, std::ios::beg);  // 移到preCluster开始
//   size_t length = 0;  // 数据长度
//   for (uint32_t i=0; i<5; i++) {  // 写预聚类数据
//     length = pairs[i].size();
//     clusterFile.write((char*)&length, sizeof(size_t));
//     clusterFile.write((char*)pairs[i].data(), sizeof(uint32_t)*length);
//   }
//   length = sizeof(uint32_t)*2+sizeof(size_t)*readsCount*2;
//   clusterFile.seekp(length, std::ios::beg);  // 移到preCluster开始
//   for (int32_t i=0; i<5; i++) {  // 写偏移
//     clusterFile.write((char*)&offset, sizeof(size_t));
//     offset += sizeof(size_t)+sizeof(uint32_t)*pairs[0].size();
//   }
//   clusterFile.close();
// }


//  r  b   s  value
// 01 64 0.05 0.9624758607888840
// 02 32 0.30 0.9510982327503508
// 04 16 0.65 0.9569802167317568
// 08 08 0.87 0.9585180051096697
// 16 04 0.97 0.9778584874251552