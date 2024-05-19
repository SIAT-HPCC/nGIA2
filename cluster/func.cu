// func.cu
// Endian:little 默认小端
// 为了方便 统一用uint32_t 避免有符号溢出
// 很多优化看起来冗余 不提升性能 是为减少不可预期的编译行为 让性能稳定 让耗时无波动
#include <iostream>  // cout
#include <fstream>  // fstream
#include <vector>  // vector
#include <unordered_map>  // unordered_map
#include <algorithm>  // stable_sort
#include "parser.h"  // 解析器
#include "timer.h"  // timer
#include "func.h"  // 数据结构与函数

// init 初始化 ok
void init(int argc, char **argv, Option &option) {
  {  // 解析命令行
    Parser::Parser parser;  // 解析器
    parser.add("packed", "-p", "packed file", "string", "", true);
    parser.add("result", "-r", "result file", "string", "", true);
    parser.add("identity", "-i", "identity 1-99", "int32_t", "", true);
    parser.add("mode", "-m", "mode 0:precise 1:fast", "int32_t", "0", false);
    if (!parser.parse(argc, argv)) exit(0);  // 解析
    option.packedFile = parser.getString("packed");  // packed文件
    option.resultFile = parser.getString("result");  // result文件
    option.identity = parser.getInt32_t("identity");  // 相似度
    option.mode = parser.getInt32_t("mode");  // 聚类模式
  }
  {  // 校验参数
    std::ifstream packedFile(option.packedFile);  // packed文件
    if (!packedFile.is_open()) {  // 没有输入文件
      std::cout << option.packedFile << " not exists\n";
      exit(0);
    }
    packedFile.close();
    std::cout << "packed:\t\t" << option.packedFile << "\n";  // 打印信息
    std::cout << "result:\t\t" << option.resultFile << "\n";  // 打印信息
    if (option.identity<1 || option.identity>99) {  // 相似度范围
      std::cout << "identity should be 1-99\n";
      exit(0);
    }
    std::cout << "identity:\t" << option.identity << "\n";  // 打印信息
    if (option.mode!=0 && option.mode!=1) {  // 聚类模式
      std::cout << "mode should be 0 or 1\n";
      exit(0);
    }
    if (option.mode == 0) std::cout << "mode:\t\tprecise\n";  // 打印信息
    if (option.mode == 1) std::cout << "mode:\t\tfast\n";  // 打印信息
  }
  {  // 配置显卡 export CUDA_VISIBLE_DEVICES=0 指定GPU
    cudaDeviceProp prop;  // 显卡属性
    if (cudaGetDeviceProperties(&prop, 0)!=cudaSuccess) {  // 获取属性
      std::cout << "find no GPU \n";
      exit(0);
    }
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 共享内存变缓存
    cudaDeviceSynchronize();  // 激活GPU
    std::cout << "use GPU:\t" << prop.name << "\n";
  }
}

__constant__ uint32_t represent[10242];  // 代表序列 常量内存 约40KB

// 不要改内外循环 寄存器使用会变少
// 不要数据预取 或操作指针 用线程数掩盖延迟
// kernel_dynamic 纯比对的动态规划
template<uint32_t entropy, uint32_t tabsize>  // 熵 字母表
__global__ void __launch_bounds__(64, 1)  // maxThread/block, minBlock/SM
kernel_dynamic0(uint32_t *reads, size_t *offsets, uint32_t *remains,
const uint32_t remainCount, uint32_t *cluster, const float threshold) {
  uint32_t index = blockDim.x*blockIdx.x+threadIdx.x+1;  // 线程编号
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
  uint32_t lsft = ceil((float)length2-(float)length2*threshold);  // 左偏移
  lsft = (lsft+31)/32*32;  // 32对齐
  uint32_t rsft = ceil((float)length1-(float)length2*threshold);  // 右偏移
  rsft = rsft+33;  // 32补全
  // 计算
  for (uint32_t i=0; i<netLength2; i+=32) {  // 遍历列
    uint32_t carrys = 0;  // 进位
    for (uint32_t e=0; e<entropy; e++) Cols[e] = read[2+(i>>5)*entropy+e];
    uint32_t jstart = max((int32_t)i-(int32_t)lsft, (int32_t)0);  // 开始
    uint32_t jend = min((int32_t)i+(int32_t)rsft, (int32_t)netLength1);  // 结束
    for (uint32_t j=jstart; j<jend; j+=32) {  // 遍历行
      for (uint32_t e=0; e<entropy; e++)
        Rows[e] = represent[2+(j>>5)*entropy+e];
      for (uint32_t k=0; k<tabsize; k++) {  // 预生成match
        uint32_t match = 0xFFFFFFFF;
        for (uint32_t e=0; e<entropy; e++) match &= Rows[e]^0xFFFFFFFF+(k>>e&1);
        matchs[k] = match;
      }
      uint32_t row = lines[j>>5];  // 上一行结果
      for (uint32_t k=0; k<32; k++) {  // 32*32的核心
        uint32_t order = 0;
        for (uint32_t e=0; e<entropy; e++) order += (Cols[e]>>k&1)<<e;
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
      lines[j>>5] = row;
    }
  }
  {  // 统计结果
    uint32_t sum = 0;
    for (uint32_t i=0; i<netLength1; i+=32) sum += 32-__popc(lines[i>>5]);
    sum-=min((netLength1+31)/32*32-netLength1,(netLength2+31)/32*32-netLength2);
    uint32_t cutoff = ceil((float)length2*threshold);
    if (sum >= cutoff) {  // 不用优化 没第二个分支 要返回了
      cluster[remains[index]] = remains[0];
      remains[index] = 0xFFFFFFFF;  // 已经聚类了
    }
  }
}

// clusteringPrecise 利用比对准确聚类
void clusteringPrecise(const Option &option, std::vector<uint32_t> &results) {
  uint32_t entropy = 0;  // 数据的熵
  uint32_t readsCount = 0;  // 序列数
  size_t *offsets = NULL;  // 序列偏移
  uint32_t *reads = NULL;  // 序列数据
  {  // 读数据start
    std::ifstream packedFile(option.packedFile);  // packed文件
    packedFile.read((char*)&entropy, sizeof(uint32_t));  // 读序列类型
    packedFile.read((char*)&readsCount, sizeof(uint32_t));  // 读序列数
    size_t distance = sizeof(uint32_t)*readsCount*2;
    packedFile.seekg(distance, std::ios::cur);  // 跳过序列长度数据
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
    for (uint32_t i=0; i<readsCount; i++) {  // 字节位置转为uint32_t偏移
      offsets[i] = (offsets[i]-position)/sizeof(uint32_t);
    }
    if (entropy == 3) std::cout << "data type:\tgene" << "\n";  // 基因
    if (entropy == 5) std::cout << "data type:\tprotein" << "\n";  // 蛋白
    std::cout << "reads count:\t" << readsCount << "\n";  // 序列数
    std::cout << "longest:\t" << reads[0] << "\n";  // 最长
    std::cout << "shortest:\t" << reads[offsets[readsCount-1]] << "\n";  // 最短
  }  // 读数据end
  uint32_t *cluster = NULL;  // 聚类结果
  uint32_t *remains = NULL;  // 剩余序列
  {  // 聚类过程start
    float threshold = (float)option.identity/100.0f;  // 相似度阈值
    cudaMallocManaged(&cluster, sizeof(uint32_t)*readsCount);  // 聚类结果
    cudaMallocManaged(&remains, sizeof(uint32_t)*readsCount);  // 剩余序列
    for (uint32_t i=0; i<readsCount; i++) {  // 初始化
      cluster[i] = 0xFFFFFFFF;  // 最大值就是没聚类 是代表序列
      remains[i] = i;  // 需要比对的序列
    }
    uint32_t remainCount = readsCount;  // 剩余序列数
    std::cout << "clustering:\n";
    while (remainCount > 0) {  // 直到剩余序列为0
      // 准备代表序列
      std::cout << "\r" << remains[0]+1 << "/" << readsCount << std::flush;
      size_t repOff = offsets[remains[0]];  // 代表序列的偏移
      size_t length = ((reads[repOff]+31)/32*entropy+2)*sizeof(uint32_t);
      cudaMemcpyToSymbol(represent, &reads[repOff], length);
      // 序列比对
      if (entropy == 3) kernel_dynamic0<3, 5><<<(remainCount+63)>>6, 64>>>
        (reads, offsets, remains, remainCount, cluster, threshold);  // 基因
      if (entropy == 5) kernel_dynamic0<5, 23><<<(remainCount+63)>>6, 64>>>
        (reads, offsets, remains, remainCount, cluster, threshold);  // 蛋白
      // 计算剩余任务
      cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount,
        cudaCpuDeviceId, 0);  // toHost
      cudaStreamSynchronize(0);  // 等数据传输完成
      uint32_t count = 0;
      for (uint32_t i=1; i<remainCount; i++) {  // 计算剩余的序列
        if (remains[i] != 0xFFFFFFFF) {
          remains[count] = remains[i];
          count += 1;
        }
      }
      remainCount = count;  // 剩余序列数就是任务数
      cudaMemPrefetchAsync(remains, sizeof(uint32_t)*remainCount, 0);  // toGPU
    }
    std::cout << "\r" << readsCount << "/" << readsCount << "\n";
    cudaDeviceSynchronize();  // 整体退出
    results.assign(readsCount, 0);  // 聚类结果
    cudaMemcpy (results.data(), cluster, sizeof(uint32_t)*readsCount,
      cudaMemcpyDeviceToHost);  // 拷贝结果回内存
    cudaDeviceSynchronize();  // 整体退出
  }  // 聚类过程end
  cudaFree(offsets);
  cudaFree(reads);
  cudaFree(cluster);
  cudaFree(remains);
}

// 不要改内外循环 寄存器使用会变少
// 不要数据预取 或操作指针 用线程数掩盖延迟
// kernel_dynamic1 局部敏感哈希的动态规划
template<uint32_t entropy, uint32_t tabsize>  // 熵 字母表
__global__ void __launch_bounds__(64, 1)  // maxThread/block, minBlock/SM
kernel_dynamic1(uint32_t *reads, size_t *offsets, uint32_t *jobs,
const uint32_t jobCount, uint32_t *cluster, const float threshold) {
  uint32_t index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  if (index >= jobCount) return;  // 超出范围
  uint32_t *represent = &reads[offsets[jobs[index*2+0]]];  // 代表序列起始位置
  uint32_t *read = &reads[offsets[jobs[index*2+1]]];  // 任务序列起始位置
  uint32_t length1 = represent[0];  // 代表序列长度
  uint32_t length2 = read[0];  // 剩余序列长度
  uint32_t netLength1 = represent[1];  // 代表序列净长度
  uint32_t netLength2 = read[1];  // 剩余序列净长度
  uint32_t lines[2048];  // 每行结果 别赋初值 开销太大
  memset(lines, 0xFF, (netLength1+31)/32*sizeof(uint32_t));  // 0:匹配 1:不匹配
  uint32_t Rows[entropy] = {0};  // 从行取的32个碱基/氨基酸
  uint32_t Cols[entropy] = {0};  // 从列取的32个碱基/氨基酸
  uint32_t matchs[tabsize] = {0};  // 匹配的碱基/氨基酸 寄存器 1匹配 0不匹配
  uint32_t lsft = ceil((float)length2-(float)length2*threshold);  // 左偏移
  lsft = (lsft+31)/32*32;  // 32对齐
  uint32_t rsft = ceil((float)length1-(float)length2*threshold);  // 右偏移
  rsft = rsft+33;  // 32补全
  // 计算
  for (uint32_t i=0; i<netLength2; i+=32) {  // 遍历列
    uint32_t carrys = 0;  // 进位
    for (uint32_t e=0; e<entropy; e++) Cols[e] = read[2+(i>>5)*entropy+e];
    uint32_t jstart = max((int32_t)i-(int32_t)lsft, (int32_t)0);  // 开始
    uint32_t jend = min((int32_t)i+(int32_t)rsft, (int32_t)netLength1);  // 结束
    for (uint32_t j=jstart; j<jend; j+=32) {  // 遍历行
      for (uint32_t e=0; e<entropy; e++)
        Rows[e] = represent[2+(j>>5)*entropy+e];
      for (uint32_t k=0; k<tabsize; k++) {  // 预生成match
        uint32_t match = 0xFFFFFFFF;
        for (uint32_t e=0; e<entropy; e++) match &= Rows[e]^0xFFFFFFFF+(k>>e&1);
        matchs[k] = match;
      }
      uint32_t row = lines[j>>5];  // 上一行结果
      for (uint32_t k=0; k<32; k++) {  // 32*32的核心
        uint32_t order = 0;
        for (uint32_t e=0; e<entropy; e++) order += (Cols[e]>>k&1)<<e;
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
      lines[j>>5] = row;
    }
  }
  {  // 统计结果
    uint32_t sum = 0;
    for (uint32_t i=0; i<netLength1; i+=32) sum += 32-__popc(lines[i>>5]);
    sum-=min((netLength1+31)/32*32-netLength1,(netLength2+31)/32*32-netLength2);
    uint32_t cutoff = ceil((float)length2*threshold);
    if (sum >= cutoff) {  // 不用优化 没第二个分支 要返回了
      cluster[jobs[index*2+1]] = jobs[index*2+0];  // 写入聚类结果
    }
  }
}

// clusteringFast 利用局部敏感哈希快速聚类
void clusteringFast(const Option &option, std::vector<uint32_t> &results) {
  uint32_t entropy = 0;  // 数据的熵
  uint32_t readsCount = 0;  // 序列数
  std::vector<uint32_t> hashTable(0);  // hashTable
  size_t *offsets = NULL;  // 序列偏移
  uint32_t *reads = NULL;  // 序列数据
  {  // 读数据start
    std::ifstream packedFile(option.packedFile);  // packed文件
    packedFile.read((char*)&entropy, sizeof(uint32_t));  // 读序列类型
    packedFile.read((char*)&readsCount, sizeof(uint32_t));  // 读序列数
    size_t distance = sizeof(uint32_t)*readsCount*2;
    packedFile.seekg(distance, std::ios::cur);  // 跳过序列长度数据
    cudaMallocManaged(&offsets, sizeof(size_t)*(readsCount+1));  // packed偏移
    cudaMemAdvise(offsets, sizeof(size_t)*(readsCount+1),
      cudaMemAdviseSetReadMostly, 0);  // 告诉编译器 只读不写
    packedFile.read((char*)offsets, sizeof(size_t)*(readsCount+1));  // 序列偏移
    hashTable.assign(readsCount*64, 0);  // hashTable
    size_t hashOffset = sizeof(uint32_t)*(2+readsCount*2);  // hashTable偏移
    hashOffset += sizeof(size_t)*readsCount*2;
    packedFile.seekg(hashOffset, std::ios::beg);  // 移到hashTable处
    packedFile.read((char*)hashTable.data(), sizeof(uint32_t)*readsCount*64);
    cudaMallocManaged(&reads, offsets[readsCount]-offsets[0]);  // 打包数据
    cudaMemAdvise(reads, offsets[readsCount]-offsets[0],
      cudaMemAdviseSetReadMostly, 0);  // 告诉编译器 只读不写
    packedFile.seekg(offsets[0], std::ios::beg);  // 移位
    packedFile.read((char*)reads, offsets[readsCount]-offsets[0]);  // 打包数据
    packedFile.close();  // 读文件完成
    size_t position = offsets[0];  // 偏移的起始位置
    for (uint32_t i=0; i<readsCount; i++) {  // 字节位置转为uint32_t偏移
      offsets[i] = (offsets[i]-position)/sizeof(uint32_t);
    }
    if (entropy == 3) std::cout << "data type:\tgene" << "\n";  // 基因
    if (entropy == 5) std::cout << "data type:\tprotein" << "\n";  // 蛋白
    std::cout << "reads count:\t" << readsCount << "\n";  // 序列数
    std::cout << "longest:\t" << reads[0] << "\n";  // 最长
    std::cout << "shortest:\t" << reads[offsets[readsCount-1]] << "\n";  // 最短
  }  // 读数据end
  uint32_t *cluster = NULL;  // 聚类结果
  uint32_t *jobs = NULL;  // 比对任务
  {  // 聚类过程start
    uint32_t row = 0, block = 0;  // minHash算法的b和r
    float threshold = (float)option.identity/100.0f;  // 相似度阈值
    if (0.01f<=threshold && threshold<0.30f) {row= 1; block=64;}  // 01-30
    if (0.30f<=threshold && threshold<0.65f) {row= 2; block=32;}  // 30-65
    if (0.65f<=threshold && threshold<0.87f) {row= 4; block=16;}  // 65-87
    if (0.87f<=threshold && threshold<0.97f) {row= 8; block= 8;}  // 87-97
    if (0.97f<=threshold && threshold<0.99f) {row=16; block= 4;}  // 97-99
    std::cout << "hash row:\t" << row << "\n";
    std::cout << "hash blk:\t" << block << "\n";
    cudaMallocManaged(&cluster, sizeof(uint32_t)*readsCount);  // 聚类结果
    memset(cluster, 0xFF, sizeof(uint32_t)*readsCount);  // 0xFFFFFFFF是未聚类的
    cudaMallocManaged(&jobs, sizeof(uint32_t)*readsCount*2);  // 剩余序列
    memset(jobs, 0, sizeof(uint32_t)*readsCount*2);  // 最初没有任务
    uint32_t jobCount = 0;  // 任务数是0
    std::unordered_map<std::string, std::vector<uint32_t>> preClusters;  // 预聚
    std::cout << "clustering:\n";  // 开始聚类
    for (uint32_t b=0; b<block; b++) {  // 遍历hashTable的block
      std::cout << "\r" << b+1 << "/" << block << std::flush;
      preClusters.clear();  // 预聚类结果
      std::string signedName = "";  // 签名
      for (uint32_t i=0; i<readsCount; i++) {  // 遍历所有序列的签名
        signedName.clear();  // 清空
        for (uint32_t r=b*row; r<b*row+row; r++) {  // block中的多row生成签名
          signedName += std::to_string(hashTable[i*64+r])+" ";
        }
        const auto &iterator = preClusters.find(signedName);  // 查找签名
        if (iterator == preClusters.end()) {  // 没找到签名就添加记录
          preClusters[signedName] = std::vector<uint32_t>{i};
        } else {  // 找到签名就添加序列
          iterator->second.push_back(i);
        }
      }
      jobCount = 0;
      for (const auto iterator:preClusters) {  // 遍历签名记录 分配比对任务
        if (iterator.second.size() == 1) continue;  // 孤狼序列不比对
        uint32_t rep = iterator.second[0];  // 代表序列
        if (cluster[rep] == 0xFFFFFFFF) cluster[rep] = rep;  // 第一条序列入类
        rep = cluster[rep];  // 代表序列相似的序列可以作为新代表序列
        for (uint32_t j=1; j<iterator.second.size(); j++) {  // 写入任务
          jobs[jobCount*2+0] = rep;
          jobs[jobCount*2+1] = iterator.second[j];
          jobCount += 1;
        }
      }
      // 序列比对
      cudaDeviceSynchronize();  // 同步数据
      if (entropy == 3) kernel_dynamic1<3, 5><<<(jobCount+63)>>6, 64>>>
        (reads, offsets, jobs, jobCount, cluster, threshold);  // 基因
      if (entropy == 5) kernel_dynamic1<5, 23><<<(jobCount+63)>>6, 64>>>
        (reads, offsets, jobs, jobCount, cluster, threshold);  // 蛋白
      cudaDeviceSynchronize();  // 同步数据
    }
    std::cout << "\r" << block << "/" << block << "\n";
  }  // 聚类过程end
  {  // 生成结果start
    for (uint32_t i=0; i<readsCount; i++) {  // 代表序列重新写为0xFFFFFFFF
      if (cluster[i] == i) cluster[i] = 0xFFFFFFFF;
    }
    results.assign(readsCount, 0);  // 聚类结果
    cudaMemcpy (results.data(), cluster, sizeof(uint32_t)*readsCount,
      cudaMemcpyDeviceToHost);  // 拷贝结果回内存
  }  // 生成结果end
  cudaFree(offsets);
  cudaFree(reads);
  cudaFree(cluster);
  cudaFree(jobs);
}

// countResult 统计结果
void conutResult(const Option &option, const std::vector<uint32_t> &results) {
  uint32_t readsCount = results.size();  // 序列数
  std::vector<uint64_t> orders(readsCount, 0);  // 前32bit代表 后32bit任务
  {  // 计算结果文件的写入顺序
    for (uint32_t i=0; i<readsCount; i++) {  // 遍历结果
      uint32_t rep = results[i];  // 记录了代表序列
      if (rep == 0xFFFFFFFF) rep = i;  // 自己就是代表序列
      orders[i] = (((uint64_t)rep)<<32)+(uint64_t)i;
    }
    std::stable_sort(orders.begin(), orders.end());  // 排序
  }
  {  // 写入结果文件
    std::ifstream fastaFile(option.packedFile);  // 输入
    std::ofstream resultFile(option.resultFile);  // 输出
    std::vector<size_t> offsets(readsCount, 0);  // 序列偏移
    fastaFile.seekg(sizeof(uint32_t)*(2+readsCount*2)+
      sizeof(size_t)*(readsCount), std::ios::beg);  // 跳到序列偏移开始位置
    fastaFile.read((char*)offsets.data(), sizeof(size_t)*readsCount);  // 读偏移
    std::string name="", read="";  // 序列名 序列数据
    uint32_t count = 0;  // 代表序列的数量
    std::cout << "write results:\n";
    for (uint32_t i=0; i<readsCount; i++) {  // 写入结果
      uint32_t rep = (orders[i]>>32)&0xFFFFFFFF;  // 代表序列
      uint32_t job = orders[i]&0xFFFFFFFF;  // 任务序列
      fastaFile.seekg(offsets[job], std::ios::beg);  // 跳到序列开始
      getline(fastaFile, name); name += "\n";  // 读序列名
      getline(fastaFile, read); read += "\n";  // 读序列数据
      if (rep == job) {  // 代表序列
        resultFile.write((char*)name.data(), name.size());
        resultFile.write((char*)read.data(), read.size());
        count += 1;
      } else {  // 任务序列
        name = "\t"+name;
        resultFile.write((char*)name.data(), name.size());
      }
      if (i%1024 == 0) std::cout<<"\r"<<i+1<<"/"<<readsCount<< std::flush;
    }
    std::cout << "\r" << readsCount << "/" << readsCount << "\n";
    std::cout << "cluster:\t" << count << "\n";
  }
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