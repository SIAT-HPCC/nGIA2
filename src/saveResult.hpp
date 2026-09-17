/*
saveResult.hpp
根据clusterResult保存聚类结果。

输出格式：
  >代表序列名
  代表序列内容
    成员序列名1
    成员序列名2
  >下一个代表序列名
  下一个代表序列内容
    ...

用法:
ngia3 clustering -p packed文件 -r result文件 -i identity值
2026-07-28 by 鞠震
*/

#ifndef SAVERESULT_HPP
#define SAVERESULT_HPP

#include <iostream>  // cout
#include <fstream>   // ifstream, ofstream
#include <vector>    // vector
#include <cstdint>   // uint32_t
#include <string>    // string

namespace SaveResult {

// save 保存聚类结果
void save(const std::vector<uint32_t>& clusterResult,
const std::string& packedFile, const std::string& resultFile) {
  const size_t seqCount = clusterResult.size();  // 序列数
  // 1. 读packed文件头获取fasta偏移
  std::ifstream packedIn(packedFile, std::ios::binary);  // packed文件
  uint32_t fileSeqCount = 0;  // 文件中的序列数
  packedIn.read((char*)&fileSeqCount, sizeof(uint32_t));  // 序列数(uint32_t)
  // 跳过nameLengths和readLengths
  packedIn.seekg(sizeof(uint32_t) + fileSeqCount * sizeof(uint32_t) * 2
    + fileSeqCount * sizeof(size_t), std::ios::beg);  // 跳过到fastaOffsets
  // 读fastaOffsets
  std::vector<size_t> fastaOffsets(seqCount);  // fasta数据偏移
  packedIn.read((char*)fastaOffsets.data(), seqCount * sizeof(size_t));
  
  // 2. 按分组整理: CSR式扁平分组(避免seqCount个vector 大规模下分配次数爆炸)
  uint32_t maxId = 0;  // 最大簇编号
  for (size_t i = 0; i < seqCount; i++) {
    if (clusterResult[i] > maxId) { maxId = clusterResult[i]; }
  }
  std::vector<uint32_t> cnt((size_t)maxId + 1, 0);  // 每簇成员数
  for (size_t i = 0; i < seqCount; i++) { cnt[clusterResult[i]]++; }  // 第1趟: 计数
  std::vector<uint32_t> start((size_t)maxId + 2, 0);  // 每簇成员起始位置
  for (size_t c = 0; c <= maxId; c++) { start[c + 1] = start[c] + cnt[c]; }
  std::vector<uint32_t> members(seqCount);  // 成员扁平存放
  std::vector<uint32_t> cur(start.begin(), start.end() - 1);  // 各簇写入位置
  for (size_t i = 0; i < seqCount; i++) {  // 第2趟: 填充(按i递增->簇内有序)
    members[cur[clusterResult[i]]++] = (uint32_t)i;
  }

  // 3. 输出结果
  std::ofstream resultOut(resultFile);  // 结果文件
  std::string line;  // 一行数据
  for (size_t c = 0; c <= maxId; c++) {
    if (cnt[c] == 0) { continue; }  // 跳过空组
    const uint32_t begin = start[c];  // 本簇成员起始位置
    uint32_t represent = members[begin];  // 代表序列（编号最小）
    // 读代表序列的完整FASTA条目
    packedIn.seekg(fastaOffsets[represent], std::ios::beg);  // 定位
    std::getline(packedIn, line);  // 读 >name 行
    resultOut << line << "\n";  // 写序列名（带>）
    std::getline(packedIn, line);  // 读序列内容
    resultOut << line << "\n";  // 写序列内容
    // 输出其他成员的序列名
    for (uint32_t k = begin + 1; k < begin + cnt[c]; k++) {
      uint32_t member = members[k];  // 成员序列
      packedIn.seekg(fastaOffsets[member], std::ios::beg);  // 定位
      std::getline(packedIn, line);  // 读 >name 行
      resultOut << "  " << line << "\n";  // 加两个空格输出序列名
    }
  }
  packedIn.close();  // 关闭输入
  resultOut.close();  // 关闭输出
}

}  // namespace SaveResult
#endif  // SAVERESULT_HPP
