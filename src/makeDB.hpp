/*
makeDB.hpp
输出文件数据内容:
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
ngia2 makeDB -f fasta文件 -p packed文件
2025-05-12 by 鞠震
*/

// 为支持TB数据集 不能把数据都读入内存 要先生成索引
// 序列长超过0xFFFF 比对时line长度超2048
// 序列数超过0x7FFFFFFF 比对时线程数超过int范围

#ifndef MAKEDB_HPP
#define MAKEDB_HPP

#include <iostream>  // cout
#include <cstdint>  // int32_t
#include <vector>  // vector
#include <unordered_map>  // unordered_map
#include <algorithm>  // stable_sort
#include <fstream>  // fstream
#include <omp.h>  // openmp
#include "timer.hpp"  // timer
#include "parser.hpp"  // parser

namespace MakeDB {  // 命名空间

//--------数据--------//
struct Option {  // 输入选项
  std::string fastaFile;  // fasta文件
  std::string packedFile;  // packed文件
};

struct Read {  // 记录一条序列的位置 为了排序
  size_t offset;  // 序列位置
  uint32_t nameLength;  // 序列名长度
  uint32_t readLength;  // 序列长度 < 0xFFFF
};

//--------函数--------//
// init 初始化
void init(int argc, char **argv, Option &option) {
  {  // 解析命令行
    Parser::Parser parser;  // 解析器
    parser.add(true, "-f", "fasta file (string)", "");  // fasta
    parser.add(true, "-p", "packed file (string)", "");  // packed
    if (!parser.parse(argc, argv)) { exit(0); }  // 解析命令行
    option.fastaFile = parser.getValue<std::string>("-f");  // fasta文件
    option.packedFile = parser.getValue<std::string>("-p");  // packed文件
  }
  bool checkPassed = true;  // 校验是否通过
  {  // 校验参数
    std::ifstream fastaFile(option.fastaFile);  // fasta文件
    if (!fastaFile.good()) {  // 文件不存在
      std::cout << option.fastaFile << " does not exist.\n";
      checkPassed = false;
    }
    std::string line = "";  // 一行数据
    getline(fastaFile, line);
    if (line.back() == '\r') {  // 行尾有\r
      std::cout << "Please remove the \\r at the end of lines.\n";
      checkPassed = false;
    }
  }
  if (!checkPassed) { exit(0); }  // 校验
  std::cout << "fasta:\t" << option.fastaFile << "\n";
  std::cout << "packed:\t" << option.packedFile << "\n";
  std::cout << "thread:\t" << omp_get_max_threads() << "\n";
}
 
// makeIndex 文件索引
void makeIndex(const Option &option, std::vector<Read> &reads) {
  {  // 读fasta文件
    std::ifstream fastaFile(option.fastaFile);  // 输入
    std::string line = "";  // 读入的一行
    Read read = {offset:0, nameLength:0, readLength:0};  // 一条数据
    std::cout << "read:\t." << std::flush;  // 进度条
    while (fastaFile.peek()!=EOF && reads.size()<0x7FFFFFFF) {  // 读文件
      read.offset = fastaFile.tellg();  // fasta起始位置
      getline(fastaFile, line);  // 读序列名
      read.nameLength = line.size();
      read.readLength = 0;  // 序列数据长度清零
      while (fastaFile.peek()!=EOF && fastaFile.peek()!='>') {  // 读序列
        getline(fastaFile, line);
        read.readLength += line.size();
      }
      if (read.readLength < 0xFFFF) {  // 小于65536 入队 打印进度
        reads.push_back(read);  // 入队
        if (reads.size()%(1024*1024) == 0) std::cout << "." << std::flush;
      }
    }
    std::cout << " finish\n";
    reads.shrink_to_fit();  // 省点内存
    std::stable_sort(reads.begin(), reads.end(),  // 排序
      [](const Read &a, const Read &b) { return a.readLength > b.readLength; });
  }
  {  // 计算偏移并写入packedFile
    const uint32_t readsCount = reads.size();  // 序列数
    std::vector<size_t> packedOffsets(readsCount, 0);  // packed偏移
    std::vector<size_t> fastaOffsets(readsCount, 0);  // fasta偏移
    size_t offset = 0;  // 偏移
    {  // 计算偏移
      offset = sizeof(uint32_t)+sizeof(size_t)*readsCount*2;
      offset += sizeof(uint32_t)*128*readsCount;  // packed数据偏移
      for (uint32_t i=0; i<readsCount; i++) {  // packed偏移
        packedOffsets[i] = offset;
        offset += sizeof(uint32_t)*(1+(reads[i].readLength+31)/32*5);
      }
      for (uint32_t i=0; i<readsCount; i++) {  // fasta偏移
        fastaOffsets[i] = offset;
        offset += reads[i].nameLength+reads[i].readLength+2;  // 包括换行
      }
    }
    {  // 把偏移写入packedFile
      std::ofstream packedFile(option.packedFile);  // 输出文件 写入结果
      packedFile.seekp(offset-1, packedFile.beg);  // 预分配
      packedFile.put('\0');
      packedFile.seekp(0, packedFile.beg);  // 从头开始写入
      packedFile.write((char *)&readsCount, sizeof(uint32_t));  // 序列数
      packedFile.write((char *)packedOffsets.data(), sizeof(size_t)*readsCount);
      packedFile.write((char *)fastaOffsets.data(), sizeof(size_t)*readsCount);
    }
  }
  // 输出信息
  std::cout << "count:\t" << reads.size() << "\n";
  std::cout << "long:\t" << reads.front().readLength << "\n";
  std::cout << "short:\t" << reads.back().readLength << "\n";
}

const std::unordered_map<char, uint32_t> transTable = {  // 转码表
  {'a',  1}, {'b',  2}, {'c',  3}, {'d',  4}, {'e',  5}, {'f',  6}, {'g',  7},
  {'A',  1}, {'B',  2}, {'C',  3}, {'D',  4}, {'E',  5}, {'F',  6}, {'G',  7},
  {'h',  8}, {'i',  9}, {'j', 10}, {'k', 11}, {'l', 12}, {'m', 13}, {'n', 14},
  {'H',  8}, {'I',  9}, {'J', 10}, {'K', 11}, {'L', 12}, {'M', 13}, {'N', 14},
  {'o', 15}, {'p', 16}, {'q', 17}, {'r', 18}, {'s', 19}, {'t', 20}, {'u', 21},
  {'O', 15}, {'P', 16}, {'Q', 17}, {'R', 18}, {'S', 19}, {'T', 20}, {'U', 21},
  {'v', 22}, {'w', 23}, {'x', 24}, {'y', 25}, {'z', 26}, {'*', 27},
  {'V', 22}, {'W', 23}, {'X', 24}, {'Y', 25}, {'Z', 26}, {'-', 28}
};

//  makeData 生成数据 acgt -> 1100 1010
inline void packing(const std::string &read, std::vector<uint32_t> &packed,
const uint32_t mask, uint32_t *hashLine) {  // 打包序列
  uint32_t length = read.size();  // 长度
  uint32_t kmer = 0;  // 生成的k-mer
  packed.assign(1+(length+31)/32*5, 0);  // 写零
  packed[0] = length;  // 长度
  uint32_t netLength = 0;  // 净长度
  for (auto &base : read) {  // 打包 + 签名
    uint32_t pack = transTable.at(base);  // 编码
    for (uint32_t e=0; e<5; e++) {  // 打包数据
      packed[1+(netLength/32)*5+e] += (pack>>e&1)<<(netLength%32);
    }
    netLength += 1;
    kmer = ((kmer<<5)+pack) & mask;  // 生成K-mer
    uint64_t seed = kmer;  // 随机数种子
    for (uint32_t i=0; i<128; i++) {  // 签名
      seed = seed*48271%0x7FFFFFFF;  // 线性同余随机数
      hashLine[i] = std::min((uint32_t)seed, hashLine[i]);
    }
  }
  packed.resize(1+(length+31)/32*5);  // 裁剪
}

// packData 打包数据
void packData(const Option &option, std::vector<Read> &reads,
std::vector<uint32_t> &hashTable) {
  const uint32_t readsCount = reads.size();  // 序列数
  std::vector<size_t> packedOffsets(readsCount, 0);  // packed偏移
  std::vector<size_t> fastaOffsets(readsCount, 0);  // fasta偏移
  hashTable.assign((size_t)readsCount*128, 0xFFFFFFFF);  // 签名空间
  {  // 读packed和fasta偏移
    std::ifstream packedFile(option.packedFile);  // 打开文件
    packedFile.seekg(sizeof(uint32_t), packedFile.beg);
    packedFile.read((char *)packedOffsets.data(), sizeof(size_t)*readsCount);
    packedFile.read((char *)fastaOffsets.data(), sizeof(size_t)*readsCount);
  }
  std::cout << "pack:\t." << std::flush;  // 打印进度
  #pragma omp parallel  // 线程跑满
  { // 打包数据 拷贝数据 多线程
    std::ifstream inputFile(option.fastaFile);  // 输入
    std::ofstream packedFile(option.packedFile, std::ios::in);  // 输出packed
    std::ofstream fastaFile(option.packedFile, std::ios::in);  // 输出fasta
    std::string line="", name="", read="";  // 读入一行 序列名 序列数据
    std::vector<uint32_t> packed(1+65536/32*5, 0);  // 一行压缩后数据
    uint32_t mask = 0xFFFFFF;  // 短词掩码，太长稀疏遇不上，太短区分性不够
    if (readsCount > 0xFFFFFF) mask = 0xFFFFFFF;
    if (readsCount > 0xFFFFFFF) mask = 0xFFFFFFFF;
    #pragma omp for schedule(static, 16) // 并行任务 写packed数据 生成签名
    for (uint32_t i=0; i<readsCount; i++) {  // 遍历序列
      inputFile.seekg(reads[i].offset, inputFile.beg);  // 移到输入文件起始
      getline(inputFile, name);  // 读序列名
      read.clear();  // 序列数据
      while (inputFile.peek()!=EOF && inputFile.peek()!='>') {  // 读序列数据
        getline(inputFile, line);
        read += line;
      }
      packing(read, packed, mask, hashTable.data()+(size_t)i*128);  // 打包
      packedFile.seekp(packedOffsets[i], packedFile.beg);  // 移到起始
      packedFile.write((char *)packed.data(), sizeof(uint32_t)*packed.size());
      line = name + "\n" + read + "\n";  // 序列内容
      fastaFile.seekp(fastaOffsets[i], fastaFile.beg);  // 移到起始
      fastaFile.write((char *)line.c_str(), line.size());  // 写序列
      if ((i+1)%(1024*1024) == 0) {std::cout << "." << std::flush;}
    }
  }
  std::cout << " finish\n";  // 打印进度
}

// groupSequence 序列分组
void groupSequence(const Option &option, std::vector<uint32_t> &hashTable) {
  struct ReadSig {  // 一个序列的签名
    uint32_t index;  // 序列编号
    uint32_t signature;  // hash签名
  };  // 只在函数内应用
  const uint32_t readsCount = hashTable.size()/128;  // 序列数
  std::vector<ReadSig> readSigs(hashTable.size(), {0, 0});  // 序列签名
  std::cout << "group:\t" << std::flush;  // 打印进度 速度太快 不打印具体进度
  #pragma omp parallel for schedule(dynamic, 1)  // 每条线程负责一组签名
  for (uint32_t i=0; i<128; i++) {  // 遍历readSigs签名组
    for (uint32_t j=0; j<readsCount; j++) {  // 遍历组内序列 转置hashTable
      readSigs[i*readsCount+j].index = j;  // 编号
      readSigs[i*readsCount+j].signature = hashTable[j*128+i];  // 签名
    }
    std::stable_sort(readSigs.begin()+i*readsCount,
      readSigs.begin()+(i+1)*readsCount, [] (const ReadSig &a,
      const ReadSig &b) { return a.signature > b.signature; });  // 排序
  }
  #pragma omp parallel for schedule(dynamic, 1)  // 每个线程负责一组签名
  for (uint32_t i=0; i<128; i++) {  // 遍历readSigs签名组
    uint32_t represent = readSigs[i*readsCount].index;  // 代表序列
    uint32_t signature = readSigs[i*readsCount].signature;  // 原始签名
    for (uint32_t j=0; j<readsCount; j++) {  // 找代表序列
      if (signature != readSigs[i*readsCount+j].signature) {  // 签名变了
        represent = readSigs[i*readsCount+j].index;  // 更新代表序列
      }
      uint32_t target = readSigs[i*readsCount+j].index;  // 目标序列
      hashTable[i*readsCount+target] = represent;  // 写入代表序列
    }
  }
  std::cout << "finish\n";  // 打印进度
  // 写入文件
  std::ofstream groupFile(option.packedFile, std::ios::in);  // 输出hashTable
  size_t offset = sizeof(uint32_t)+sizeof(size_t)*readsCount*2;  // 偏移
  groupFile.seekp(offset, groupFile.beg);
  groupFile.write((char *)hashTable.data(), sizeof(uint32_t)*hashTable.size());
}

//--------主函数--------//
void makeDB(int argc, char **argv) {
  Timer::Timer timer;  // 计时器
  timer.printStamp();  // 时间戳
  Option option = {fastaFile:"", packedFile:""};  // 选项
  init(argc, argv, option);  // 初始化
  std::vector<Read> reads(0);  // 序列集合
  makeIndex(option, reads);  // 生成文件索引
  std::vector<uint32_t> hashTable(0);  // 签名表
  packData(option, reads, hashTable);  // 打包数据
  groupSequence(option, hashTable);  // 序列分组
  timer.printStamp();  // 时间戳
  timer.printDuration();  // 耗时
}

}  // namespace MakeDB
#endif  // MAKEDB_HPP
