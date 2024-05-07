/*
makeDB.cpp
输出文件数据内容:
  int32_t 序列的熵
  int32_t 序列数
  vector<size_t> packed数据偏移
  vector<size_t> fasta数据偏移
  packed数据
    uint32_t 长度
    uint32_t 净长度
    uint32_t 压缩数据
  fasta数据
    string 序列名
    string 序列
用法:
makeDB -f fasta文件 -p packed文件 -t (0:基因 1:蛋白序)
2024-05-05 by 鞠震
*/
// 为支持TB大小数据集 不能把数据都读入内存 要先生成索引
// 小数据集会自动缓存入内存 大数据集无法缓存入内存 因此无需为优化小数据集把数据读入内存
// 序列长超过65536 比对时候 line[2048]不够用
// 序列数超过20亿 比对时候线程数超过int范围

#include <iostream>  // cout
#include <vector>  // vector
#include <fstream>  // fstream
#include <algorithm>  // sort
#include <unordered_map>  // unordered_map
#include <omp.h>  // openmp
#include "parser.h"
#include "timer.h"

//--------数据--------//
struct Option {  // 输入选项
  std::string fastaFile;  // fasta文件
  std::string packedFile;  // packed文件
  int32_t entropy;  // 数据的熵 基因2 蛋白5
};

struct Read {  // 记录一条序列的位置 为了排序
  size_t offset;  // 序列位置
  int32_t nameLength;  // 序列名长度
  int32_t readLength;  // 序列长度 <=65536
};

//--------函数--------//
// parse 解析输入选项
void parse(int argc, char **argv, Option &option) {
  {  // 解析命令行
    Parser::Parser parser;  // 解析器
    parser.add("fasta", "-f", "fasta file", "string", "", true);
    parser.add("packed", "-p", "packed file", "string", "", true);
    if (!parser.parse(argc, argv)) exit(0);  // 解析
    option.fastaFile = parser.getString("fasta");  // fasta文件
    option.packedFile = parser.getString("packed");  // packed文件
  }
  {  // 校验参数 判断gene/protein
    std::ifstream fastaFile(option.fastaFile);  // fasta文件
    if (!fastaFile.is_open()) {  // 判断文件存在
      std::cout << option.fastaFile << " not exists\n";
      exit(0);
    } else {  // 判断gene/protein
      if (fastaFile.peek() == EOF) {std::cout << "empty file\n"; exit(0);}
      std::string line;  // 一行数据
      int32_t count = 0;  // 核酸/氨基酸个数
      getline(fastaFile, line);  // 序列名
      getline(fastaFile, line);  // 序列数据
      if (line.back() == '\r') line.pop_back();  // 去除\r
      for (char a:line) count += a=='a'||a=='A'||a=='c'||a=='C'||a=='g'||
        a=='G'||a=='t'||a=='T'||a=='u'||a=='U'||a=='n'||a=='N'||a=='-'?1:0;
      option.entropy = count>line.size()*0.9?3:5;  // gene:2 protein:5
    }
    fastaFile.close();
  }
  std::cout << "fasta:\t" << option.fastaFile << "\n";
  std::cout << "packed:\t" << option.packedFile << "\n";
  if (option.entropy == 3) {  // 基因
    std::cout << "type:\tgene\n";
  } else {  // 蛋白
    std::cout << "type:\tprotein\n";
  }
}

// makeIndex 文件索引
void makeIndex(const Option &option, std::vector<Read> &reads) {
  {  // 读数据
    std::ifstream fastaFile(option.fastaFile);  // 输入
    std::string line;  // 读入的一行
    Read read;  // 一条数据
    std::cout << "read:\t." << std::flush;  // 进度条
    while(fastaFile.peek() != EOF) {  // 读到文件结束
      read.offset = fastaFile.tellg();  // fasta起始位置
      getline(fastaFile, line);  // 读序列名
      if (line.back() == '\r') line.pop_back();  // 去除\r
      read.nameLength = line.size();
      read.readLength = 0;  // 序列数据长度清零
      while (fastaFile.peek() != EOF && fastaFile.peek() != '>') {  // 读序列
        getline(fastaFile, line);
        if (line.back() == '\r') line.pop_back();  // 去除\r
        read.readLength += line.size();
      }
      if (read.readLength <= 65536) reads.push_back(read);  // 不超65536 入队
      if(reads.size()%(1024*1024) == 0) std::cout << "." << std::flush;  // 进度
    }
    fastaFile.close();
    std::cout << " finish\n";
  }
  // 排序 并取前2G(20亿)个序列
  std::stable_sort(reads.begin(), reads.end(), [](const Read &a, const Read &b)
    {return a.readLength > b.readLength;});
  if (reads.size() >= 0x7FFFFFFF) reads.resize(0x7FFFFFFF);  // 不超2G条
  reads.shrink_to_fit();  // 省点内存
  std::cout << "count:\t" << reads.size() << "\n";
  std::cout << "long:\t" << reads.front().readLength << "\n";
  std::cout << "short:\t" << reads.back().readLength << "\n";
}

const std::unordered_map<char, uint32_t> transTableGen = {  // 基因转码表
  {'a',1}, {'c',2}, {'g',3}, {'t',4}, {'u',4},
  {'A',1}, {'C',2}, {'G',3}, {'T',4}, {'U',4}
};  // 只用于makeData函数
const std::unordered_map<char, uint32_t> transTablePro = {  // 蛋白转码表
  {'a', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5}, {'g', 6}, {'h', 7},
  {'A', 1}, {'C', 2}, {'D', 3}, {'E', 4}, {'F', 5}, {'G', 6}, {'H', 7},
  {'i', 8}, {'k', 9}, {'l',10}, {'m',11}, {'n',12}, {'o',13}, {'p',14},
  {'I', 8}, {'K', 9}, {'L',10}, {'M',11}, {'N',12}, {'O',13}, {'P',14},
  {'q',15}, {'r',16}, {'s',17}, {'t',18}, {'u',19}, {'v',20}, {'w',21},
  {'Q',15}, {'R',16}, {'T',17}, {'T',18}, {'U',19}, {'V',20}, {'W',21},
  {'y',22},
  {'Y',22}
};  // 只用于makeData函数

// makeData 生成数据 acgt -> 1010 1100
template <int32_t entropy>  // 基因熵3 蛋白熵5
void makeData(const std::string &read, std::vector<uint32_t> &buffer) {
  buffer.assign(2+(read.size()+31)/32*entropy, 0);  // 初始化
  uint32_t packs[entropy] = {0};  // 打包后数据 编译会展开成寄存器
  uint32_t *packed = buffer.data()+2;  // 打包后数据存储位置
  uint32_t netLength = 0;  // 净长度
  const std::unordered_map<char, uint32_t> *transTable;  // 转码表
  transTable = entropy==3?&transTableGen:&transTablePro;
  for (auto base:read) {
    auto iterator = (*transTable).find(base);  // 查找结果
    if (iterator == (*transTable).end()) continue;  // 未知碱基/氨基酸
    uint32_t pack = iterator->second;  // 编码
    for (int32_t e=0; e<entropy; e++) packs[e] >>= 1;
    for (int32_t e=0; e<entropy; e++) packs[e] += (pack>>e&1)<<31;
    netLength += 1;
    if (netLength%32 == 0) {  // 每32个氨基酸存储一次
      for (int32_t e=0; e<entropy; e++) *(packed+e) = packs[e];
      packed += entropy;
    }
  }
  if (netLength%32 > 0) {  // 需要补齐
    for (int32_t e=0; e<entropy; e++) packs[e] >>= (32-netLength%32);
    for (int32_t e=0; e<entropy; e++) *(packed+e) = packs[e];
  }
  buffer[0] = read.size();  // 长度
  buffer[1] = netLength;  // 净长度
}

// makeDB 生成数据库
void makeDB(const Option &option, std::vector<Read> &reads) {
  int32_t readsCount = reads.size();  // 序列数
  std::vector<size_t> inputOffsets(readsCount);  // 输入文件偏移
  std::vector<size_t> packedOffsets(readsCount);  // packed偏移
  std::vector<size_t> fastaOffsets(readsCount);  // fasta偏移
  std::ofstream packedFile(option.packedFile);  // 输出文件
  {  // 计算偏移 写入 序列类型 序列数 packed偏移 fasta偏移
    for (int32_t i=0; i<readsCount; i++) inputOffsets[i] = reads[i].offset;
    size_t offset = sizeof(int32_t)*2+sizeof(size_t)*readsCount*2;  // 基础偏移
    for (int32_t i=0; i<readsCount; i++) {  // packed偏移
      packedOffsets[i] = offset;
      offset += sizeof(uint32_t)*(2+(reads[i].readLength+31)/32*option.entropy);
    }
    for (int32_t i=0; i<readsCount; i++) {  // fasta偏移
      fastaOffsets[i] = offset;
      offset += reads[i].nameLength+reads[i].readLength+2;
    }
    packedFile.write((char*)&option.entropy, sizeof(int32_t));  // 序列的熵
    packedFile.write((char*)&readsCount, sizeof(int32_t));  // 序列数
    packedFile.write((char*)packedOffsets.data(), sizeof(size_t)*readsCount);
    packedFile.write((char*)fastaOffsets.data(), sizeof(size_t)*readsCount);
  }
  reads.resize(0); reads.shrink_to_fit();  // 省点内存
  packedFile.close();
  #pragma omp parallel
  {  // 打包数据 拷贝数据 90%以上耗时在这里 需要多核处理器
    std::ifstream inputFile(option.fastaFile);  // 输入
    std::ofstream packedFile(option.packedFile, std::ios::in);  // 输出packed
    std::ofstream fastaFile(option.packedFile, std::ios::in);  // 输出fasta
    std::string line, name, read;  // 读入一行 序列名 序列数据
    std::vector<uint32_t> buffer;  // 生成的数据
    #pragma omp master
    {std::cout << "pack:\t." << std::flush;}
    #pragma omp for schedule (dynamic)  // 并行任务 写packed数据
    for (int32_t i=0; i<readsCount; i++) {  // 遍历序列
      inputFile.seekg(inputOffsets[i], std::ios::beg);  // 移到输入文件起始
      getline(inputFile, name);  // 读序列名
      if (name.back() == '\r') name.pop_back();  // 去除\r
      read.clear();  // 用前先清空
      while (inputFile.peek() != EOF && inputFile.peek() != '>') {  // 读序列
        getline(inputFile, line);
        if (line.back() == '\r') line.pop_back();  // 去除\r
        read += line;
      }
      if (option.entropy == 3) {  // 基因序列
        makeData<3>(read, buffer);
      } else {  // 蛋白序列
        makeData<5>(read, buffer);
      }
      name += '\n';
      read += '\n';
      packedFile.seekp(packedOffsets[i], std::ios::beg);  // 移到packed文件起始
      packedFile.write((char*)buffer.data(), sizeof(uint32_t)*buffer.size());
      fastaFile.seekp(fastaOffsets[i], std::ios::beg);  // 移到fasta文件起始
      fastaFile.write((char*)name.c_str(), name.size());  // 写序列名
      fastaFile.write((char*)read.c_str(), read.size());  // 写序列
      if (i%(1024*1024) == 0) std::cout << "." << std::flush;  // 打印进度
    }
    #pragma omp master
    {std::cout << " finish\n";}
    inputFile.close();
    packedFile.close();
    fastaFile.close();
  }
}

//--------主函数--------//
int main(int argc, char **argv) {
  Timer::Timer timer;  // 开始计时
  Option option;  // 输入选项
  parse(argc, argv, option);  // 解析输入选项
  std::vector<Read> reads;  // 序列长度与偏移
  makeIndex(option, reads);  // 生成文件索引
  makeDB(option, reads);  // 生成数据库
  timer.getDuration();  // 结束计时
  timer.getTimeNow();  // 时间戳
}
