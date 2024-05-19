/*
makeDB.cpp
输出文件数据内容:
  uint32_t 序列的熵 * 1
  uint32_t 序列数 * 1
  vector<uint32_t> 序列名长度 * readsCount
  vector<uint32_t> 序列数据长度 * readsCount
  vector<size_t> packed数据偏移 * readsCount
  vector<size_t> fasta数据偏移 * readsCount
  hashTable数据
    uint32_t hash签名 * readsCount*64
  packed数据
    uint32_t 数据长度 * 1
    uint32_t 净长度 * 1
    uint32_t 压缩数据 * n
  fasta数据
    string 序列名
    string 序列
用法:
makeDB -f fasta文件 -p packed文件
2024-05-xx by 鞠震
*/
// 为支持TB大小数据集 不能把数据都读入内存 要先生成索引
// 小数据集会自动缓存入内存 大数据集无法缓存入内存 因此无需为优化小数据集把数据读入内存
// 序列长超过65536 比对时候 line[2048]不够用
// 序列数超过20亿 比对时候线程数超过int范围

#include <iostream>  // cout
#include <fstream>  // fstream
#include <vector>  // vector
#include <unordered_map>  // unordered_map
#include <algorithm>  // stable_sort
#include <omp.h>  // openmp
#include "parser.h"  // parser
#include "timer.h"  // timer

//--------数据--------//
struct Option {  // 输入选项
  std::string fastaFile;  // fasta文件
  std::string packedFile;  // packed文件
  uint32_t entropy;  // 数据的熵 基因3 蛋白5
};

struct Read {  // 记录一条序列的位置 为了排序
  size_t offset;  // 序列位置
  uint32_t nameLength;  // 序列名长度
  uint32_t readLength;  // 序列长度 <=65536
};

//--------函数--------//
// init 初始化
void init(int argc, char **argv, Option &option) {
  {  // 解析命令行
    Parser::Parser parser;  // 解析器
    parser.add("fasta", "-f", "fasta file", "string", "", true);  // fasta
    parser.add("packed", "-p", "packed file", "string", "", true);  // packed
    if (!parser.parse(argc, argv)) exit(0);  // 解析
    option.fastaFile = parser.getString("fasta");  // fasta文件
    option.packedFile = parser.getString("packed");  // packed文件
    std::cout << "fasta:\t" << option.fastaFile << "\n";
    std::cout << "packed:\t" << option.packedFile << "\n";
  }
  {  // 校验参数
    std::ifstream fastaFile(option.fastaFile);  // fasta文件
    if (!fastaFile.is_open()) {  // 判断文件存在
      std::cout << option.fastaFile << " not exists\n";
      exit(0);
    } else {  // 判断gene/protein
      std::string line = "";  // 一行数据
      uint32_t count = 0;  // 核酸/氨基酸个数
      getline(fastaFile, line);  // 序列名
      getline(fastaFile, line);  // 序列数据
      if (line.back() == '\r') line.pop_back();  // 去除\r
      for (char a:line) count += a=='a'||a=='A'||a=='c'||a=='C'||a=='g'||
        a=='G'||a=='t'||a=='T'||a=='u'||a=='U'||a=='n'||a=='N'||a=='-'?1:0;
      if (count >= line.size()*0.9f) option.entropy = 3;  // 基因
      if (count < line.size()*0.9f) option.entropy = 5;  // 蛋白
    }
    fastaFile.close();
    if (option.entropy == 3) std::cout << "type:\tgene\n";  // 基因
    if (option.entropy == 5) std::cout << "type:\tprotein\n";  // 蛋白
  }  // 用单独的if分别判断 gene/protein 后面改熵不容易出错
}

// makeIndex 文件索引
void makeIndex(const Option &option, std::vector<Read> &reads) {
  {  // 读数据
    std::ifstream fastaFile(option.fastaFile);  // 输入
    std::string line = "";  // 读入的一行
    Read read = {offset:0, nameLength:0, readLength:0};  // 一条数据
    std::cout << "read:\t." << std::flush;  // 进度条
    while(fastaFile.peek()!=EOF && reads.size()<0x7FFFFFFF) {  // 读完文件
      read.offset = fastaFile.tellg();  // fasta起始位置
      getline(fastaFile, line);  // 读序列名
      if (line.back() == '\r') line.pop_back();  // 去除\r
      read.nameLength = line.size();
      read.readLength = 0;  // 序列数据长度清零
      while (fastaFile.peek()!=EOF && fastaFile.peek()!='>') {  // 读序列
        getline(fastaFile, line);
        if (line.back() == '\r') line.pop_back();  // 去除\r
        read.readLength += line.size();
      }
      if (read.readLength <= 65536) reads.push_back(read);  // 不超65536 入队
      if ((reads.size()&1024*1024-1) == 0) std::cout << "." << std::flush;
    }
    fastaFile.close();
    std::cout << " finish\n";
    reads.shrink_to_fit();  // 省点内存
  }
  std::stable_sort(reads.begin(), reads.end(), [](const Read &a, const Read &b)
    {return a.readLength > b.readLength;});  // 排序
  std::cout << "count:\t" << reads.size() << "\n";
  std::cout << "long:\t" << reads.front().readLength << "\n";
  std::cout << "short:\t" << reads.back().readLength << "\n";
}

std::unordered_map<char, uint32_t> transTableGen = {  // 基因转码表
  {'a',1}, {'c',2}, {'g',3}, {'t',4}, {'u',4},
  {'A',1}, {'C',2}, {'G',3}, {'T',4}, {'U',4}
};  // 只用于packData函数
std::unordered_map<char, uint32_t> transTablePro = {  // 蛋白转码表
  {'a', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5}, {'g', 6}, {'h', 7},
  {'A', 1}, {'C', 2}, {'D', 3}, {'E', 4}, {'F', 5}, {'G', 6}, {'H', 7},
  {'i', 8}, {'k', 9}, {'l',10}, {'m',11}, {'n',12}, {'o',13}, {'p',14},
  {'I', 8}, {'K', 9}, {'L',10}, {'M',11}, {'N',12}, {'O',13}, {'P',14},
  {'q',15}, {'r',16}, {'s',17}, {'t',18}, {'u',19}, {'v',20}, {'w',21},
  {'Q',15}, {'R',16}, {'S',17}, {'T',18}, {'U',19}, {'V',20}, {'W',21},
  {'y',22},
  {'Y',22}
};  // 只用于packData函数
// makeData 生成数据 acgt -> 1010 1100
template <uint32_t entropy>
inline void packData(const std::string &read, std::vector<uint32_t> &packed) {
  packed.assign(2+(read.size()+31>>5)*entropy, 0);  // 初始化
  uint32_t packs[entropy] = {0};  // 打包后数据 编译会展开成寄存器
  uint32_t netLength = 0;  // 净长度
  std::unordered_map<char, uint32_t> *transTable;
  if (entropy == 3) transTable = &transTableGen;
  if (entropy == 5) transTable = &transTablePro;
  for (auto base:read) {
    auto iterator = (*transTable).find(base);  // 查找结果
    if (iterator == (*transTable).end()) continue;  // 未知碱基/氨基酸
    uint32_t pack = iterator->second;  // 编码
    for (uint32_t e=0; e<entropy; e++) {  // 打包数据
      packs[e] = ((pack>>e&1)<<31)+(packs[e]>>1);
    }
    netLength += 1;
    if ((netLength&31) == 0) {  // 每32个氨基酸存储一次
      for (uint32_t e=0; e<entropy; e++) {
        packed[2+((netLength>>5)-1)*entropy+e] = packs[e];
      }
    }
  }
  if ((netLength&31) > 0) {  // 补齐
    for (uint32_t e=0; e<entropy; e++) {
      packed[2+(netLength>>5)*entropy+e] = packs[e]>>32-(netLength&31);
    }
  }
  packed[0] = read.size();  // 长度
  packed[1] = netLength;  // 净长度
}

std::unordered_map<char, uint32_t> kmerTableGen = {  // 基因转码表
  {'a',0}, {'c',1}, {'g',2}, {'t',3}, {'u',3},
  {'A',0}, {'C',1}, {'G',2}, {'T',3}, {'U',3}
};  // 只用于hashData函数
std::unordered_map<char, uint32_t> kmerTablePro = {  // 蛋白转码表
  {'a', 0}, {'c', 1}, {'d', 2}, {'e', 3}, {'f', 4}, {'g', 5}, {'h', 6},
  {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3}, {'F', 4}, {'G', 5}, {'H', 6},
  {'i', 7}, {'k', 8}, {'l', 9}, {'m', 9}, {'n', 2}, {'o',10}, {'p',11},
  {'I', 7}, {'K', 8}, {'L', 9}, {'M', 9}, {'N', 2}, {'O',10}, {'P',11},
  {'q', 3}, {'r', 8}, {'s', 0}, {'t', 0}, {'u',12}, {'v', 7}, {'w',13},
  {'Q', 3}, {'R', 8}, {'S', 0}, {'T', 0}, {'U',12}, {'V', 7}, {'W',13},
  {'y', 4},
  {'Y', 4}
};  // 只用于hashData函数
// hashData 生成hash签名 基因4**8 蛋白16**4 = 65536
template <uint32_t entropy>
inline void hashData(const std::string &read, std::vector<uint32_t> &kmers,
std::vector<uint32_t> &hashLine) {
  {  // 生成k-mer
    kmers.assign(2048, 0);  // k-mer初始化
    hashLine.assign(64, 0);  // hash签名初始化
    uint32_t kmer = 0;  // 生成的k-mer
    std::unordered_map<char, uint32_t> *kmerTable;  // k-mer转码矩阵
    if (entropy == 2) kmerTable = &kmerTableGen;
    if (entropy == 4) kmerTable = &kmerTablePro;
    for (auto base:read) {  // 遍历read
      auto iterator = (*kmerTable).find(base);  // 查找结果
      if (iterator == (*kmerTable).end()) continue;  // 未知碱基/氨基酸
      kmer = ((kmer<<entropy)+iterator->second)&0xFFFF;  // 生成K-mer
      kmers[kmer/32] = (kmers[kmer>>5]&~(1<<(kmer&31)))+(1<<(kmer&31));  // 记录
    }
  }
  {  // 生成签名 index = (ax+a）%65536 a与65536互质 a=1,3,5,7,9...
    for (uint32_t i=0; i<64; i++) {  // 随机排列64次
      uint32_t a = i*2+1;  // 随机系数
      for (uint32_t j=0; j<65536; j++) {  // 遍历kmer记录
        uint32_t index = (a*j+a)&65535;  // 生成的随机数
        if ((kmers[index>>5]&(1<<(index&31))) > 0) {  // 找到第一个kmer
          hashLine[i] = index;
          break;
        }
      }
    }
  }
}

// makeDB 生成数据库
void makeDB(const Option &option, std::vector<Read> &reads) {
  uint32_t entropy = option.entropy;  // 熵
  uint32_t readsCount = reads.size();  // 序列数
  std::vector<size_t> inputOffsets(readsCount, 0);  // 输入文件偏移
  std::vector<size_t> packedOffsets(readsCount, 0);  // packed偏移
  std::vector<size_t> fastaOffsets(readsCount, 0);  // fasta偏移
  {  // 计算偏移 写入 熵 hashTable偏移 packed偏移 fasta偏移
    std::vector<uint32_t> nameLengths(readsCount, 0);  // 序列名长度
    std::vector<uint32_t> readLengths(readsCount, 0);  // 序列数据长度
    for (uint32_t i=0; i<readsCount; i++) {
      inputOffsets[i] = reads[i].offset;  // 输入文件的偏移
      nameLengths[i] = reads[i].nameLength;  // 序列名长度
      readLengths[i] = reads[i].readLength;  // 序列数据长度
    }
    std::ofstream packedFile(option.packedFile);  // 输出文件
    packedFile.write((char*)&entropy, sizeof(uint32_t));  // 序列的熵
    packedFile.write((char*)&readsCount, sizeof(uint32_t));  // 序列数
    packedFile.write((char*)nameLengths.data(), sizeof(uint32_t)*readsCount);
    packedFile.write((char*)readLengths.data(), sizeof(uint32_t)*readsCount);
    size_t offset = sizeof(uint32_t)*(2+readsCount*2);  // 当前指针
    offset += sizeof(size_t)*readsCount*2+sizeof(uint32_t)*64*readsCount;
    for (uint32_t i=0; i<readsCount; i++) {  // packed偏移
      packedOffsets[i] = offset;
      offset += sizeof(uint32_t)*(2+(readLengths[i]+31>>5)*option.entropy);
    }
    for (uint32_t i=0; i<readsCount; i++) {  // fasta偏移
      fastaOffsets[i] = offset;
      offset += nameLengths[i]+readLengths[i]+2;  // 包括换行
    }
    packedFile.write((char*)packedOffsets.data(), sizeof(size_t)*readsCount);
    packedFile.write((char*)fastaOffsets.data(), sizeof(size_t)*readsCount);
    packedFile.close();
    reads.resize(0); reads.shrink_to_fit();  // 省点内存
  }
  #pragma omp parallel
  {  // 打包数据 拷贝数据 多线程
    std::ifstream inputFile(option.fastaFile);  // 输入
    std::ofstream hashFile(option.packedFile, std::ios::in);  // 输出hashTable
    std::ofstream packedFile(option.packedFile, std::ios::in);  // 输出packed
    std::ofstream fastaFile(option.packedFile, std::ios::in);  // 输出fasta
    std::string line="", name="", read="";  // 读入一行 序列名 序列数据
    std::vector<uint32_t> packed(0);  // 压缩数据
    std::vector<uint32_t> kmers(2048, 0);  // k-mer统计结果
    std::vector<uint32_t> hashLine(64, 0);  // 一行哈希签名
    #pragma omp master
    {std::cout << "pack:\t." << std::flush;}  // 打印进度
    #pragma omp for schedule (dynamic)  // 并行任务 写packed数据 生成签名
    for (uint32_t i=0; i<readsCount; i++) {  // 遍历序列
      inputFile.seekg(inputOffsets[i], std::ios::beg);  // 移到输入文件起始
      getline(inputFile,name); if(name.back()=='\r')name.pop_back();  // 序列名
      read.clear();  // 序列数据
      while (inputFile.peek()!=EOF && inputFile.peek()!='>') {  // 读序列
        getline(inputFile, line); if (line.back() == '\r') line.pop_back();
        read += line;
      }
      if (option.entropy == 3) packData<3>(read, packed);  // 打包gene
      if (option.entropy == 5) packData<5>(read, packed);  // 打包protein
      if (option.entropy == 3) hashData<2>(read, kmers, hashLine);  // gene
      if (option.entropy == 5) hashData<4>(read, kmers, hashLine);  // protein
      size_t hashOffset = sizeof(uint32_t)*(2+readsCount*2);  // hashTable偏移
      hashOffset += sizeof(size_t)*readsCount*2+sizeof(uint32_t)*64*i;
      hashFile.seekp(hashOffset, std::ios::beg);
      hashFile.write((char*)hashLine.data(),sizeof(uint32_t)*hashLine.size());
      packedFile.seekp(packedOffsets[i], std::ios::beg);  // 移到packed文件起始
      packedFile.write((char*)packed.data(), sizeof(uint32_t)*packed.size());
      line = name+"\n"+read+"\n";
      fastaFile.seekp(fastaOffsets[i], std::ios::beg);  // 移到fasta文件起始
      fastaFile.write((char*)line.c_str(), line.size());  // 写序列
      if ((i+1&1024*1024-1) == 0) std::cout << "." << std::flush;  // 打印进度
    }
    #pragma omp master
    {std::cout << " finish\n";}
    inputFile.close();
    hashFile.close();
    packedFile.close();
    fastaFile.close();
  }
}

//--------主函数--------//
int main(int argc, char **argv) {
  Timer::Timer timer;  // 开始计时
  Option option = {fastaFile:"", packedFile:"", entropy:0};  // 选项
  init(argc, argv, option);  // 初始化
  std::vector<Read> reads(0);  // 序列集合
  makeIndex(option, reads);  // 生成文件索引
  makeDB(option, reads);  // 生成数据库
  timer.getDuration();  // 结束计时
  timer.getTimeNow();  // 时间戳
}

// 基因4**8 entroy = 2
// 蛋白16**4 entropy = 4
// 长度65536 64个随机排列 (ax+b)%65536
