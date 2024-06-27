/*
makeDB.cpp
输出文件数据内容:
  uint32_t 序列的熵 * 1
  uint32_t 序列数 * 1
  uint32_t hash签名数 * 1
  vector<uint32_t> 序列名长度 * readsCount
  vector<uint32_t> 序列数据长度 * readsCount
  vector<size_t> packed数据偏移 * readsCount
  vector<size_t> fasta数据偏移 * readsCount
  hashTable数据
    uint32_t hash签名 * readsCount*hash签名数
  packed数据 (序列数条记录)
    uint32_t 数据长度 * 1
    uint32_t 净长度 * 1
    uint32_t 压缩数据 * (数据长度+31)/32*熵
  fasta数据 (序列数条记录)
    string 序列名 * 1
    string 序列 * 1
用法:
makeDB -f fasta文件 -p packed文件
2024-06-20 by 鞠震
*/

// 为支持TB大小数据集 不能把数据都读入内存 要先生成索引
// 小数据集会自动缓存入内存 大数据集无法缓存入内存
// 因此无需为优化小数据集把数据读入内存 序列长超过65536 比对时候
// line[2048]不够用 序列数超过20亿 比对时候线程数可能超过int范围

#include "parser.h"      // parser
#include "timer.h"       // timer
#include <algorithm>     // stable_sort
#include <fstream>       // fstream
#include <iostream>      // cout
#include <omp.h>         // openmp
#include <unordered_map> // unordered_map
#include <vector>        // vector
#define SIGNEDCOUNT 64   // 签名尺寸 越大越准 速度越慢

//--------数据--------//
struct Option {           // 输入选项
  std::string fastaFile;  // fasta文件
  std::string packedFile; // packed文件
  uint32_t entropy;       // 数据的熵 基因3 蛋白5
};

struct Read {          // 记录一条序列的位置 为了排序
  size_t offset;       // 序列位置
  uint32_t nameLength; // 序列名长度
  uint32_t readLength; // 序列长度 <=65536
};

//--------函数--------//
// init 初始化
void init(int argc, char **argv, Option &option) {
  {                        // 解析命令行
    Parser::Parser parser; // 解析器
    parser.add("fasta", "-f", "fasta file", "string", "", true);   // fasta
    parser.add("packed", "-p", "packed file", "string", "", true); // packed
    if (!parser.parse(argc, argv)) { // 解析命令行
      exit(0);
    }
    option.fastaFile = parser.getString("fasta");   // fasta文件
    option.packedFile = parser.getString("packed"); // packed文件
    std::cout << "fasta:\t" << option.fastaFile << "\n";
    std::cout << "packed:\t" << option.packedFile << "\n";
  }
  {                                            // 校验参数
    std::ifstream fastaFile(option.fastaFile); // fasta文件
    if (!fastaFile.is_open()) {                // 判断文件存在
      std::cout << option.fastaFile << " not exists\n";
      exit(0);
    }
    // 判断 gene/protein 熵分别单独判断 改熵不容易出错
    std::string line = "";     // 一行数据
    uint32_t count = 0;        // 核酸/氨基酸个数
    getline(fastaFile, line);  // 序列名
    getline(fastaFile, line);  // 序列数据
    if (line.back() == '\r') { // 去除行尾\r
      std::cout << "remove \\r at the line";
      exit(0);
    }
    std::vector<char> bases = {'a', 'A', 'c', 'C', 'g', 'G', 't',
                               'T', 'u', 'U', 'n', 'N', '-'};
    for (char a : line) { // 统计基因个数
      count += std::find(bases.begin(), bases.end(), a) != bases.end() ? 1 : 0;
    }
    fastaFile.close();
    option.entropy = count >= line.size() * 0.9f ? 3 : option.entropy; // 基因
    option.entropy = count < line.size() * 0.9f ? 5 : option.entropy;  // 蛋白
    const std::string types[6] = {"", "", "", "gene", "", "protein"};
    std::cout << "type:\t" << types[option.entropy] << "\n"; // 基因/蛋白
  }
}

// makeIndex 文件索引
void makeIndex(const Option &option, std::vector<Read> &reads) {
  std::ifstream fastaFile(option.fastaFile);                // 输入
  std::string line = "";                                    // 读入的一行
  Read read = {offset : 0, nameLength : 0, readLength : 0}; // 一条数据
  std::cout << "read:\t." << std::flush;                    // 进度条
  while (fastaFile.peek() != EOF && reads.size() < 0x7FFFFFFF) { // 读完文件
    read.offset = fastaFile.tellg(); // fasta起始位置
    getline(fastaFile, line);        // 读序列名
    read.nameLength = line.size();
    read.readLength = 0; // 序列数据长度清零
    while (fastaFile.peek() != EOF && fastaFile.peek() != '>') { // 读序列
      getline(fastaFile, line);
      read.readLength += line.size();
    }
    if (read.readLength < 0xFFFF) {            // 不超65536
      reads.push_back(read);                   // 入队
      if (reads.size() % (1024 * 1024) == 0) { // 打印进度
        std::cout << "." << std::flush;
      }
    }
  }
  fastaFile.close();
  std::cout << " finish\n";
  reads.shrink_to_fit(); // 省点内存
  std::stable_sort(reads.begin(), reads.end(),
                   [](const Read &a, const Read &b) {
                     return a.readLength > b.readLength;
                   }); // 排序
  std::cout << "count:\t" << reads.size() << "\n";
  std::cout << "long:\t" << reads.front().readLength << "\n";
  std::cout << "short:\t" << reads.back().readLength << "\n";
}

std::unordered_map<char, uint32_t> transTableGen = { // 基因转码表
    {'a', 1}, {'c', 2}, {'g', 3}, {'t', 4}, {'u', 4},
    {'A', 1}, {'C', 2}, {'G', 3}, {'T', 4}, {'U', 4}}; // 只用于packData函数
std::unordered_map<char, uint32_t> transTablePro = {   // 蛋白转码表
    {'a', 1},  {'c', 2},  {'d', 3},  {'e', 4},  {'f', 5},  {'g', 6},  {'h', 7},
    {'A', 1},  {'C', 2},  {'D', 3},  {'E', 4},  {'F', 5},  {'G', 6},  {'H', 7},
    {'i', 8},  {'k', 9},  {'l', 10}, {'m', 11}, {'n', 12}, {'o', 13}, {'p', 14},
    {'I', 8},  {'K', 9},  {'L', 10}, {'M', 11}, {'N', 12}, {'O', 13}, {'P', 14},
    {'q', 15}, {'r', 16}, {'s', 17}, {'t', 18}, {'u', 19}, {'v', 20}, {'w', 21},
    {'Q', 15}, {'R', 16}, {'S', 17}, {'T', 18}, {'U', 19}, {'V', 20}, {'W', 21},
    {'y', 22}, {'Y', 22}}; // 只用于packData函数
// makeData 生成数据 acgt -> 1010 1100
template <uint32_t entropy>
inline void packData(const std::string &read, std::vector<uint32_t> &packed) {
  packed.assign(2 + (read.size() + 31) / 32 * entropy, 0); // 初始化
  uint32_t packs[entropy] = {0}; // 打包后数据 编译会展开成寄存器
  uint32_t netLength = 0;        // 净长度
  std::unordered_map<char, uint32_t> *transTable = NULL; // 转码表
  const void *tables[6] = {0, 0, 0, &transTableGen, 0, &transTablePro};
  transTable = (std::unordered_map<char, uint32_t> *)tables[entropy];
  for (char base : read) {
    auto iterator = (*transTable).find(base);  // 查找
    if (iterator != (*transTable).end()) {     // 找到记录
      uint32_t pack = iterator->second;        // 编码
      for (uint32_t e = 0; e < entropy; e++) { // 打包数据
        packs[e] = ((pack >> e & 1) << 31) + (packs[e] >> 1);
      }
      netLength += 1;
      if (netLength % 32 == 0) { // 每32个氨基酸存储一次
        for (uint32_t e = 0; e < entropy; e++) {
          packed[2 + (netLength / 32 - 1) * entropy + e] = packs[e];
        }
      }
    }
  }
  if (netLength % 32 > 0) { // 补齐
    for (uint32_t e = 0; e < entropy; e++) {
      packed[2 + netLength / 32 * entropy + e] =
          packs[e] >> 32 - (netLength % 32);
    }
  }
  packed[0] = read.size(); // 长度
  packed[1] = netLength;   // 净长度
}

std::unordered_map<char, uint32_t> kmerTableGen = { // 基因转码表
    {'a', 0}, {'c', 1}, {'g', 2}, {'t', 3}, {'u', 3},
    {'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}, {'U', 3}}; // 只用于hashData函数
std::unordered_map<char, uint32_t> kmerTablePro = {    // 蛋白转码表
    {'a', 0}, {'c', 1}, {'d', 2}, {'e', 3}, {'f', 4},  {'g', 5},  {'h', 6},
    {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3}, {'F', 4},  {'G', 5},  {'H', 6},
    {'i', 7}, {'k', 8}, {'l', 9}, {'m', 9}, {'n', 2},  {'o', 10}, {'p', 11},
    {'I', 7}, {'K', 8}, {'L', 9}, {'M', 9}, {'N', 2},  {'O', 10}, {'P', 11},
    {'q', 3}, {'r', 8}, {'s', 0}, {'t', 0}, {'u', 12}, {'v', 7},  {'w', 13},
    {'Q', 3}, {'R', 8}, {'S', 0}, {'T', 0}, {'U', 12}, {'V', 7},  {'W', 13},
    {'y', 4}, {'Y', 4}};      // 只用于hashData函数
const uint32_t aArray[64] = { // 类似线性同余 kmer = (a*index+a)%n, an互质
    1,       30031,   60061,   90091,   120121,  150151,  180181,  210211,
    240241,  270271,  300301,  330331,  360361,  390391,  420421,  450451,
    480481,  510511,  540541,  570571,  600601,  630631,  660661,  690691,
    720721,  750751,  780781,  810811,  840841,  870871,  900901,  930931,
    960961,  990991,  1021021, 1051051, 1081081, 1111111, 1141141, 1171171,
    1201201, 1231231, 1261261, 1291291, 1321321, 1351351, 1381381, 1411411,
    1441441, 1471471, 1501501, 1531531, 1561561, 1591591, 1621621, 1651651,
    1681681, 1711711, 1741741, 1771771, 1801801, 1831831, 1861861, 1891891};
const uint32_t afArray[64] = { // 线性同余求逆 index = af*(kmer-a)%n, (af*a)%n=1
    1,         58556847,  153938869, 114282691, 259295497, 98327351,  151421309,
    234267275, 160618129, 118476863, 115687877, 2954707,   222401689, 158665927,
    151853197, 216012443, 13692705,  152550607, 237072853, 43061475,  91656745,
    225315927, 209795485, 254931115, 201959857, 30713695,  98753509,  172040691,
    16703929,  181058023, 87049389,  245731515, 94660161,  233697263, 212000757,
    172665091, 211673929, 45512055,  169843133, 159143627, 262178001, 32700031,
    131259909, 67307027,  74381017,  168075015, 252483789, 33530587,  118842721,
    232463119, 56374805,  243767587, 258309225, 78681751,  200057309, 234935531,
    129132529, 8764831,   18892837,  238928435, 181195769, 22920231,  74257645,
    106515707};
template <uint32_t entropy> // 2:基因 4:蛋白
inline void hashData(const std::string &read, std::vector<uint32_t> &indexs,
                     std::vector<uint32_t> &hashLine) {
  const uint32_t signedCount = hashLine.size(); // 签名尺寸 越大越准 速度越慢
  uint32_t kmer = 0;                            // 生成的k-mer
  std::unordered_map<char, uint32_t> *kmerTable = NULL; // k-mer表
  const void *tables[5] = {0, 0, &kmerTableGen, 0, &kmerTablePro};
  kmerTable = (std::unordered_map<char, uint32_t> *)tables[entropy];
  indexs.assign(signedCount, 0xFFFFFFFF);       // 初始都排最后
  for (uint32_t i = 0; i < read.size(); i++) {  // 遍历read
    auto iterator = (*kmerTable).find(read[i]); // 查找结果
    if (iterator != (*kmerTable).end()) {       // 找到了 碱基/氨基酸
      kmer = ((kmer << entropy) + iterator->second) & 0xFFFFFFF; // 生成K-mer
      for (uint32_t j = 0; j < signedCount; j++) { // 查找最早kmer
        const uint32_t a = aArray[j];
        const uint32_t af = afArray[j];
        const uint32_t index = af * (kmer - a) & 0xFFFFFFF;
        hashLine[j] = index < indexs[j] ? kmer : hashLine[j]; // 更新kmer
        indexs[j] = std::min(index, indexs[j]);               // 更新index
      }
    } // 测试发现 不跳过前6个碱基效果更好
  }
}

// makeDB 生成数据库
void makeDB(const Option &option, std::vector<Read> &reads) {
  const uint32_t entropy = option.entropy;  // 熵
  const uint32_t readsCount = reads.size(); // 序列数
  const uint32_t signedCount = SIGNEDCOUNT; // 签名尺寸 越大越准 速度越慢
  std::vector<size_t> inputOffsets(readsCount, 0);  // 输入文件偏移
  std::vector<size_t> packedOffsets(readsCount, 0); // packed偏移
  std::vector<size_t> fastaOffsets(readsCount, 0);  // fasta偏移
  { // 计算偏移 写入 熵 hashTable偏移 packed偏移 fasta偏移
    std::vector<uint32_t> nameLengths(readsCount, 0); // 序列名长度
    std::vector<uint32_t> readLengths(readsCount, 0); // 序列数据长度
    for (uint32_t i = 0; i < readsCount; i++) {
      inputOffsets[i] = reads[i].offset;    // 输入文件的偏移
      nameLengths[i] = reads[i].nameLength; // 序列名长度
      readLengths[i] = reads[i].readLength; // 序列数据长度
    }
    reads.resize(0);
    reads.shrink_to_fit();                                    // 省点内存
    std::ofstream packedFile(option.packedFile);              // 输出文件
    packedFile.write((char *)&entropy, sizeof(uint32_t));     // 序列的熵
    packedFile.write((char *)&readsCount, sizeof(uint32_t));  // 序列数
    packedFile.write((char *)&signedCount, sizeof(uint32_t)); // hash签名大小
    packedFile.write((char *)nameLengths.data(), sizeof(uint32_t) * readsCount);
    packedFile.write((char *)readLengths.data(), sizeof(uint32_t) * readsCount);
    size_t offset = packedFile.tellp(); // 当前指针
    offset += sizeof(size_t) * readsCount * 2 +
              sizeof(uint32_t) * readsCount * signedCount; // packed起始位置
    for (uint32_t i = 0; i < readsCount; i++) {            // packed偏移
      packedOffsets[i] = offset;
      offset += sizeof(uint32_t) * (2 + (readLengths[i] + 31) / 32 * entropy);
    }
    for (uint32_t i = 0; i < readsCount; i++) { // fasta偏移
      fastaOffsets[i] = offset;
      offset += nameLengths[i] + readLengths[i] + 2; // 包括换行
    }
    packedFile.write((char *)packedOffsets.data(), sizeof(size_t) * readsCount);
    packedFile.write((char *)fastaOffsets.data(), sizeof(size_t) * readsCount);
    packedFile.close();
  }
#pragma omp parallel proc_bind(close)
  { // 打包数据 拷贝数据 多线程
    std::ifstream inputFile(option.fastaFile);               // 输入
    std::ofstream hashFile(option.packedFile, std::ios::in); // 输出hashTable
    std::ofstream packedFile(option.packedFile, std::ios::in); // 输出packed
    std::ofstream fastaFile(option.packedFile, std::ios::in);  // 输出fasta
    std::string line = "", name = "", read = ""; // 读入一行 序列名 序列数据
    std::vector<uint32_t> packed(65536 / 32 * 5 + 2, 0); // 压缩数据
    std::vector<uint32_t> indexs(signedCount, 0);   // 哈希签名的序号
    std::vector<uint32_t> hashLine(signedCount, 0); // 一行哈希签名
#pragma omp master
    { std::cout << "pack:\t." << std::flush; } // 打印进度
    const size_t hashOffset = sizeof(uint32_t) * (3 + readsCount * 2) +
                              sizeof(size_t) * readsCount * 2; // hashTable偏移
#pragma omp for schedule(dynamic) // 并行任务 写packed数据 生成签名
    for (uint32_t i = 0; i < readsCount; i++) {        // 遍历序列
      inputFile.seekg(inputOffsets[i], std::ios::beg); // 移到输入文件起始
      getline(inputFile, name);
      read.clear(); // 序列数据
      while (inputFile.peek() != EOF && inputFile.peek() != '>') { // 读序列数据
        getline(inputFile, line);
        read += line;
      }
      if (entropy == 3) { // 打包+签名 gene
        packData<3>(read, packed);
        hashData<2>(read, indexs, hashLine);
      }
      if (entropy == 5) { // 打包+签名 protein
        packData<5>(read, packed);
        hashData<4>(read, indexs, hashLine);
      }
      size_t hashOffsetI = hashOffset + sizeof(uint32_t) * signedCount * i;
      hashFile.seekp(hashOffsetI, std::ios::beg);
      hashFile.write((char *)hashLine.data(), sizeof(uint32_t) * signedCount);
      packedFile.seekp(packedOffsets[i], std::ios::beg); // 移到packed文件起始
      packedFile.write((char *)packed.data(), sizeof(uint32_t) * packed.size());
      line = name + "\n" + read + "\n";
      fastaFile.seekp(fastaOffsets[i], std::ios::beg); // 移到fasta文件起始
      fastaFile.write((char *)line.c_str(), line.size()); // 写序列
      if ((i + 1) % (1024 * 1024) == 0) {                 // 打印进度
        std::cout << "." << std::flush;
      }
    }
#pragma omp master
    { std::cout << " finish\n"; }
    inputFile.close();
    hashFile.close();
    packedFile.close();
    fastaFile.close();
  }
}

//--------主函数--------//
int main(int argc, char **argv) {
  Timer::Timer timer; // 开始计时
  Option option = {fastaFile : "", packedFile : "", entropy : 0}; // 选项
  init(argc, argv, option);                                       // 初始化
  std::vector<Read> reads(0); // 序列集合
  makeIndex(option, reads);   // 生成文件索引
  makeDB(option, reads);      // 生成数据库
  timer.getDuration();        // 结束计时
  timer.getTimeNow();         // 时间戳
}
