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
    uint32_t hash签名 * readsCount * hash签名数
  packed数据 (序列数条记录)
    uint32_t 数据长度 * 1
    uint32_t 净长度 * 1
    uint32_t 压缩数据 * (数据长度+31)/32 * 熵
  fasta数据 (序列数条记录)
    string 序列名 * 1
    string 序列 * 1
用法:
makeDB -f fasta文件 -p packed文件
2024-07-14 by 鞠震
*/

// 为支持TB数据集 不能把数据都读入内存 要先生成索引
// 序列长超过65535 比对时 line[2048]不够用
// 序列数超过20亿 比对时候线程数可能超过int范围

#include <algorithm>  // stable_sort
#include <cstring>  // memset
#include <fstream>  // fstream
#include <iostream>  // cout
#include <omp.h>  // openmp
#include <unordered_map> // unordered_map
#include <vector>  // vector
#include "parser.h"  // parser
#include "timer.h"  // timer

#define SIGNEDCOUNT 66  // 签名尺寸 保证相似度0.1时 准确率大于0.99

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
    if (!parser.parse(argc, argv)) exit(0);  // 解析命令行
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
    }
    // 判断 gene/protein 熵分别单独判断 改熵不容易出错
    std::string line = "";  // 一行数据
    getline(fastaFile, line);  // 序列名
    getline(fastaFile, line);  // 序列数据
    fastaFile.close();
    if (line.back() == '\r') {  // 去除行尾\r
      std::cout << "remove \\r at the end of lines\n";
      exit(0);
    }
    const char chrS[13] = {'a','A','c','C','g','G','t','T','u','U','n','N','-'};
    uint32_t count = 0;  // 核酸/氨基酸个数
    for (char chr : line) { // 统计基因个数
      count += std::find(chrS, chrS+13, chr) != chrS + 13 ? 1 : 0;
    }
    option.entropy = count >= line.size() * 0.9f ? 3 : option.entropy; // 基因
    option.entropy = count < line.size() * 0.9f ? 5 : option.entropy;  // 蛋白
    const std::string types[6] = {"", "", "", "gene", "", "protein"};
    std::cout << "type:\t" << types[option.entropy] << "\n"; // 基因/蛋白
  }
}

// makeIndex 文件索引
void makeIndex(const Option &option, std::vector<Read> &reads) {
  std::ifstream fastaFile(option.fastaFile);  // 输入
  std::string line = "";  // 读入的一行
  Read read = {offset : 0, nameLength : 0, readLength : 0};  // 一条数据
  std::cout << "read:\t." << std::flush;  // 进度条
  while (fastaFile.peek() != EOF && reads.size() < 0x7FFFFFFF) {  // 读完文件
    read.offset = fastaFile.tellg();  // fasta起始位置
    getline(fastaFile, line);  // 读序列名
    read.nameLength = line.size();
    read.readLength = 0;  // 序列数据长度清零
    while (fastaFile.peek() != EOF && fastaFile.peek() != '>') {  // 读序列
      getline(fastaFile, line);
      read.readLength += line.size();
    }
    if (read.readLength <= 0xFFFF) {  // 不超65535 入队 打印进度
      reads.push_back(read);  // 入队
      if (reads.size() % (1024 * 1024) == 0) std::cout << "." << std::flush;
    }
  }
  fastaFile.close();
  std::cout << " finish\n";
  reads.shrink_to_fit(); // 省点内存
  std::stable_sort(reads.begin(), reads.end(), [](const Read &a, const Read &b)
    { return a.readLength > b.readLength; }); // 排序
  std::cout << "count:\t" << reads.size() << "\n";
  std::cout << "long:\t" << reads.front().readLength << "\n";
  std::cout << "short:\t" << reads.back().readLength << "\n";
}

std::unordered_map<char, uint32_t> transTableGen = {  // 基因转码表
  {'a', 1}, {'c', 2}, {'g', 3}, {'t', 4}, {'u', 4},
  {'A', 1}, {'C', 2}, {'G', 3}, {'T', 4}, {'U', 4}
};  // 只用于packData函数
std::unordered_map<char, uint32_t> transTablePro = {  // 蛋白转码表
  {'a', 1},  {'c', 2},  {'d', 3},  {'e', 4},  {'f', 5},  {'g', 6},  {'h', 7},
  {'A', 1},  {'C', 2},  {'D', 3},  {'E', 4},  {'F', 5},  {'G', 6},  {'H', 7},
  {'i', 8},  {'k', 9},  {'l', 10}, {'m', 11}, {'n', 12}, {'o', 13}, {'p', 14},
  {'I', 8},  {'K', 9},  {'L', 10}, {'M', 11}, {'N', 12}, {'O', 13}, {'P', 14},
  {'q', 15}, {'r', 16}, {'s', 17}, {'t', 18}, {'u', 19}, {'v', 20}, {'w', 21},
  {'Q', 15}, {'R', 16}, {'S', 17}, {'T', 18}, {'U', 19}, {'V', 20}, {'W', 21},
  {'y', 22}, {'Y', 22}
};  // 只用于packData函数
//  makeData 生成数据 acgt -> 1010 1100
template <uint32_t entropy>
inline void packing(const std::string &read, uint32_t *packed) {
  uint32_t packLength = (read.size() + 31) / 32 * entropy + 2;  // 打包后长度
  std::memset(packed, 0, sizeof(uint32_t) * packLength);  // 初始化
  uint32_t packs[entropy] = {0};  // 打包后数据 编译会展开成寄存器
  uint32_t netLength = 0;  // 净长度
  std::unordered_map<char, uint32_t> *tables[6] =
    {NULL, NULL, NULL, &transTableGen, NULL, &transTablePro};
  std::unordered_map<char, uint32_t> *transTable = tables[entropy];  // 转码表
  for (char base : read) {  // 打包序列
    auto iterator = (*transTable).find(base);  // 查找
    if (iterator != (*transTable).end()) {  // 找到记录
      uint32_t pack = iterator->second;  // 编码
      for (uint32_t e = 0; e < entropy; e++) {  // 打包数据
        packs[e] = ((pack >> e & 1) << 31) + (packs[e] >> 1);
      }
      netLength += 1;
      if (netLength % 32 == 0) {  // 每32个氨基酸存储一次
        for (uint32_t e = 0; e < entropy; e++) {
          packed[2 + (netLength / 32 - 1) * entropy + e] = packs[e];
        }
      }
    }
  }
  if (netLength % 32 > 0) { // 补齐
    for (uint32_t e = 0; e < entropy; e++) {
      packed[2+netLength/32*entropy+e] = packs[e] >> 32 - (netLength % 32);
    }
  }
  packed[0] = read.size(); // 长度
  packed[1] = netLength;   // 净长度
}

std::unordered_map<char, uint32_t> kmerTableGen = {  // 基因转码表
  {'a', 0}, {'c', 1}, {'g', 2}, {'t', 3}, {'u', 3},
  {'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}, {'U', 3}
};  // 只用于hashData函数
std::unordered_map<char, uint32_t> kmerTablePro = {  // 蛋白转码表
  {'a', 0}, {'c', 1}, {'d', 2}, {'e', 3}, {'f', 4},  {'g', 5},  {'h', 6},
  {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3}, {'F', 4},  {'G', 5},  {'H', 6},
  {'i', 7}, {'k', 8}, {'l', 9}, {'m', 9}, {'n', 2},  {'o', 10}, {'p', 11},
  {'I', 7}, {'K', 8}, {'L', 9}, {'M', 9}, {'N', 2},  {'O', 10}, {'P', 11},
  {'q', 3}, {'r', 8}, {'s', 0}, {'t', 0}, {'u', 12}, {'v', 7},  {'w', 13},
  {'Q', 3}, {'R', 8}, {'S', 0}, {'T', 0}, {'U', 12}, {'V', 7},  {'W', 13},
  {'y', 4}, {'Y', 4}
};  // 只用于hashData函数
const uint32_t aArray[128] = {  // 类线性同余 kmer = (a*index+a)%n, an互质
  1,          9699691,    19399381,   29099071,   38798761,   48498451,
  58198141,   67897831,   77597521,   87297211,   96996901,   106696591,
  116396281,  126095971,  135795661,  145495351,  155195041,  164894731,
  174594421,  184294111,  193993801,  203693491,  213393181,  223092871,
  232792561,  242492251,  252191941,  261891631,  271591321,  281291011,
  290990701,  300690391,  310390081,  320089771,  329789461,  339489151,
  349188841,  358888531,  368588221,  378287911,  387987601,  397687291,
  407386981,  417086671,  426786361,  436486051,  446185741,  455885431,
  465585121,  475284811,  484984501,  494684191,  504383881,  514083571,
  523783261,  533482951,  543182641,  552882331,  562582021,  572281711,
  581981401,  591681091,  601380781,  611080471,  620780161,  630479851,
  640179541,  649879231,  659578921,  669278611,  678978301,  688677991,
  698377681,  708077371,  717777061,  727476751,  737176441,  746876131,
  756575821,  766275511,  775975201,  785674891,  795374581,  805074271,
  814773961,  824473651,  834173341,  843873031,  853572721,  863272411,
  872972101,  882671791,  892371481,  902071171,  911770861,  921470551,
  931170241,  940869931,  950569621,  960269311,  969969001,  979668691,
  989368381,  999068071,  1008767761, 1018467451, 1028167141, 1037866831,
  1047566521, 1057266211, 1066965901, 1076665591, 1086365281, 1096064971,
  1105764661, 1115464351, 1125164041, 1134863731, 1144563421, 1154263111,
  1163962801, 1173662491, 1183362181, 1193061871, 1202761561, 1212461251,
  1222160941, 1231860631};
const uint32_t afArray[128] = {  // 求逆 index = af*(kmer-a)%n, (af*a)%n=1
  1,          2444632899, 2858362493, 846064575,  1485729433, 2671958811,
  1902620885, 3849255383, 3339722161, 3940178035, 881810861,  2015768431,
  3943149897, 728432459,  3725791493, 3157110919, 1145654625, 3918508963,
  1618161373, 2520654111, 4050556409, 2317605243, 2856819509, 1617077559,
  2710220561, 1620530387, 777239053,  887939279,  1528708265, 721586603,
  3789188965, 1143535591, 978019009,  3657063427, 322051901,  3604057727,
  3360277849, 103034843,  4204309909, 524535959,  236619889,  4119456051,
  1732738669, 2071153199, 75255817,   2460871691, 1791456709, 1235105607,
  2199863329, 928980067,  1686435741, 4055970783, 1360592057, 2907495995,
  1336141813, 1170957303, 2023270865, 1180374419, 3037965005, 3667028879,
  2145053545, 1857431147, 602088485,  2072407719, 2841503105, 2594632899,
  2446637053, 1529221439, 3713350681, 4062635163, 2251442773, 573732695,
  1583223601, 2620643827, 1893809965, 2141406447, 3950186185, 199506123,
  263617157,  1532665351, 1897943777, 825943331,  1141530717, 1301882527,
  3958866809, 4022027003, 982308021,  2100429495, 2655241361, 3166516819,
  1197030285, 2256459343, 989706793,  3347129131, 1481275621, 2366883175,
  671057985,  2365844867, 1603203261, 439579647,  2037961433, 965933403,
  2558297877, 3926066711, 3366108657, 3122670259, 145006573,  1233002415,
  2276715913, 3746948491, 522723141,  657122503,  4111762849, 1953172963,
  3561711901, 2022570335, 1339173433, 2880899003, 726294901,  375292279,
  1867775825, 3271931667, 2020371533, 2977571087, 1615767785, 1101758443,
  1297640869, 43580455};
template <uint32_t entropy>  // 2:基因 4:蛋白
inline void hashing(const std::string &read, const uint32_t mask,
uint32_t *indexs, uint32_t *hashLine) {
  std::memset(indexs, 0xFF, sizeof(uint32_t) * SIGNEDCOUNT);  // 最后一个序列
  std::unordered_map<char, uint32_t> *tables[5] =
    {NULL, NULL, &kmerTableGen, NULL, &kmerTablePro};
  std::unordered_map<char, uint32_t> *kmerTable = tables[entropy];  // 转码表
  uint32_t kmer = 0;  // 生成的k-mer
  for (uint32_t i = 0; i < read.size(); i++) {  // 遍历read
    auto iterator = (*kmerTable).find(read[i]);  // 查找结果
    if (iterator != (*kmerTable).end()) {  // 找到了 碱基/氨基酸
      kmer = ((kmer << entropy) + iterator->second) & mask;  // 生成K-mer
      for (uint32_t j = 0; j < SIGNEDCOUNT; j++) {  // 查找最早kmer
        const uint32_t a = aArray[j];
        const uint32_t af = afArray[j];
        const uint32_t index = af * (kmer - a) & mask;
        hashLine[j] = index < indexs[j] ? kmer : hashLine[j]; // 更新kmer
        indexs[j] = index < indexs[j] ? index : indexs[j];    // 更新index
      }
    }  // 测试发现 不跳过前n个碱基效果更好
  }
}

// packData 打包数据
void packData(const Option &option, std::vector<Read> &reads,
std::vector<uint32_t> &hashTable) {
  const uint32_t entropy = option.entropy;  // 熵
  const uint32_t readsCount = reads.size();  // 序列数
  std::vector<uint32_t> nameLengths(readsCount, 0);  // 序列名长度
  std::vector<uint32_t> readLengths(readsCount, 0);  // 序列数据长度
  std::vector<size_t> packedOffsets(readsCount, 0);  // packed偏移
  std::vector<size_t> fastaOffsets(readsCount, 0);  // fasta偏移
  std::vector<size_t> inputOffsets(readsCount, 0);  // 输入文件偏移
  {  // 计算偏移 写入 熵 hashTable偏移 packed偏移 fasta偏移
    for (uint32_t i = 0; i < readsCount; i++) {
      nameLengths[i] = reads[i].nameLength;  // 序列名长度
      readLengths[i] = reads[i].readLength;  // 序列数据长度
      inputOffsets[i] = reads[i].offset;  // 输入文件的偏移
    }
    size_t offset = sizeof(uint32_t) * (3 + readsCount * 2);  // 当前指针
    offset += sizeof(size_t) * readsCount * 2 +
      sizeof(uint32_t) * readsCount * SIGNEDCOUNT;  // packed起始位置
    for (uint32_t i = 0; i < readsCount; i++) {  // packed偏移
      packedOffsets[i] = offset;
      offset += sizeof(uint32_t) * (2 + (readLengths[i] + 31) / 32 * entropy);
    }
    for (uint32_t i = 0; i < readsCount; i++) {  // fasta偏移
      fastaOffsets[i] = offset;
      offset += nameLengths[i] + readLengths[i] + 2;  // 包括换行
    }
    std::ofstream packedFile(option.packedFile);  // 输出文件 写入结果
    offset = fastaOffsets.back() + nameLengths.back() + readLengths.back() + 1;
    packedFile.seekp(offset, packedFile.beg);
    std::string line = "\n";
    packedFile.write((char *)line.data(), line.size());  // 先扩充文件到指定大小
    packedFile.seekp(0, packedFile.beg);  // 指针返回
    packedFile.write((char *)&entropy, sizeof(uint32_t));
    packedFile.write((char *)&readsCount, sizeof(uint32_t));
    uint32_t signedCount = SIGNEDCOUNT;  // 为了写入文件
    packedFile.write((char *)&signedCount, sizeof(uint32_t));
    packedFile.write((char *)nameLengths.data(), sizeof(uint32_t) * readsCount);
    packedFile.write((char *)readLengths.data(), sizeof(uint32_t) * readsCount);
    packedFile.write((char *)packedOffsets.data(), sizeof(size_t) * readsCount);
    packedFile.write((char *)fastaOffsets.data(), sizeof(size_t) * readsCount);
    packedFile.close();
    // 省点内存
    reads.resize(0);
    reads.shrink_to_fit();
    nameLengths.resize(0);
    nameLengths.shrink_to_fit();
    readLengths.resize(0);
    readLengths.shrink_to_fit();
  }
  hashTable.assign((size_t)readsCount * SIGNEDCOUNT, 0);  // 申请空间
  #pragma omp parallel num_threads(16)  // 16线程足够
  { // 打包数据 拷贝数据 多线程
    std::ifstream inputFile(option.fastaFile);  // 输入
    std::ofstream packedFile(option.packedFile, std::ios::in);  // 输出packed
    std::ofstream fastaFile(option.packedFile, std::ios::in);  // 输出fasta
    std::string line = "", name = "", read = "";  // 读入一行 序列名 序列数据
    uint32_t packed[65536 / 32 * 5 + 2] = {0};  // 压缩数据
    uint32_t indexs[SIGNEDCOUNT] = {0};  // 哈希签名的序号
    uint32_t mask = 0xFFFFFF;  // 短词掩码，太长稀疏遇不上，太短区分性不够
    if (readsCount > 0xFFFFFF) mask = 0xFFFFFFF;
    if (readsCount > 0xFFFFFFF) mask = 0xFFFFFFFF;
    #pragma omp master
    { std::cout << "pack:\t." << std::flush; }  // 打印进度
    #pragma omp for schedule(static, 16) // 并行任务 写packed数据 生成签名
    for (uint32_t i = 0; i < readsCount; i++) {  // 遍历序列
      inputFile.seekg(inputOffsets[i], inputFile.beg);  // 移到输入文件起始
      getline(inputFile, name);
      read.clear();  // 序列数据
      while (inputFile.peek() != EOF && inputFile.peek() != '>') {  // 读序列数据
        getline(inputFile, line);
        read += line;
      }
      if (entropy == 3) {  // 打包+签名 gene
        packing<3>(read, packed);
        hashing<2>(read, mask, indexs, &hashTable[(size_t)i * SIGNEDCOUNT]);
      }
      if (entropy == 5) {  // 打包+签名 protein
        packing<5>(read, packed);
        hashing<4>(read, mask, indexs, &hashTable[(size_t)i * SIGNEDCOUNT]);
      }
      packedFile.seekp(packedOffsets[i], packedFile.beg);  // 移到packed文件起始
      uint32_t length = (packed[0] + 31) / 32 * entropy + 2;  // 要拷贝长度
      packedFile.write((char *)packed, sizeof(uint32_t) * length);  // 拷贝
      line = name + "\n" + read + "\n";  // 序列内容
      fastaFile.seekp(fastaOffsets[i], fastaFile.beg);  // 移到fasta文件起始
      fastaFile.write((char *)line.c_str(), line.size());  // 写序列
      if ((i + 1) % (1024 * 1024) == 0)  std::cout << "." << std::flush;
    }
    #pragma omp master
    { std::cout << " finish\n"; }
    inputFile.close();
    packedFile.close();
    fastaFile.close();
  }
}

// groupSequence 序列分组
void groupSequence(const Option &option,
const std::vector<uint32_t> &hashTable) {
  const uint32_t readsCount = hashTable.size() / SIGNEDCOUNT;  // 序列数
  std::vector<uint32_t> groupFinal((size_t)readsCount * SIGNEDCOUNT, 0); // 分组
  std::cout << "group:\t." << std::flush;  // 打印进度
  #pragma omp parallel num_threads(SIGNEDCOUNT)
  {  // 分组 每个线程负责一组
    const uint32_t order = omp_get_thread_num();  // 给线程分配任务
    std::unordered_map<uint32_t, std::vector<uint32_t>> groups(0);  // 分组
    for (uint32_t i = 0; i < readsCount; i++) {  // 遍历所有序列的签名
      const uint32_t kmer = hashTable[(size_t)i * SIGNEDCOUNT + order];  // kemr
      const auto &iterator = groups.find(kmer);  // 查找k-mer
      if (iterator == groups.end()) {  // 没找到签名就添加记录
        groups[kmer] = {i + 0x80000000};
      } else {  // 找到了就添加序列
        iterator->second.push_back(i);
      }
      #pragma omp master
      if ((i + 1) % (1024 * 1024) == 0) {  // 打印进度
        std::cout << "." << std::flush;
      }
    }
    uint32_t *groupResult = groupFinal.data() + (size_t)order * readsCount;
    for (const auto &[key, value] : groups) {  // 拷贝结果
      std::memcpy(groupResult, value.data(), sizeof(uint32_t) * value.size());
      groupResult += value.size();
    }
  }
  std::cout << " finish\n";
  std::ofstream groupFile(option.packedFile, std::ios::in);  // 输出hashTable
  size_t groupOffset = sizeof(uint32_t) * (3 + readsCount * 2) +
    sizeof(size_t) * readsCount * 2; // hashTable偏移
  groupFile.seekp(groupOffset, groupFile.beg);
  groupFile.write((char *)groupFinal.data(),
    sizeof(uint32_t) * groupFinal.size());  // 写入结果
  groupFile.close();
}

//--------主函数--------//
int main(int argc, char **argv) {
  Timer::Timer timer;  // 开始计时
  Option option = {fastaFile : "", packedFile : "", entropy : 0};  // 选项
  init(argc, argv, option);  // 初始化
  std::vector<Read> reads(0);  // 序列集合
  makeIndex(option, reads);  // 生成文件索引
  std::vector<uint32_t> hashTable(0);  // 签名表
  packData(option, reads, hashTable);  // 打包数据
  groupSequence(option, hashTable);  // 序列分组
  timer.getDuration();  // 结束计时
  timer.getTimeNow();  // 时间戳
}
