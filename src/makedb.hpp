/*
makedb.hpp
输出文件数据内容:
  uint32_t            seqCount     1
  uint32_t[seqCount]  nameLengths  序列名字长度
  uint32_t[seqCount]  readLengths  序列数据长度
  vector<size_t>      packed数据偏移  seqCount
  vector<size_t>      fasta数据偏移   seqCount
  签名/分组数据 (GROUP_BANDS*readsCount条记录 每条uint32_t=4字节)
    packData写入: 每序列GROUP_BANDS个minhash签名
      布局 seq-major: 序列i的签名在 signBase+i*GROUP_BANDS
    groupSignatures覆盖写: 每band一行代表索引 rep[band][i]
      布局 band-major: band b的行在 signBase+b*seqCount
      (同签名桶内 rep[i]=索引更小的最近序列, 组头=自己)
  packed数据 (readsCount条记录)
    uint32_t 数据长度 1
    uint32_t 压缩数据 (数据长度+31)/32*4
  fasta数据 (readsCount条记录)
    string 序列名含换行
    string 序列含换行
长度 < 65536
用法:
ngia3 -m makedb -f fasta文件 -p packed文件
2026-07-09 by 鞠震
*/

// 序列长 < 0xFFFF     比对时line  < 2048
// 序列数 < 0x7FFFFFFF 比对时线程数 < int范围

#ifndef MAKEDB_HPP
#define MAKEDB_HPP

#include <iostream>  // cout
#include <fstream>   // ifstream/fstream
#include <vector>    // vector
#include <string>    // string
#include <algorithm> // stable_sort
#include <array>     // array
#include <cstdint>   // uint32_t
#include <utility>   // pair
#include <omp.h>     // openmp

namespace MakeDB {  // 命名空间

#define GROUP_BANDS 64  // 签名/分组次数

//--------------------------------  数据结构  --------------------------------//
struct seqIndex {  // 序列索引
  size_t fastaOffset;  // fasta文件偏移
  size_t packedOffsetPack;  // packed文件中 打包偏移
  size_t packedOffsetSeq;  // packed文件中 序列偏移
  uint32_t nameLength;  // 序列名字长度
  uint32_t readLength;  // 序列数据长度
  uint32_t seqLength;  // 序列记录长度 名字+数据+换行
};

//--------------------------------  功能函数  --------------------------------//
// check 自检
bool check(const std::string& fastaFile, const std::string& packedFile) {
  bool checkPassed = true;  // 校验是否通过
  {  // 校验参数
    // 1. 文件存在
    std::ifstream file(fastaFile);  // fasta文件
    if (!file.good()) {  // 文件不存在
      std::cout << fastaFile << " does not exist.\n";
      checkPassed = false;
    }
    // 2. 行尾没有\r，只\n换行
    std::string line;  // 一行数据
    getline(file, line);
    if (!line.empty() && line.back() == '\r') {  // 行尾有\r
      std::cout << "Please remove the \\r at the end of lines.\n";
      checkPassed = false;
    }
    // 3. 文件以换行结尾
    file.seekg(0, std::ios::end);
    if (file.tellg() > 0) {
      file.seekg(-1, std::ios::end);
      char lastChar;
      file.read(&lastChar, 1);
      if (lastChar != '\n') {
        std::cout << "File must end with \\n.\n";
        checkPassed = false;
      }
    }
  }
  std::cout << "fasta:\t" << fastaFile << "\n";  // 输入数据
  std::cout << "packed:\t" << packedFile << "\n";  // 打包输出
  std::cout << "thread:\t" << omp_get_max_threads() << "\n";  // 可用线程数
  return checkPassed;
}

// makeIndex 第一遍读文件，构建索引
std::vector<seqIndex> makeIndex(const std::string& fastaFile) {
  std::vector<seqIndex> indices;  // 序列索引
  size_t wholeCount = 0;  // 扫描的序列数（含超长丢弃的）
  size_t seqCount = 0;  // 保留的序列数
  {  // 1. 从文件读序列索引
    std::cout << "read:\t." << std::flush;  // 进度条
    std::ifstream fileIn;  // fasta文件
    char ioBuf[1 << 20];  // 1MB读缓冲 减少read系统调用(须在open前设置)
    fileIn.rdbuf()->pubsetbuf(ioBuf, sizeof(ioBuf));  // 设置缓冲
    fileIn.open(fastaFile);  // 打开文件
    std::string line = "";  // 一行数据
    seqIndex index = {};  // 序列索引 初始化为0
    while (fileIn.peek() != EOF) {  // 第一遍读文件
      index.fastaOffset = fileIn.tellg();  // fasta偏移
      getline(fileIn, line);  // 读 >name 行
      index.nameLength = line.size();  // 序列名字长度
      index.seqLength = line.size() + 1;  // 包括换行符
      index.readLength = 0;  // 序列数据长度
      while (fileIn.peek() != EOF && fileIn.peek() != '>') {
        getline(fileIn, line);  // 序列数据
        index.readLength += line.size();  // 序列数据长度
        index.seqLength += line.size() + 1;  // 包括换行符
      }
      wholeCount += 1;  // 文件中的序列总数
      if (index.readLength < 0xFFFF) {  // 只保留长度小于65536的序列
        indices.push_back(index);  // 直接push整个结构体
      }
      if (wholeCount % (1024 * 1024) == 0) { std::cout << "." << std::flush; }
    }
    indices.shrink_to_fit();  // 紧凑化
    seqCount = indices.size();  // 保留的序列数
    std::cout << " finish\n";  // 进度条结束
    fileIn.close();
  }
  {  // 2. 从索引计算packed偏移
    std::cout << "sort:\t" << std::flush;
    std::stable_sort(indices.begin(), indices.end(),
      [](const seqIndex& a, const seqIndex& b) {
        return a.readLength > b.readLength;
      }
    );  // 长度递减 稳定排序
    std::cout << "finish\n";
    std::cout << "index:\t" << std::flush;
    size_t baseOffset = sizeof(uint32_t);  // 基础偏移(seqCount为4字节)
    baseOffset += seqCount * (sizeof(uint32_t) + sizeof(size_t)) * 2;  // 偏移
    baseOffset += seqCount * sizeof(uint32_t) * GROUP_BANDS;  // 分组区 4字节/条
    for (size_t i = 0; i < seqCount; i++) {
      indices[i].packedOffsetPack = baseOffset;
      uint32_t byteCount = 1 + (indices[i].readLength + 31) / 32 * 4;
      baseOffset += byteCount * sizeof(uint32_t);
    }
    for (size_t i = 0; i < seqCount; i++) {
      indices[i].packedOffsetSeq = baseOffset;
      baseOffset += indices[i].nameLength + indices[i].readLength + 2;
    }
    std::cout << "finish\n";
  }
  {  // 3. 输出信息
    std::cout << "total:\t" << wholeCount << "\n";
    std::cout << "valid:\t" << seqCount << "\n";
  }
  return indices;
}

// 按照indices的结果，多线程的写入packed文件，签名直接写入文件
bool packData(const std::string& fastaFile, const std::string& packedFile,
const std::vector<seqIndex>& indices) {  // 返回是否成功
  // 精简15转码表 http://bioinfor.imu.edu.cn/raacbook/public/info.html?id=49
  static const std::array<uint32_t, 256> transTable = []() {
    std::array<uint32_t, 256> t = {};
    t['a']= 1; t['c']= 2; t['d']= 3; t['e']= 4; t['f']= 5; t['g']= 6; t['h']= 7;
    t['A']= 1; t['C']= 2; t['D']= 3; t['E']= 4; t['F']= 5; t['G']= 6; t['H']= 7;
    t['i']= 8; t['k']= 9; t['l']=10; t['m']=10; t['n']=11; t['p']=12; t['q']= 4;
    t['I']= 8; t['K']= 9; t['L']=10; t['M']=10; t['N']=11; t['P']=12; t['Q']= 4;
    t['r']=13; t['s']=14; t['t']=15; t['u']=15; t['v']= 8; t['w']= 5; t['y']= 5;
    t['R']=13; t['S']=14; t['T']=15; t['U']=15; t['V']= 8; t['W']= 5; t['Y']= 5;
    return t;
  }();
  const size_t seqCount = indices.size();  // 序列数
  if (seqCount == 0) { return false; }  // 空数据 返回失败
  if (seqCount >= 0x7FFFFFFF) {  // 打包时严格限制 有效序列数 < int32最大正数
    std::cout << "too many sequences: " << seqCount
              << " (max " << (0x7FFFFFFF - 1) << ")\n";
    return false;  // 返回失败
  }
  {  // 1. 写入文件头部：序列数、name/read净长度、packed偏移表、fasta偏移表
    std::ofstream fileOut(packedFile, std::ios::binary);
    uint32_t seqCount32 = (uint32_t)seqCount;  // 序列数(4字节 已校验<2^31)
    fileOut.write((char*)&seqCount32, sizeof(uint32_t));  // 序列数(uint32_t)
    std::vector<uint32_t> nameLengths(seqCount);  // 序列名字净长度
    std::vector<uint32_t> readLengths(seqCount);  // 序列数据净长度
    std::vector<size_t> packOffsets(seqCount);  // 打包偏移
    std::vector<size_t> seqOffsets(seqCount);  // fasta偏移
    for (size_t i = 0; i < seqCount; i++) {
      nameLengths[i] = indices[i].nameLength;  // 包含开头的>
      readLengths[i] = indices[i].readLength;
      packOffsets[i] = indices[i].packedOffsetPack;  // 打包偏移
      seqOffsets[i] = indices[i].packedOffsetSeq;  // fasta偏移
    }
    fileOut.write((char*)nameLengths.data(), seqCount * sizeof(uint32_t));
    fileOut.write((char*)readLengths.data(), seqCount * sizeof(uint32_t));
    fileOut.write((char*)packOffsets.data(), seqCount * sizeof(size_t));
    fileOut.write((char*)seqOffsets.data(), seqCount * sizeof(size_t));
  }
  {  // 2. 并发打包和签名，结果直接写入文件
    std::cout << "pack:\t" << std::flush;
    const size_t signBase = sizeof(uint32_t) + seqCount * sizeof(size_t) * 2
      + seqCount * sizeof(uint32_t) * 2;  // 签名的起始偏移(seqCount为4字节)
    #pragma omp parallel
    {  // 每个线程有单独的私有变量
      std::ifstream fileIn(fastaFile);  // fasta文件
      std::fstream fileOut(packedFile,
        std::ios::binary | std::ios::in | std::ios::out);  // packed文件
      std::vector<uint32_t> packedBuffer(65536 / 32 * 4 + 1);  // 打包后序列
      std::vector<uint32_t> hashSigns(GROUP_BANDS);  // GROUP_BANDS个哈希签名
      std::string name = "", read = "", buffer = "";  // 名字、序列、缓冲
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < seqCount; i++) {  // 遍历序列
        {  // 2.1 从fasta文件读取一条完整记录并解析
          fileIn.seekg(indices[i].fastaOffset);  // 跳转到序列起始位置
          buffer.resize(indices[i].seqLength);  // 分配内存
          fileIn.read(&buffer[0], indices[i].seqLength);  // 读数据
          size_t pos = buffer.find('\n');  // 分割名字和序列
          name = buffer.substr(0, pos);  // 名字
          read = buffer.substr(pos + 1);  // 序列
          read.erase(std::remove(read.begin(), read.end(), '\n'), read.end());
        }
        {  // 2.2 打包 + 签名
          const uint32_t length = (uint32_t)read.size();  // 序列长度
          packedBuffer.assign(1 + (length + 31) / 32 * 4, 0);  // 分配内存
          hashSigns.assign(GROUP_BANDS, 0xFFFFFFFF);  // 重置签名
          uint32_t kmer = 0;  // 8-mer 滑动窗口
          packedBuffer[0] = length;
          for (uint32_t j = 0; j < length; j++) {
            uint32_t pack = transTable[(uint8_t)read[j]];  // 查表
            for (uint32_t e = 0; e < 4; e++) {  // 打包 4-bit
              packedBuffer[1 + j / 32 * 4 + e] += (pack >> e & 1) << (j % 32);
            }
            kmer <<= 4;  // 8-mer=32bit 左移自然溢出 丢弃最旧字母
            kmer += pack;
            // 前7位不签名; 长度<8的短序列在最后一位用不完整kmer签名(刻意,保证短读可分组)
            if (j < 7 && j < length - 1) { continue; }  // 前7个不签名
            for (uint32_t k = 0; k < GROUP_BANDS; k++) {  // GROUP_BANDS个签名
              uint32_t seed = k * 0x9e3779b1 + 0x85ebca6b;  // 黄金比例常数
              uint32_t sign = kmer ^ seed;  // wang's 哈希
              sign = (sign ^ 61) ^ (sign >> 16);
              sign = sign + (sign << 3);
              sign = sign ^ (sign >> 4);
              sign = sign * 0x27d4eb2d;
              sign = sign ^ (sign >> 15);
              sign = (sign ^ (sign >> 8)) * 0x9e3779b1;  // 额外的雪崩效应
              sign = sign ^ (sign >> 14);
              if (sign < hashSigns[k]) { hashSigns[k] = sign; }  // 更新签名
            }
          }
        }
        {  // 2.3 写入packed文件
          fileOut.seekp(signBase + i * GROUP_BANDS * sizeof(uint32_t));  // 写签名
          fileOut.write((char*)hashSigns.data(), GROUP_BANDS * sizeof(uint32_t));
          fileOut.seekp(indices[i].packedOffsetPack);  // 写打包
          uint32_t lengthTemp = packedBuffer.size() * sizeof(uint32_t);
          fileOut.write((char*)packedBuffer.data(), lengthTemp);
          fileOut.seekp(indices[i].packedOffsetSeq);  // 写fasta数据
          buffer.clear();
          buffer += name; buffer += "\n";
          buffer += read; buffer += "\n";
          fileOut.write(buffer.data(), buffer.size());
        }
        // 2.4 打印进度
        if (i % (1024 * 1024) == 0) {
          #pragma omp critical
          { std::cout << "." << std::flush; }
        }
      }
    }
    std::cout << " finish\n";  // 进度条结束
  }
  {  // 3. 校验文件大小 防止写失败(如磁盘满)导致的静默损坏
    std::ifstream verifyIn(packedFile, std::ios::binary);  // 校验
    verifyIn.seekg(0, std::ios::end);  // 跳到文件末尾
    size_t actual = (size_t)verifyIn.tellg();  // 实际文件大小
    const seqIndex& last = indices.back();  // 排序后最后一条记录
    size_t expected = last.packedOffsetSeq + last.nameLength
      + last.readLength + 2;  // 期望总大小 = 末条fasta起点+其记录字节数
    if (actual != expected) {  // 大小不符 = 写入不完整
      std::cout << "packData write incomplete: " << actual
                << " bytes, expected " << expected << " bytes\n";
      return false;  // 返回失败
    }
    std::cout << "file size:\t" << actual << " bytes\n";  // 输出文件大小
  }
  return true;
}

// 根据文件中的签名分组：每band内签名相同=同一组，记录组内最近序列索引，覆盖写回原签名区
bool groupSignatures(const std::string& packedFile, size_t seqCount) {
  if (seqCount == 0) { return true; }  // 空数据 直接返回
  std::cout << "group:\t" << std::flush;
  const size_t signBase = sizeof(uint32_t) + seqCount * sizeof(size_t) * 2
    + seqCount * sizeof(uint32_t) * 2;  // 签名的起始偏移(seqCount为4字节)
  std::vector<uint32_t> signs(GROUP_BANDS * seqCount);  // 读入的签名
  std::ifstream fileIn(packedFile, std::ios::binary);  // 输入文件
  fileIn.seekg(signBase);  // 跳到签名区
  fileIn.read((char*)signs.data(), signs.size() * sizeof(uint32_t));  // 读签名
  if (!fileIn) {  // 读取失败 文件损坏或IO错误
    std::cout << "groupSignatures: read sign area failed ("
              << packedFile << " truncated?)\n";
    return false;  // 返回失败
  }
  fileIn.close();  // 关闭输入
  bool ioFailed = false;  // 并行写失败标记
  // band只有GROUP_BANDS行 线程数超过它纯属浪费内存(每线程约20B×seqCount)
  const int nThreads = std::min(omp_get_max_threads(), (int)GROUP_BANDS);
  #pragma omp parallel num_threads(nThreads)  // 每个线程有单独的私有变量 处理一个band
  {
    std::vector<uint64_t> keyed(seqCount);  // 打包(签名<<32|序号) 基数排序源
    std::vector<uint64_t> tmp(seqCount);  // 基数排序乒乓缓冲
    std::vector<uint32_t> row(seqCount);  // 线程行缓冲 算完一个band写回一次
    std::fstream fileOut(packedFile,
      std::ios::binary | std::ios::in | std::ios::out);  // packed文件
    #pragma omp for schedule(dynamic, 1)
    for (int band = 0; band < GROUP_BANDS; band++) {
      bool fail = false;  // 写失败标记快照 前面已有band写失败则跳过
      #pragma omp atomic read
      fail = ioFailed;
      if (fail) { continue; }  // 已失败 跳过后续band
      for (uint32_t i = 0; i < seqCount; i++) {  // 打包键(签名<<32|序号 输入序号升序)
        uint32_t key = signs[(size_t)i * GROUP_BANDS + band];  // r=1 键=签名
        keyed[i] = ((uint64_t)key << 32) | i;
      }
      // 稳定LSD基数排序(按key 4字节): 输出与"按(签名,索引)比较排序"逐位一致
      uint64_t* src = keyed.data();  // 当前源
      uint64_t* dst = tmp.data();  // 当前目标
      for (int pass = 0; pass < 4; pass++) {  // 32位key 共4趟(8bit/趟)
        uint32_t cnt[256] = {0};  // 桶计数
        const int shift = 32 + pass * 8;  // 仅排key的4字节
        for (uint32_t i = 0; i < seqCount; i++) {  // 计数
          cnt[(src[i] >> shift) & 0xFF]++;
        }
        uint32_t sum = 0;  // 前缀和->各桶起点
        for (int b = 0; b < 256; b++) { uint32_t c = cnt[b]; cnt[b] = sum; sum += c; }
        for (uint32_t i = 0; i < seqCount; i++) {  // 稳定散布
          dst[cnt[(src[i] >> shift) & 0xFF]++] = src[i];
        }
        std::swap(src, dst);  // 乒乓(4趟后结果回到keyed)
      }
      row[(uint32_t)src[0]] = (uint32_t)src[0];  // 第一条 总是自己
      uint32_t prevKey = (uint32_t)(src[0] >> 32);  // 上一条签名
      uint32_t prevIdx = (uint32_t)src[0];  // 上一条序号
      for (uint32_t y = 1; y < seqCount; y++) {  // 遍历序列
        uint32_t key = (uint32_t)(src[y] >> 32);  // 本条签名
        uint32_t idx = (uint32_t)src[y];  // 本条序号
        row[idx] = (key == prevKey) ? prevIdx : idx;  // 相同签名=前一个
        prevKey = key; prevIdx = idx;
      }
      fileOut.seekp(signBase + (size_t)band * seqCount * sizeof(uint32_t));  // 定位
      fileOut.write((char*)row.data(), seqCount * sizeof(uint32_t));  // band间不相交
      fileOut.flush();  // 立即落盘 使写错误在此暴露(否则缓冲尾部错误丢在析构里)
      if (!fileOut) {  // 写失败(如磁盘满)
        #pragma omp atomic write
        ioFailed = true;
      }
    }
  }
  if (ioFailed) {  // 有band写失败
    std::cout << "groupSignatures write failed (disk full?)\n";
    return false;  // 返回失败
  }
  std::cout << " finish\n";  // 进度条结束
  return true;
}

}  // namespace MakeDB
#endif  // MAKEDB_HPP
