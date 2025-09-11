// makedbUnitTest.cpp
// 这是makedb功能的单元测试代码
// 如果修改了makedb的功能，可以用这个代码校验结果
// 不要优化这个代码，保持简单，保持笨拙，保！持！正！确！
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "timer.hpp"

// **** 修改这里 ****//
#define FASTAFILE "../proteinTest.fasta"  // 输入文件
#define PACKEDFILE "../proteinTest.packed2"  // 输出文件
#define MASK 0xFFFFFF  // k-mer的掩码
// **** 修改这里 ****//

struct Sequence {  // 一条序列
  std::string name;  // 序列名
  std::string read;  // 序列内容
};

void readFile(std::vector<Sequence>& seqs) {  // 读序列
  std::ifstream fastaFile(FASTAFILE);
  std::string line = "";
  Sequence seq = {name:"", read:""};
  while (fastaFile.peek() != EOF) {
    getline(fastaFile, seq.name);
    seq.read = "";
    while (fastaFile.peek()!=EOF && fastaFile.peek()!='>') {
      getline(fastaFile, line);
      seq.read += line;
    }
    if (seq.read.size() < 0xFFFF) { seqs.push_back(seq); }  // 太长的不要
  }
  std::stable_sort(seqs.begin(), seqs.end(), [](const Sequence &a,
    const Sequence &b) { return a.read.size() > b.read.size(); });
  std::cout << "count\t" << seqs.size() << std::endl;
}

void writeIndex(const std::vector<Sequence>& seqs) {  // 写索引
  const uint32_t seqCount = seqs.size();  // 序列数
  std::vector<size_t> packed_offsets(seqCount);  // packed偏移
  std::vector<size_t> fasta_offsets(seqCount);  // fasta偏移
  {  // 计算偏移
    size_t offset = sizeof(uint32_t)+sizeof(size_t)*seqCount*2;  // packed偏移
    offset += sizeof(uint32_t)*seqCount*128;
    for (uint32_t i=0; i<seqCount; i++) {  // pakced偏移
      packed_offsets[i] = offset;
      offset += (1+(seqs[i].read.size()+31)/32*5)*sizeof(uint32_t);
    }
    for (uint32_t i=0; i<seqCount; i++) {  // fasta偏移
      fasta_offsets[i] = offset;
      offset += seqs[i].name.size()+seqs[i].read.size()+2;
    }
  }
  {  // 写入文件
    std::ofstream packedFile(PACKEDFILE);  // 输出packed
    packedFile.write((char*)&seqCount, sizeof(uint32_t));
    packedFile.write((char*)packed_offsets.data(), seqCount*sizeof(size_t));
    packedFile.write((char*)fasta_offsets.data(), seqCount*sizeof(size_t));
    packedFile.seekp(fasta_offsets[0], packedFile.beg);
    for (uint32_t i=0; i<seqCount; i++) {
      std::string line = seqs[i].name + "\n" + seqs[i].read + "\n";
      packedFile.write(line.c_str(), line.size());
    }
  }
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

void packSeq(const std::vector<Sequence> &seqs) {  // 打包序列
  std::vector<uint32_t> packedRead(1+65536/32*5, 0);
  std::ofstream packedFile(PACKEDFILE, std::ios::in);
  {  // 移动到数据位置
    size_t offset = sizeof(uint32_t)+sizeof(size_t)*seqs.size()*2;
    offset += sizeof(uint32_t)*seqs.size()*128;
    packedFile.seekp(offset, std::ios::beg);
  }
  std::cout << "packing\n";
  for (Sequence seq : seqs) {  // 打包数据
    uint32_t length = seq.read.size();
    uint32_t netLength = 0;
    packedRead.assign(1+(length+31)/32*5, 0);
    for (char c : seq.read) {
      uint32_t packed = transTable.at(c);
      for (uint32_t e=0; e<5; e++) {
        packedRead[1+netLength/32*5+e] |= ((packed>>e)&1)<<(netLength%32);
      }
      netLength++;
    }
    packedRead[0] = length;
    netLength = 1+(length+31)/32*5;
    packedFile.write((char*)packedRead.data(), netLength*sizeof(uint32_t));
  }
}

void groupSeq(const std::vector<Sequence> &seqs) {  // 分组
  const uint32_t seqCount = seqs.size();  // 序列数
  struct Record {  // 一条序列的签名记录
    uint32_t index;
    uint32_t signature;
  };
  std::vector<Record> records(seqCount*128, {index:0, signature:0});
  {  // 生成签名
    std::cout << "signing\t." << std::flush;
    for (uint32_t i=0; i<seqCount; i++) {
      for (uint32_t s=0; s<128; s++) {  // 赋初值
        records[s*seqCount+i].index = i;
        records[s*seqCount+i].signature = 0xFFFFFFFF;
      }
      uint32_t kmer = 0;
      for (char c : seqs[i].read) {
        kmer = ((kmer<<5)+transTable.at(c)) & MASK;
        uint64_t signature = kmer;
        for (uint32_t s=0; s<128; s++) {
          signature = signature*48271%0x7FFFFFFF;
          records[s*seqCount+i].signature =
            std::min((uint32_t)signature, records[s*seqCount+i].signature);
        }
      }
      if ((i+1)%(1024*128) == 0) { std::cout << "." << std::flush; }
    }
    std::cout << std::endl;
  }
  std::vector<uint32_t> groups(128*seqCount, 0);  // 分组
  {  // 序列分组
    std::cout << "group\t" << std::flush;
    for (uint32_t s=0; s<128; s++) {
      std::stable_sort(records.begin()+s*seqCount,
        records.begin()+(s+1)*seqCount, [](const Record &a, const Record &b)
        { return a.signature > b.signature; });  // 排序
      uint32_t index = records[s*seqCount].index;  // 第一个索引
      uint32_t haedSig = records[s*seqCount].signature;  // 第一个签名
      for (uint32_t i=0; i<seqCount; i++) {  // 写入代表序列
        if (records[s*seqCount+i].signature != haedSig) {
          index = records[s*seqCount+i].index;
        }
        uint32_t target = records[s*seqCount+i].index;  // 目标序列
        groups[s*seqCount+target] = index;
      }
    }
    std::cout << "finish\n";
  }
  {  // 写入数据
    std::cout << "writing\t" << std::flush;
    std::ofstream packedFile(PACKEDFILE, std::ios::in);
    size_t offset = sizeof(uint32_t)+sizeof(size_t)*seqCount*2;
    packedFile.seekp(offset, packedFile.beg);
    packedFile.write((char*)groups.data(), groups.size()*sizeof(uint32_t));
    packedFile.close();
    std::cout << "finish\n";
  }
}

int main() {
  std::vector<Sequence> seqs(0);  // 序列
  readFile(seqs);  // 读序列
  writeIndex(seqs);  // 写索引
  packSeq(seqs);  // 打包序列j
  groupSeq(seqs);  // 预分组序列
}
