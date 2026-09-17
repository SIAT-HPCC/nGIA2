#include <iostream>  // cout
#include <cmath>  // fabs
#include "timer.hpp"  // Timer
#include "parser.hpp"  // Parser
#include "makedb.hpp"  // makeDB
#include "buildGraph.hpp"  // buildGraph
#include "fastClustering.hpp"  // FastClustering (快速聚类)
#include "leidenParallel.hpp"  // LeidenOmp (并行Leiden聚类)
#include "saveResult.hpp"  // saveResult

class AutoTimer {  // 函数return时，自动打印时间
public:
  AutoTimer() {
    timer.start();
    timer.printStamp();
    std::cout << std::endl;
  }
  ~AutoTimer() {  // return时会析构，自动执行，提前return也会执行
    std::cout << std::endl;
    timer.printStamp();
    timer.printDuration();
  }
private:
  Timer::Timer timer;
};

int main(int argc, char **argv) {
  AutoTimer autoTimer;  // 自动打印时间戳
  
  // -------- 参数 -------- //
  std::string mode = "";  // 模式
  std::string fastaFile = "";  // fasta文件
  std::string packedFile = "";  // packed文件
  std::string resultFile = "";  // result文件
  uint32_t identity = 95;  // 相似度 默认95
  uint32_t bands = 64;  // 建图用的band数 默认64
  {  // 解析参数
    Parser::Parser parser;
    parser.add(true, "-m", "mode (string) makedb/clustering/fast", "");  // 模式
    parser.add(false, "-f", "fasta file (string)", "data.fasta");  // fasta
    parser.add(false, "-p", "packed file (string)", "packed.bin");  // packed
    parser.add(false, "-r", "result file (string)", "result.txt");  // result
    parser.add(false, "-i", "identity (1-99)", "95");  // 相似度 默认95
    parser.add(false, "-b", "bands (1-64)", "64");  // 建图band数 默认64
    if (!parser.parse(argc, argv)) { return 0; } // 报错就提前结束
    mode = parser.getValue<std::string>("-m");
    fastaFile = parser.getValue<std::string>("-f");
    packedFile = parser.getValue<std::string>("-p");
    resultFile = parser.getValue<std::string>("-r");
    identity = parser.getValue<uint32_t>("-i");
    bands = parser.getValue<uint32_t>("-b");
  }
  // -------- A. 生成数据库 -------- //
  if (mode == "makedb") {  // 生成数据库
    // 1. 自检
    if (!MakeDB::check(fastaFile, packedFile)) {  // 自检没过就输出用法
      std::cout << "\nUsage:\n";
      std::cout << argv[0] << " -m makedb -f fastaFile -p packedFile\n";
      return 0;
    }
    // 2. 构建索引
    std::vector<MakeDB::seqIndex> indices = MakeDB::makeIndex(fastaFile);
    size_t seqCount = indices.size();  // 可用序列数
    if (seqCount == 0) {  // 无有效序列 空文件或全部超长
      std::cout << "no valid sequence (empty file or all reads >= 65535)\n";
      return 1;
    }
    if (seqCount >= 0x7FFFFFFF) {  // 序列数须 < 2^31-1(与makedb.hpp头注释一致)
      std::cout << "too many sequences: " << seqCount
                << " (max " << (0x7FFFFFFF - 1) << ")\n";
      return 1;
    }
    // 3. 打包数据
    if (!MakeDB::packData(fastaFile, packedFile, indices)) {  // 打包失败
      std::cout << "packData failed, abort makedb.\n";
      return 1;
    }
    indices.clear(); indices.shrink_to_fit();  // 释放内存
    // 4. 分组：按签名排序索引，覆盖写回原签名区
    if (!MakeDB::groupSignatures(packedFile, seqCount)) {  // 分组失败
      std::cout << "groupSignatures failed, abort makedb.\n";
      return 1;
    }

  // -------- B. 精准的聚类 --------//
  } else if (mode == "clustering") {  // 聚类
    // 1. 自检
    if (!BuildGraph::check(packedFile, resultFile, identity, bands)) {
      std::cout << "\nUsage:\n";
      std::cout << argv[0] << " -m clustering -p packedFile -r resultFile";
      std::cout << " -i identity -b bands\n";
      return 0;
    }
    // 2. 读数据
    BuildGraph::Data inputData = BuildGraph::readData(packedFile, bands);
    if (inputData.seqCount == 0) {  // readData失败(分配错误等)
      std::cout << "readData failed, abort clustering.\n";
      return 1;
    }
    // 3. 建图
    std::cout << "Building graph..." << std::endl;
    bool graphOk = true;  // 建图是否成功
    std::vector<std::tuple<int, int, float>> edge_list =
      BuildGraph::buildGraph(inputData, identity, bands, graphOk);
    if (!graphOk) {  // 建图失败 提前结束
      std::cout << "buildGraph failed, abort clustering.\n";
      return 1;
    }
    BuildGraph::freeMemory(inputData);
    std::cout << "Build graph done." << std::endl;
    // 4. 聚类: C++并行Leiden (无权无向 分辨率γ=identity派生 1轮 seed42)
    std::vector<uint32_t> clusterResult;
    {
      float resolution = (100.0f - std::fabs((float)identity - 50.0f) * 2.0f) / 100.0f;  // 两端0.02、中间1
      if (resolution < 0.0f) resolution = 0.0f;  // 浮点误差钳制
      resolution = std::min(resolution, 1.0f);
      clusterResult = LeidenOmp::cluster(edge_list, (int)inputData.seqCount,
        resolution, 42, 1);  // 并行Leiden 一轮 seed42
      if (clusterResult.size() != inputData.seqCount) {  // 数量对不上
        std::cout << "cluster result size mismatch: "
                  << clusterResult.size() << " vs " << inputData.seqCount
                  << std::endl;
        return 1;
      }
    }
    // 5. 保存结果
    SaveResult::save(clusterResult, packedFile, resultFile);

    // -------- C. 快速的聚类 --------//
  } else if (mode == "fast") {  // 超快速聚类
    // 1. 自检
    if (!FastClustering::check(packedFile, resultFile, identity, bands)) {
      std::cout << "\nUsage:\n";
      std::cout << argv[0] << " -m fast -p packedFile -r resultFile";
      std::cout << " -i identity -b bands\n";
      return 0;
    }
    // 2. 读数据
    FastClustering::Data inputData = FastClustering::readData(packedFile, bands);
    if (inputData.seqCount == 0) {  // readData失败(分配错误等)
      std::cout << "readData failed, abort fast.\n";
      return 1;
    }
    // 3. 快速聚类: 逐band贪心并组
    std::cout << "Fast clustering..." << std::endl;
    bool fastOk = true;  // 聚类是否成功
    std::vector<uint32_t> clusterResult =
      FastClustering::cluster(inputData, identity, bands, fastOk);
    if (!fastOk) {  // 聚类失败 提前结束
      std::cout << "fast clustering failed, abort.\n";
      return 1;
    }
    FastClustering::freeMemory(inputData);
    std::cout << "Fast clustering done." << std::endl;
    // 4. 保存结果
    SaveResult::save(clusterResult, packedFile, resultFile);

    // -------- D. 错误提示 --------//
  } else {  // 模式错误
    std::cout << "Usage:\n" << argv[0] << " -m makedb/clustering/fast\n";
  }

  return 0;
}
