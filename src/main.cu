#include <iostream>  // cout
#include "makeDB.hpp"  // makeDB
#include "clustering.hpp"  // clustering

int main(int argc, char **argv) {
  if (argc>1 && std::string(argv[1]) == "makeDB") {  // 生成数据库
    MakeDB::makeDB(argc-1, argv+1);
  } else if (argc>1 && std::string(argv[1]) == "clustering") {  // 聚类
    Clustering::clustering(argc-1, argv+1);
  } else {  // 打印用法
    std::cout << "Usage: \n" << argv[0] << " makeDB/clustering\n";
  }
  return 0;
}
