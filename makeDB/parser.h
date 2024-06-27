/*
  parser.h
  命令行参数解析器 用法如下
  Parser::Parser parser;  // 声明
  void parser.add(参数名，短名，参数描述，参数类型，默认值，是否必要); -注册参数
  以上全是string类型
  参数类型："int32_t", "double", "string"
  必要参数未满足 打印帮助
  bool parser.parse(argc, argv); -解析参数
  int32_t parser.getInt32_t(参数名); -返回 int32_t
  double parser.getDouble(参数名); -返回 double
  std::string parser.getString(参数名); -返回 string
  函数接口没有错误处理，一定不能瞎用！
  函数接口没有错误处理，一定不能瞎用！
  函数接口没有错误处理，一定不能瞎用！
  2024/05/15 by 鞠震
*/

#include <iostream> // cout
#include <map>      // map
#ifndef __PARSERH__ // 防御声明 pragma once 通用性有问题
#define __PARSERH__
//--------类声明--------//
namespace Parser { // 命名空间
class Parser {     // 解析器
private:
  struct Record {             // 一条记录
    std::string shortName;    // 短名
    std::string describe;     // 参数描述
    std::string dataType;     // 参数类型
    std::string defaultValue; // 默认值
    bool isNecessary;         // 是否有必要
    bool isAssignment;        // 是否已赋值
  };
  std::map<std::string, Record> records; // 记录们
  void help(const std::string command) { // 打印帮助
    std::cout << "usage: " << command << " ";
    for (const auto &record : records) { // 遍历记录
      if (record.second.isNecessary) {   // 只打印必须参数
        std::cout << record.second.shortName << " <" << record.first << "> ";
      }
    }
    std::cout << " ...\noption:\n";
    for (const auto &record : records) { // 必要选项的详细信息
      if (!record.second.isNecessary)
        continue;
      std::cout << "  " << record.second.shortName;
      std::cout << "\t" << record.second.describe;
      std::cout << " (" << record.second.dataType << ")";
      std::cout << " *\n";
    }
    for (const auto &record : records) { // 非必要选项的详细信息
      if (record.second.isNecessary)
        continue;
      std::cout << "  " << record.second.shortName;
      std::cout << "\t" << record.second.describe;
      std::cout << " (" << record.second.dataType << ")";
      std::cout << "\n";
    }
    std::cout << "  * is necessary.\n";
  }

public:
  Parser() { records.clear(); }  // 构建
  ~Parser() { records.clear(); } // 析构

  void add(const std::string name, const std::string shortName,
           const std::string describe, const std::string dataType,
           const std::string defaultValue, const bool isNecessary) { // 注册参数
    Record record = {"", "", "", "", false, false};
    record.shortName = shortName;
    record.describe = describe;
    record.dataType = dataType;
    record.defaultValue = defaultValue;
    record.isNecessary = isNecessary;
    record.isAssignment = false;
    records[name] = record;
  }

  int32_t getInt32_t(const std::string name) { // 读取int参数
    return std::stoi(records[name].defaultValue);
  }
  float getFloat(const std::string name) { // 读取float参数
    return std::stof(records[name].defaultValue);
  }
  std::string getString(const std::string name) { // 读取string参数
    return records[name].defaultValue;
  }

  bool parse(int argc, char **argv) {    // 解析函数
    for (int32_t i = 1; i < argc; i++) { // 读参数
      for (auto &record : records) {
        if (record.second.shortName == argv[i] && i + 1 < argc) {
          record.second.defaultValue = argv[i + 1];
          record.second.isAssignment = true;
          i += 1;
        }
      }
    }
    bool checkSuccess = true;            // 校验是否成功
    for (const auto &record : records) { // 校验参数
      checkSuccess &=
          record.second.isNecessary ? record.second.isAssignment : true;
      if (record.second.dataType == "int32_t") { // int类型
        try {
          auto a = std::stoi(record.second.defaultValue);
        } catch (...) {
          checkSuccess = false;
        }
      } else if (record.second.dataType == "float") { // float类型
        try {
          auto a = std::stof(record.second.defaultValue);
        } catch (...) {
          checkSuccess = false;
        }
      }
    }
    if (!checkSuccess)
      help(argv[0]);
    return checkSuccess;
  }
};
} // namespace Parser
#endif // __PARSERH__
