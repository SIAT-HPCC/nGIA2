/*
leidenParallel.hpp
并行(OpenMP)模块度社区发现 —— 供聚类流水线替代Python Leiden。

设计目标:
  - 无权、无向图
  - 分辨率 γ 参数(对应python路径 resolution = (100-|identity-50|*2)/100)
  - RBConfiguration质量(配置模型+γ): 移动增益 ∝ w - γ·k·K/(Σdeg)
  - 局部移动采用确定性顺序扫掠+立即更新(教科书Louvain局部移动): 同步并行扫掠会因
    相邻节点互迁而震荡不收敛; 顺序版只接受严格正增益 -> 单调收敛、结果与线程数无关;
    该阶段单遍O(m)且通常几十遍内收敛, 开销小
  - OpenMP并行用于逐层聚合直方图等批量阶段; 精炼步=受约束随机顺序合并(顺序)
  - 不做逐位复刻: 与python leidenalg结果可能不同, 但质量语义一致

结构: 在收缩层级上循环: 局部移动(并行扫掠) -> 精炼(受约束随机合并) ->
      按精炼分区收缩成超图, 直到无任何合并。原始节点簇号 = 沿每层分区映射链追到顶层。

用法:
  auto clu = LeidenOmp::cluster(edges, nodeCount, resolution, seed, n_iterations);
  edges: {(u,v,w)} 权重忽略(无权), 无向
2026-09 by 鞠震
*/

#ifndef LEIDENPARALLEL_HPP
#define LEIDENPARALLEL_HPP

#include <cstdint>   // uint32_t
#include <vector>    // vector
#include <tuple>     // tuple
#include <utility>   // pair
#include <algorithm> // sort
#include <parallel/algorithm> // __gnu_parallel::sort (GCC libstdc++)
#include <random>    // mt19937_64
#include <numeric>   // accumulate

namespace LeidenOmp {

//--------------------------------  CSR 图(无向 支持自环/重边合并)  --------------------------------//
struct CSR {
  int n = 0;  // 节点数
  std::vector<int> row;    // CSR行起点 n+1
  std::vector<int> adj;    // 邻居
  std::vector<double> awt; // 邻接权重(自环在邻接中出现两次 每方向awt=权)
  std::vector<double> deg; // 度 = Σ行内awt(自环两遍计数)
  double totalDeg = 0.0;   // Σdeg(=2*总边权 无自环时)

  // 从"已规范化(u<=v)且已按(cu,cv)升序唯一"的边表直接建CSR(跳过排序/合并)
  static CSR buildFromSorted(int n_, const std::vector<int>& from,
                             const std::vector<int>& to,
                             const std::vector<double>& wt) {
    CSR g;
    g.n = n_;
    const int m = (int)from.size();
    g.row.assign(n_ + 1, 0);
    for (int e = 0; e < m; e++) { g.row[from[e] + 1]++; g.row[to[e] + 1]++; }
    for (int v = 0; v < n_; v++) g.row[v + 1] += g.row[v];
    g.adj.resize(2 * m); g.awt.resize(2 * m);
    g.deg.assign(n_, 0.0);
    std::vector<int> cur(g.row.begin(), g.row.end() - 1);
    for (int e = 0; e < m; e++) {
      int u = from[e], v = to[e]; double w = wt[e];
      g.adj[cur[u]] = v; g.awt[cur[u]] = w; cur[u]++;
      g.adj[cur[v]] = u; g.awt[cur[v]] = w; cur[v]++;
    }
    for (int v = 0; v < n_; v++) {
      double s = 0;
      for (int k = g.row[v]; k < g.row[v + 1]; k++) s += g.awt[k];
      g.deg[v] = s;
    }
    g.totalDeg = std::accumulate(g.deg.begin(), g.deg.end(), 0.0);
    return g;
  }

  static CSR build(int n_, const std::vector<std::pair<int,int>>& e0,
                   const std::vector<double>& w0) {
    // 规范化 u<=v 并合并重复边(权值为整数计数 求和精确 与排序顺序无关)
    std::vector<std::pair<std::pair<int,int>, double>> es;
    es.reserve(e0.size());
    for (size_t i = 0; i < e0.size(); i++) {
      if (w0[i] <= 0.0) continue;
      int u = std::min(e0[i].first, e0[i].second);
      int v = std::max(e0[i].first, e0[i].second);
      es.emplace_back(std::make_pair(u, v), w0[i]);
    }
    __gnu_parallel::sort(es.begin(), es.end());  // 并行排序(结果确定)
    std::vector<int> from, to;
    std::vector<double> wt;
    for (size_t i = 0; i < es.size();) {
      size_t j = i; double s = 0;
      while (j < es.size() && es[j].first == es[i].first) { s += es[j].second; j++; }
      from.push_back(es[i].first.first); to.push_back(es[i].first.second);
      wt.push_back(s); i = j;
    }
    return buildFromSorted(n_, from, to, wt);
  }
};

//--------------------------------  邻居社区权值收集(度数一般较小 线性搜索即可)  --------------------------------//
static void neighWeights(const CSR& g, const std::vector<int>& memb, int v,
    std::vector<int>& cids, std::vector<double>& wsum) {
  cids.clear(); wsum.clear();
  for (int k = g.row[v]; k < g.row[v + 1]; k++) {
    int u = g.adj[k];
    if (u == v) continue;  // 自环(社区内部边)与社区归属无关 不影响移动增益
    double w = g.awt[k];
    int c = memb[u];
    bool dup = false;
    for (size_t i = 0; i < cids.size(); i++) {
      if (cids[i] == c) { wsum[i] += w; dup = true; break; }
    }
    if (!dup) { cids.push_back(c); wsum.push_back(w); }
  }
}

//--------------------------------  局部移动: 顺序扫掠+立即更新(教科书Louvain局部移动)  --------------------------------//
// 同步并行扫掠会因相邻节点互迁而震荡(两个单点互相提议迁入对方->来回翻转)。
// 改为确定性顺序执行: 每次移动立即更新聚合值, 只接受严格正增益 ->
// 模块度单调上升必然收敛, 结果与线程数无关。每遍O(m), 通常<=几十遍, 开销小。
static int localMoving(const CSR& g, std::vector<int>& memb,
    std::vector<double>& tot, double gamma, int maxSweeps = 300) {
  const double M = g.totalDeg;
  if (M <= 0.0) return 0;
  const int n = g.n;
  int totalMoves = 0;
  std::vector<int> lc;
  std::vector<double> lw;
  for (int sweep = 0; sweep < maxSweeps; sweep++) {
    int moves = 0;
    for (int v = 0; v < n; v++) {
      if (g.row[v] == g.row[v + 1]) continue;  // 孤立点无邻居
      const int old = memb[v];
      const double k = g.deg[v];
      neighWeights(g, memb, v, lc, lw);
      double wOld = 0.0;
      for (size_t i = 0; i < lc.size(); i++) if (lc[i] == old) wOld = lw[i];
      int best = old;
      double bestGain = 0.0;
      for (size_t i = 0; i < lc.size(); i++) {
        int c = lc[i];
        if (c == old) continue;
        // 增益(把v并入c): (w_c - γk(K_c+k)/M) - (w_old - γk·K_old/M)
        double gain = (lw[i] - gamma * k * (tot[c] + k) / M)
                    - (wOld - gamma * k * tot[old] / M);
        if (gain > bestGain) { bestGain = gain; best = c; }
      }
      if (best != old && bestGain > 0.0) {  // 严格正增益才动
        memb[v] = best;
        tot[old] -= k;
        tot[best] += k;
        moves++;
      }
    }
    totalMoves += moves;
    if (moves == 0) break;
  }
  return totalMoves;
}

//--------------------------------  精炼: 受约束随机顺序合并(顺序执行)  --------------------------------//
// 初始每节点独立子社区; 随机顺序: 仅单点子社区可并入同粗社区邻居的子社区,
// 仅单点子社区参与合并, 可保持单点。
static void refine(const CSR& g, const std::vector<int>& coarse,
    std::vector<int>& smemb, std::vector<double>& stot,
    int& nSub, double gamma, uint64_t seed) {
  const int n = g.n;
  const double M = g.totalDeg;
  if (M <= 0.0 || n == 0) return;
  // 子分区: 每节点独立子社区 度即聚合值
  smemb.resize(n); stot.resize(n);
  std::vector<int> sz(n, 1);  // 子社区大小
  for (int v = 0; v < n; v++) { smemb[v] = v; stot[v] = g.deg[v]; }
  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  std::mt19937_64 rng(seed);
  for (size_t idx = order.size() - 1; idx > 0; idx--) {  // Fisher-Yates
    size_t ri = (size_t)(rng() % (idx + 1));
    std::swap(order[idx], order[ri]);
  }
  for (int v : order) {
    int sc = smemb[v];
    if (sz[sc] != 1) continue;  // 只有单点子社区参与
    const int cc = coarse[v];   // 粗社区约束
    // 候选: 同粗社区邻居的子社区(排除自身)
    std::vector<int> cand;
    for (int k = g.row[v]; k < g.row[v + 1]; k++) {
      int u = g.adj[k];
      if (u == v) continue;
      if (coarse[u] != cc) continue;
      int s2 = smemb[u];
      bool dup = false;
      for (int x : cand) if (x == s2) { dup = true; break; }
      if (!dup) cand.push_back(s2);
    }
    const size_t kk = cand.size();
    if (kk == 0) continue;
    int target = cand[(size_t)(rng() % kk)];  // 随机挑一个同粗社区候选
    if (target == sc) continue;
    // 增益(把v并入target) 自环不参与
    double wOld = 0, wNew = 0;
    for (int k = g.row[v]; k < g.row[v + 1]; k++) {
      int u = g.adj[k];
      if (u == v) continue;  // 自环不影响增益
      double w = g.awt[k];
      int s2 = smemb[u];
      if (s2 == sc) wOld += w;
      if (s2 == target) wNew += w;
    }
    double kv = g.deg[v];
    double gain = (wNew - gamma * kv * (stot[target] + kv) / M)
                - (wOld - gamma * kv * stot[sc] / M);
    if (gain >= 0) {
      smemb[v] = target;
      stot[sc] -= g.deg[v];
      stot[target] += g.deg[v];
      sz[sc] = 0;
      sz[target] += 1;
    }
  }
  // 压缩编号
  std::vector<int> map2(n, -1);
  int kk2 = 0;
  for (int v = 0; v < n; v++) {
    int s2 = smemb[v];
    if (map2[s2] < 0) map2[s2] = kk2++;
  }
  for (int v = 0; v < n; v++) smemb[v] = map2[smemb[v]];
  nSub = kk2;
  // stot按新编号重排(前kk2个用得到即可 上层collapse只关心成员)
  std::vector<double> ntot(kk2, 0.0);
  for (int v = 0; v < n; v++) ntot[smemb[v]] += g.deg[v];
  stot = std::move(ntot);
}

//--------------------------------  按分区收缩成超图  --------------------------------//
static CSR collapse(const CSR& g, const std::vector<int>& memb,
    const int nComm, std::vector<int>& map) {
  map = memb;  // 节点 -> 超图节点
  // 汇总原规范边 -> (cu,cv) 权重(自环计一次)
  std::vector<std::pair<std::pair<int,int>, double>> es;
  for (int u = 0; u < g.n; u++) {
    for (int e = g.row[u]; e < g.row[u + 1]; e++) {
      int v = g.adj[e];
      if (u > v) continue;  // 无向边按规范方向统计一次
      double w = g.awt[e];
      int cu = map[u], cv = map[v];
      if (cu > cv) std::swap(cu, cv);  // 规范化为(u<=v)供buildFromSorted
      es.emplace_back(std::make_pair(cu, cv), w);
    }
  }
  __gnu_parallel::sort(es.begin(), es.end());  // 并行排序(结果确定)
  std::vector<int> from, to;
  std::vector<double> wt;
  for (size_t i = 0; i < es.size();) {  // 同(社区对)合并(整数计数 求和精确)
    size_t j = i; double s = 0;
    while (j < es.size() && es[j].first == es[i].first) { s += es[j].second; j++; }
    from.push_back(es[i].first.first); to.push_back(es[i].first.second);
    wt.push_back(s); i = j;
  }
  return CSR::buildFromSorted(nComm, from, to, wt);  // 已排序唯一 直接建CSR
}

//--------------------------------  入口  --------------------------------//
// edges: (u,v,w) 忽略权重(无权); 无向; resolution=γ;
// n_iterations: 重复聚合链轮数(链已收敛时额外轮次为空操作); <1 视为1
inline std::vector<uint32_t> cluster(
    const std::vector<std::tuple<int,int,float>>& edges,
    int nodeCount, float resolution, unsigned seed = 42, int n_iterations = 1) {
  std::vector<uint32_t> result;
  if (nodeCount <= 0) return result;
  result.assign((size_t)nodeCount, 0);
  const double gamma = (double)resolution;
  // 原始边表(无权 权重=1); 重复边(同u,v多条)由CSR合并为一条并累加权值 —— 无向多重边语义;
  // 上游buildGraph已做全图去重 正常不会有重复边
  std::vector<std::pair<int,int>> e0;
  std::vector<double> w0;
  e0.reserve(edges.size());
  for (const auto& e : edges) {
    int u = std::get<0>(e), v = std::get<1>(e);
    if (u < 0 || v < 0 || u >= nodeCount || v >= nodeCount) continue;
    if (u == v) continue;  // 自环对模块度无影响 忽略
    e0.emplace_back(u, v);
    w0.push_back(1.0);
  }
  CSR g = CSR::build(nodeCount, e0, w0);
  if (n_iterations < 1) n_iterations = 1;  // 至少一轮
  std::vector<int> curId(nodeCount);
  for (int i = 0; i < nodeCount; i++) curId[i] = i;
  for (int iter = 0; iter < n_iterations; iter++) {  // 重复聚合链(收敛后为空操作)
    for (int level = 0; level < 32; level++) {  // 在收缩层级上循环直到无合并
      std::vector<int> memb(g.n);
      for (int v = 0; v < g.n; v++) memb[v] = v;  // 每层从单点开始
      // identity分区下 tot[c]=deg[c]; 直接拷贝(元素级 与顺序无关 保证可复现;
      // 勿用并行原子累加: double加法无结合律 线程调度会引入舍入差异)
      std::vector<double> tot = g.deg;
      int moved = localMoving(g, memb, tot, gamma);
      if (moved == 0) break;  // 无任何合并 收敛
      // 精炼(在粗分区约束内细化)
      std::vector<int> smemb;
      std::vector<double> stot;
      int nSub = 0;
      refine(g, memb, smemb, stot, nSub, gamma,
          (uint64_t)seed + (uint64_t)iter * 0x85ebca6bu
          + (uint64_t)level * 0x9e3779b1u);
      // 收缩(按精炼分区)
      std::vector<int> map;
      CSR g2 = collapse(g, smemb, nSub, map);
      // 原节点curId前进一步
      for (int i = 0; i < nodeCount; i++) curId[i] = map[curId[i]];
      g = std::move(g2);
      if (g.n <= 1) break;  // 只剩一个超点
    }
    if (g.n <= 1) break;  // 已收敛到单点 无需更多轮
  }
  // 顶层每个节点=一簇 压缩编号输出
  std::vector<int> map2(nodeCount, -1);
  int k = 0;
  for (int i = 0; i < nodeCount; i++) {
    if (map2[curId[i]] < 0) map2[curId[i]] = k++;
  }
  for (int i = 0; i < nodeCount; i++) result[i] = (uint32_t)map2[curId[i]];
  return result;
}

}  // namespace LeidenOmp
#endif  // LEIDENPARALLEL_HPP
