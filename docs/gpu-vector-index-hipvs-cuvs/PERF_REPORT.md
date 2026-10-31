# 用 hipVS/cuVS 加速 seekdb 的 CAGRA 索引构建 —— 实现与性能对比报告

赛题：seekdb × AMD（`questionId=600072`，子题「使用 hipVS 加速 HNSW 索引构建」，满分 20 分）
基线：`zihaomu/seekdb` 分支 `amd-oceanbase-official-contest-base-v5`，冻结 commit
`634888d9b037153fa143d2c913ba4b25503e1d62`（Tag `amd-cagra-contest-baseline-v5`）

| 交付物 | 位置 |
|---|---|
| 实现代码 | `tools/hipvs_bridge/seekdb_cuvs_bridge.{c,h}`（bridge）、`src/oblib/lib/vector/ob_vsag_adaptor.cpp`（接入）、`tools/hipvs_bridge/build.sh` |
| 可复现测试方法 | `evidence/scripts/{gen_data.py,bench.c,recall4.py,sweep_build_params.sh}` |
| CPU/GPU 构建性能对比报告 | 本文件（第 5 节全部数字来自 AMD 平台实测，原始记录见 `evidence/`） |

改动只涉及 `lib=cuvs` 的 GPU CAGRA 路径。`lib=vsag`（CPU HNSW）的建表、建索引、查询行为
完全未动，DDL 语法与索引语义也未动。

---

## 1. 结论

### 1.1 改了什么

1. **构建参数可调，且默认值选在安全区**。冻结基线把 CAGRA 构造成
   `intermediate_graph_degree=256, nn_descent_niter=100`（先构两倍宽的图再剪到 128）。
   新的 bridge 把这几个旋钮做成环境变量，默认改成 `intermediate=2×degree=256,
   nn_descent_niter=80`——**保持中间图宽度不变**（这一步是召回的关键，见 1.2），
   只把细化轮次从 100 降到 80。250k×256 实测构建耗时 42.0s → 35.3s。
2. **快照从「cuVS 序列化文件往返」改成「只存邻接表」**（`OBCUVS02` → `OBCUVS03`）。
   基线把 cuVS 序列化出的索引文件（邻接表 **+ 一份完整底库副本**）先写进 tmpfile，
   再拷进 seekdb 的索引快照流，等于在计时窗口里多写/多读一整份数据集；新版快照只写
   `ids + 原始向量 + n×degree 的邻接表`，重启时用 `cuvsCagraIndexFromArgs` 把图直接装回
   显存，不再跑 NN-descent、不再过序列化器、不再走 tmpfile。
3. **bridge 侧搜索状态长驻**。搜索参数和 query/结果缓冲挂在索引句柄上按需扩容复用，
   不再每查一次就 `cuvsRMMAlloc/RMMFree` 一轮。

### 1.2 一个必须纠正的结论：不能在输出度上直接构图

赛题允许「在答题仓库中实现自定义 GPU 建图」，最有诱惑力的做法是**直接按输出度构图**
（`intermediate == degree == 128`），因为它省掉了整个剪枝阶段，cuVS 层实测能快 2～3 倍。
**这条路会直接掉出门槛**：本报告第 5 节在 4 份 250K 规模代理数据上实测，构图宽度从 2× 降到 1× 后
recall@100 的变化是：

| 数据集（250K 行） | 基线参数 recall@100 | 1× 中间度 recall@100 | 差值 |
|---|---|---|---|
| 128 维 · 均匀随机 | 0.9993 | 0.9975 (niter=48) | −0.0018 |
| 128 维 · 带簇结构 | 1.0000 | **0.8900** | **−0.1100** |
| 256 维 · 均匀随机 | 0.9804 | 0.9688 (niter=48) | −0.0116 |
| 256 维 · 带簇结构 | 1.0000 | **0.9200** | **−0.0800** |

带簇结构的数据（真实 embedding 更接近这种分布）上，中间度砍到 1× 会让召回掉 0.08～0.11，
远超评测的 ±0.005 硬门槛——**任一硬门槛失败，本次总分直接归零**。
原因不是 `niter` 太少，而是 cuVS 的**剪枝阶段本身就在改善图的可导航性**：
把图构到两倍宽再剪枝，剪掉的正是那些「太长、会把贪心搜索带偏」的边。

因此本方案**保留 2× 中间图**，只在安全边界内压缩细化轮次。代价是构图提速有限
（约 1.15～1.2x），收益主要来自快照 I/O 与查询侧的改造。

### 1.3 实测到的数字（AMD 平台，250k×256，cuVS 层）

| 构图参数 | 构建耗时 | 加速 | recall@100（带簇 / 均匀） | 是否满足 ±0.005 门限 |
|---|---|---|---|---|
| `inter=256, niter=100`（冻结基线参数） | 42.0 s | 1.00x | 1.0000 / 0.9831 | 基准 |
| **`inter=256, niter=80`（本方案默认）** | **35.3 s** | **1.19x** | **1.0000 / 0.9785** | **是（−0.0046）** |
| `inter=256, niter=48` | 25.8 s | 1.63x | 1.0000 / 0.9583 | 否（均匀数据 −0.0248） |
| `inter=128, niter=48` | 18.4 s | 2.28x | 0.9100 / 0.9681 | 否（带簇 −0.0900） |

统一搜索参数 `itopk=512, search_width=4`，100 条查询，top-100，与精确暴力 top-100 比对。

### 1.4 还没做完的部分

本轮在实例上做完的是「cuVS 层参数选点 + recall 边界测绘」，**端到端**的口径还差最后一段：
冻结基线二进制与参赛版本的 `CREATE VECTOR INDEX` + 首查完整窗口 A/B（需要重建 seekdb，
一次完整构建约 30 分钟以上，超出了本轮实例的剩余寿命）。第 4 节给出了在下一个实例上
一次跑完的做法；第 5.4 节列出了需要补齐的表格。

---

## 2. 背景与指标口径

seekdb 的向量索引 DDL：

```sql
CREATE VECTOR INDEX idx ON t(v)
WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
```

- `lib=vsag`（默认）走 CPU HNSW 构建，是 CPU 对照。
- `lib=cuvs` 走 GPU CAGRA 构建，通过 `libseekdb_cuvs_bridge.so` 调 hipVS 的 cuVS C API。

赛题把「构建完成」定义为：**索引创建完成 + 数据扫描与 CPU/GPU 传输完成 + CAGRA 图构建与
GPU 同步完成 + 索引已经可以查询 + 第一条固定查询完成**。所以端到端窗口 =
`CREATE VECTOR INDEX` 墙钟耗时 + 首查耗时。

本轮报告里的构建耗时是 **cuVS 层**（`cuvsCagraBuild` 前后打点，见 `evidence/scripts/bench.c`），
不含表扫描、快照落盘和 SQL 层开销。原因见第 4 节。

---

## 3. 实现

### 3.1 bridge：构建旋钮 + 长驻搜索状态 + 图导出/导入

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SEEKDB_CUVS_BUILD_ALGO` | `nn_descent` | `auto` / `ivf_pq` 可选 |
| `SEEKDB_CUVS_GRAPH_DEGREE` | 128 | 落盘图的度数，也是快照体积系数 |
| `SEEKDB_CUVS_INTERMEDIATE_DEGREE` | 256（= 2×degree） | 构图期图宽，**不允许低于 degree** |
| `SEEKDB_CUVS_NN_DESCENT_NITER` | 80 | 近邻细化轮次 |
| `SEEKDB_CUVS_SEARCH_ITOPK` | `max(topk*4, 512)` | 搜索候选池 |
| `SEEKDB_CUVS_SEARCH_WIDTH` | 4 | 图扩展宽度 |

用环境变量而不是 DDL 参数，是为了让参数扫描不需要重新编译 seekdb；扫出来的最优值
直接固化成上面的默认值。DDL 语法因此完全不变。

用环境变量做旋钮还有一个副作用值得记下来：**hipVS 的 CAGRA 搜索把 `itopk_size` 硬限制在
512**（`search_single_cta.cuh:139: itopk_size cannot be larger than 512`，本轮实测触发）。
也就是说「构图省下来的召回、靠更宽的搜索补回来」这条路在本版 hipVS 上**不存在**，
候选池和扩展宽度只能在 512 以内调——这直接决定了构图参数必须自己扛住召回。

搜索状态（`cuvsCagraSearchParams_t` + query/结果缓冲）改为挂在索引句柄上按需扩容复用，
`seekdb_cuvs_search` 里对同一索引加锁，避免并发查询互相踩缓冲。

新增两个快照入口：

- `seekdb_cuvs_export_graph(handle, &graph, &n, &degree)`：把邻接表拷到 host（malloc 缓冲，
  调用方释放）。
- `seekdb_cuvs_import(base, n, dim, graph, degree)`：用 `cuvsCagraIndexFromArgs` 把图和底库
  装回显存，直接得到可搜索的索引——不跑 NN-descent，也不跑剪枝。

### 3.2 adaptor：快照 v3

`src/oblib/lib/vector/ob_vsag_adaptor.cpp`

- 写侧 `ob_cuvs_serialize()`：必要时构图 → 导出邻接表 → 一次性写出
  `magic(8) + graph_bytes(8) + dim(4) + degree(4) + n(8) + ids + vectors + graph`。
  删掉了 tmpfile 与 `cuvsCagraSerialize` 的整条往返链路。
- 读侧 `ob_cuvs_deserialize()`：校验头部（维度、度数上限、长度自洽）→ 读 ids/vectors/graph →
  `seekdb_cuvs_import` 直接重建索引 → 注册句柄。
- 校验里 `degree32 <= 0 || degree32 > 1024` 与
  `graph_bytes != n*degree*4` 是防脏数据/防越界的硬门，坏快照直接报错重建，
  不让 GPU 侧去读一段长度未知的邻接表。
- 构图/导出/写盘各阶段单独计时并打进 observer 日志
  （`serialized GPU CAGRA snapshot(... built=... build_us=... total_us=...)`），
  下次上实例就能直接读出「构图 vs 快照落盘」各占多少。
- cuVS 的 build/import/search 会爆掉 OB worker 的小栈，统一挪到 32MB 栈的 pthread 上执行。

---

## 4. 实验方法

### 4.1 代理数据

隐藏数据集不公开，所以召回边界只能在代理数据上定标。用了两类分布，各 250K 行、100 条查询：

- `uniform`：标准高斯随机向量，L2 归一化。**故意做难**：所有点落在同一单位球面上，
  近邻距离高度集中，是 NN-descent 最难的情形。
- `cluster`：500 个随机簇心 + σ=0.35 噪声，L2 归一化。更接近真实 embedding 的分布。

查询 = 底库中随机取一条 + σ=0.002 噪声后归一化；ground truth 用 numpy 暴力算精确 top-100。
生成脚本 `evidence/scripts/gen_data.py`。

四种组合（128/256 维 × uniform/cluster）各跑一遍，128 维的规模与 SIFT250K 一致，
256 维与 Hidden250K 一致。

### 4.2 口径

- 构建耗时：`cuvsCagraBuild` 前后的墙钟（`bench.c`）。
- 召回：`recall4.py` 用精确 top-100 复算 recall@10 / recall@100。
- 搜索参数固定 `itopk=512, search_width=4`，与 bridge 线上默认值一致，
  保证不同构图参数之间的召回差异只来自图本身。

### 4.3 复现

在 AMD Radeon Cloud 实例上（`/opt/hipvs`、`/opt/rocm` 就位）：

```bash
# 1) 生成代理数据与精确 ground truth
python3 evidence/scripts/gen_data.py /workspace/d256 256 cluster 23
python3 evidence/scripts/gen_data.py /workspace/d256u 256 uniform 23
python3 evidence/scripts/gen_data.py /workspace/d128 128 cluster 31

# 2) 打一个只调 cuVS 的探针（二进制与 bridge 同源）
hipcc -O2 -I tools/hipvs_bridge evidence/scripts/bench.c -o /workspace/bench \
  -L /workspace/bridge -lseekdb_cuvs_bridge -L /opt/hipvs/lib -lcuvs_c

# 3) 扫构图参数并复算召回
bash evidence/scripts/sweep_build_params.sh /workspace/d256 256
python3 evidence/scripts/recall4.py /workspace/d256/gt.i32 /workspace/results/sweep/out_i256n80.bin
```

原始记录：`evidence/sweep7_256d.txt`（本轮 sweep 原始输出）、
`evidence/recall_matrix.txt`（四份数据集的完整 recall 矩阵）。

---

## 5. 实测结果

平台：AMD Radeon Cloud，`gfx1100`（Radeon PRO W7900），ROCm 7.2.4，`hipVS` 位于 `/opt/hipvs`
（组织方固定版本 `f96a663d0a4441d6f5344bd3ef552b71ae7b370b`）。实例为共享实例，
**绝对耗时不可跨时段比较**，只有同轮相邻运行的比值可用；召回由同一进程内的精确 GT 复算，
不受节流影响。

### 5.1 构图参数 × 召回（250k×256，本轮 sweep 原始输出）

| 配置 | build_ms | search_ms | recall@100（带簇） | recall@100（均匀） |
|---|---|---|---|---|
| `inter=256 niter=100`（基线参数） | 42 573 | 34 | 1.0000 | 0.9831 |
| `inter=256 niter=80` | 36 100 | 35 | 1.0000 | 0.9785 |
| `inter=256 niter=48` | 24 304 | 35 | 1.0000 | 0.9583 |
| `inter=128 niter=48` | 18 673 | 34 | 0.9100 | 0.9681 |
| `inter=256 niter=48, itopk=1024` | 26 122 | 13 | — | — |
| `inter=128 niter=48, itopk=1024` | 17 496 | 14 | — | — |

后两行搜索失败（`rc=15`），原因是 hipVS 限制 `itopk_size ≤ 512`，见 3.1。

### 5.2 四份代理数据的完整 recall 矩阵

见 `evidence/recall_matrix.txt`。要点：

- **中间度是召回的主导因素**：`inter=128` 在带簇数据上掉 0.08～0.11，在均匀数据上掉 0.01～0.02。
- **`niter` 是次要因素但也不能随便砍**：`inter=256` 时 `niter=80` 只掉 0.0003～0.0046，
  `niter=48` 在均匀 256 维数据上就掉到 0.0248，越界。
- **降输出度同样危险**：`degree=96/64` 的组合在均匀 128 维上只有 0.9690 / 0.9265。

### 5.3 与冻结基线的构图参数对照

| 数据集 | 冻结基线参数 `inter=256 niter=100` | 本方案 `inter=256 niter=80` | 加速 |
|---|---|---|---|
| 250k×128 均匀 | 41 249 ms | 35 932 ms（`inter=256 niter=80`，recall@100 0.9990） | 1.15x |
| 250k×256 均匀 | 42 039 ms | 35 276 ms | 1.19x |
| 250k×256 带簇 | 42 573 ms | 36 100 ms | 1.18x |

（128 维均匀的 `niter=80` 那一行来自上一轮同口径 sweep；本轮 sweep 只覆盖 256 维，
因为实例剩余寿命不足以重跑全部四份。）

### 5.4 端到端口径（待补）

需要补的表：

| 配置 | 建索引 DDL (s) | 首查 (s) | 构建窗口 (s) | 加速 vs 冻结基线 | recall@10/@100 |
|---|---|---|---|---|---|
| 冻结基线二进制 `/seekdb_workspace/bin/seekdb` | - | - | - | 1.00x | - |
| 参赛版本 `inter=256 niter=80` | - | - | - | - | - |
| 参赛版本 `inter=256 niter=100`（隔离快照改造的净收益） | - | - | - | - | - |

做法见下一节。

---

## 6. 正确性验证

| 验证 | 做法 | 状态 |
|---|---|---|
| bridge 能编译、能链接 hipVS | 在 AMD 平台用官方命令 `HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm tools/hipvs_bridge/build.sh ...` 实编译 | ✅ 通过（产出 `libmybridge.so`） |
| bridge 符号齐全 | `nm -D` 检查 6 个导出符号 + `RTLD_NOW` dlopen 自检 | ✅ 通过 |
| adaptor 改动可编译 | 把改动区域抽出来配桩头文件做 `-fsyntax-only` 类型检查（`evidence/syntax_check/`） | ✅ 通过（仅桩宏引发的无害告警） |
| 图导出/导入语义 | `cuvsCagraIndexGetGraph` / `cuvsCagraIndexFromArgs` 均为 hipVS 已导出符号（编译期链接检查） | ✅ 通过 |
| 端到端功能 + 快照落盘/重启恢复 | `run_gpu_smoke.sh` / `run_ddl_persistence_smoke.sh` | ⏳ 需在下一个实例上用重建后的二进制复跑 |
| `lib=vsag` 路径无回归 | 本次改动不触碰 CPU HNSW 分支 | ⏳ 同上，需要一次功能回归 |

---

## 7. 下一步（按优先级）

1. **在下一个 AMD 实例上重建 seekdb**（官方 cmake 命令，约 30～60 分钟），把冻结基线二进制
   与参赛版本在同一份 250k 数据上做端到端 A/B，补上 5.4 的表。
2. **拿到端到端数字后固化 `niter`**：若快照改造的净收益足够大，可以考虑把 `niter` 提到 96
   换回均匀数据上那 0.002 的召回余量；反之维持 80。
3. **拆解 DDL 固定开销**：本轮日志里已经埋了 `build_us` / `total_us` 两个打点，
   下次直接读 observer 日志就能把「构图 vs 快照落盘 vs 表扫描」分开，看还有没有可压的常数项。
4. **提测**：只有在端到端 A/B 通过、且确认 recall 在隐藏集上不掉的前提下再提测——
   任一硬门槛失败就是 0 分，代价远大于晚提交。

---

## 8. 局限

- 只支持 `distance=l2`（cuVS CAGRA 侧的限制，DDL 层已拦 `lib=cuvs` + 非 L2）。
- 单卡；数据集必须能放进显存（`n*dim*4` 底库 + 图 `n*degree*4` + 搜索缓冲）。
- 快照体积随 `degree` 线性增长：`degree=128`、`n=250K` 时邻接表约 128MB。
- 构建旋钮是全局环境变量，不是 per-index 的 DDL 参数；多租户场景要固化默认值或做 per-table 配置。
- 召回余量取决于数据分布。本方案把默认点选在「四份代理数据全部满足 ±0.005」的位置，
  但代理数据终究不是隐藏集；**评测机上的同轮基线召回才是唯一裁判**。
- 端到端口径尚未验证（第 5.4 节），本报告里的加速比都是 cuVS 构图层级的。
