# 用 hipVS/cuVS 加速 seekdb 的 CAGRA 索引构建 —— 实现与性能对比报告

赛题：seekdb × AMD（`questionId=600072`，子题「使用 hipVS 加速 HNSW 索引构建」，满分 20 分）
基线：`zihaomu/seekdb` 分支 `amd-oceanbase-official-contest-base-v5`，冻结 commit
`634888d9b037153fa143d2c913ba4b25503e1d62`（Tag `amd-cagra-contest-baseline-v5`）
参赛分支：`amd-cagra-v2`（HEAD `f310b91` + 本次把默认构图参数固化的 bridge 改动）
所有数字都在 **AMD Radeon Cloud 实例**（`gfx1100` / ROCm 7.2.4 / `/opt/hipvs`）上实测，
原始记录见 `evidence/`。

| 交付物 | 位置 |
|---|---|
| 实现代码 | `tools/hipvs_bridge/seekdb_cuvs_bridge.{c,h}`（bridge）、`src/oblib/lib/vector/ob_vsag_adaptor.cpp`（接入）、`tools/hipvs_bridge/build.sh` |
| 可复现测试方法 | 第 4 节协议 + `evidence/scripts/`（cuVS 层探针、代理数据生成、召回复算）+ `evidence/*.sh`（端到端 A/B、显存采样、重启校验） |
| CPU/GPU 构建性能对比报告 | 本文件第 5.5 节（同一实例、同一份表、`lib=vsag` CPU HNSW vs `lib=cuvs` GPU CAGRA） |

改动只涉及 `lib=cuvs` 的 GPU CAGRA 路径。DDL 语法、指标语义、`lib=vsag`（CPU HNSW）
分支的行为都未改动，第 6 节用基线自带的 DDL 契约用例和一次真实 CPU 构建验证了这一点。

---

## 1. 结论

### 1.1 参赛配置与端到端收益

口径按赛题定义：**构建窗口 = `CREATE VECTOR INDEX ... WITH (lib=cuvs)` 墙钟 + 第一条固定查询墙钟**。
主口径表（256 维四轮、128 维两轮）每一对都是同一实例、同一时段、同一份 250K 数据、同一组
observer 参数下的**同轮相邻**运行；共享实例会被节流，跨时段的绝对值不可比，所以另附一张
「跨时段参照」表并明确它不能当成绩。

**250k×256（Hidden250K 规模）——主口径：参赛默认值的同轮重复 A/B**

每一轮都在同一时段内先跑冻结基线二进制、再跑参赛二进制，同表、同 DDL、同 observer 参数，
参赛臂**不设任何 `SEEKDB_CUVS_*` 构图变量**（也就是线上默认值）。`evidence/ff_ab.log` 是
第 1 轮的原始输出，`evidence/ff_rounds_summary.txt` 是四轮的离线复算（含每轮的 QPS 和显存）。

| 轮 | 基线窗口 (ms) | 参赛窗口 (ms) | 加速 | Δrecall@10 | Δrecall@100 | QPS 几何比 | 显存峰值 基线/参赛 |
|---|---|---|---|---|---|---|---|
| 1 | 49 820 | 49 367 | 1.009x | +0.0010 | +0.0014 | 1.012 | 0.803 / 0.803 GB |
| 3 | 49 771 | 49 968 | 0.996x | −0.0010 | +0.0007 | 0.990 | 0.803 / 0.803 GB |
| 4 | 49 567 | 47 073 | 1.053x | +0.0010 | −0.0010 | 1.060 | 0.803 / 0.803 GB |
| 5 | 48 891 | 48 864 | 1.001x | −0.0010 | +0.0010 | 1.021 | 0.803 / 0.803 GB |
| **均值** | | | **1.015x** | 范围 ±0.0010 | 范围 −0.0010～+0.0014 | **1.021**（最差 0.990） | **1.000x** |

第 2 轮的基线 observer 没能起来（`FAIL start`），整轮作废、不计入上表。
参赛臂自己的绝对召回是 `@10 0.9980～0.9990`、`@100 0.9795～0.9805`（门槛是 0.90 / 0.85）。

**其它轮次（跨时段参照，只用来排序，不能当成绩）**

| 二进制（git revision） | 构图参数 | DDL (ms) | 首查 (ms) | 窗口 (ms) | 对同批基线 | recall@10 / @100 | QPS@10 / @100 | 显存峰值 |
|---|---|---|---|---|---|---|---|---|
| 基线 `634888d`（轮 2） | `inter=256 niter=100` | 50 491 | 56 | **50 547** | 1.00x | 1.0000 / 0.9799 | 67.4 / 54.8 | 0.803 GB |
| 基线 `634888d`（轮 1） | `inter=256 niter=100` | 49 491 | 55 | **49 546** | — | 0.9990 / 0.9797 | 69.8 / 55.2 | 0.803 GB |
| 参赛版本 `f310b91`（零环境变量认证运行） | 同上（默认值即基线同图） | 48 358 | 115 | **48 473** | 1.02x（vs 轮 1/轮 2，跨轮） | 0.9970 / 0.9795 | 70.2 / 55.9 | 0.803 GB |
| 参赛版本（同配置复跑） | 同上 | 48 131 | 67 | **48 198** | 1.049x（vs 轮 2，跨轮） | 1.0000 / 0.9795 | 70.9 / 52.6 | 0.803 GB |

**250k×128（SIFT250K 规模，同轮 A/B，bridge 用与交付件字节一致的 `out_final`）**

| 二进制 | 构图参数 | DDL (ms) | 首查 (ms) | 窗口 (ms) | 加速 | recall@10 / @100 | QPS@10 / @100 | 显存峰值 |
|---|---|---|---|---|---|---|---|---|
| 基线 `634888d` | `inter=256 niter=100` | 45 609 | 54 | **45 663** | 1.00x | 1.0000 / 0.9997 | 73.4 / 62.4 | 0.684 GB |
| 参赛版本 | 同上 | 45 047 | 164 | **45 212** | **1.010x** | 1.0000 / 0.9993 | 73.7 / 62.3 | 0.684 GB |
| 上一轮同口径复跑 | 同上 | 46 000 | 57 | 46 057 | 1.006x（vs 该轮基线窗口 46 319） | 1.0000 / 0.9997 | — | 0.684 GB |

一句话：**同轮实测加速比 256 维均值 1.015x、128 维 1.006～1.010x，四条硬门槛每一轮都通过**；
参赛默认值刻意做出与基线**完全相同的图**，所以构建时间与基线同量级（同轮抖动 0.996～1.053x，
均值 1.015x 来自把窗口里的常数项拿掉），换来的是召回差稳定在 ±0.001 以内（门槛 0.005）、
QPS 几何比最差一轮 0.990（门槛 0.90）、显存峰值逐轮与基线相同（1.000x，门槛 1.25x）。
第 1.3 节说明为什么没有更快的安全选项，第 7 节给出对应的分数推算。

### 1.2 改了什么

1. **快照从「cuVS 序列化文件往返」改成「只存邻接表」**（`OBCUVS02` → `OBCUVS03`）。
   基线把 `cuvsCagraSerialize` 产出的索引文件（邻接表 **+ 一份完整底库副本**）先写进 tmpfile，
   再拷进 seekdb 的索引快照流——等于在计时窗口里多写多读一整份数据集（实测基线
   `graph_bytes=256 001 713`，我们的原始邻接表 `128 000 000`）。新版快照只写
   `magic + 长度 + ids + 原始向量 + n×degree 邻接表`，重启时用 `cuvsCagraIndexFromArgs`
   把图直接装回显存：不跑 NN-descent、不过序列化器、不走 tmpfile。
   256 维实测这一段（导出 + 落盘）只花 **0.31 s**（observer 打点
   `build_us=43 351 029 → total_us=43 663 360`）。
2. **构图参数做成环境变量，并把默认值固化在与基线完全相同的图上**
   （`intermediate=2×degree=256`、`nn_descent_niter=100`、`graph_degree=128`）。
   这不是偷懒——第 1.3 节说明：在这版 hipVS 上，任何**能在召回门槛内存住**的构图参数改动，
   收益都只有几个百分点；而任何**收益明显**的改动都会撞门槛。
3. **bridge 侧搜索状态长驻**：搜索参数与 query/结果缓冲挂在索引句柄上按需扩容复用，
   不再每次查询走一轮 `cuvsRMMAlloc/RMMFree`；同一索引的查询串行化，避免并发踩缓冲。
4. **额外的可选建图器（默认关闭）**：`SEEKDB_CUVS_BUILD_ALGO=3` 走
   `cuvsAllNeighborsBuild` 精确 kNN 图，实测窗口 **25 015 ms（1.98x）**、召回
   `1.0000 / 0.9998`（比基线还高）——但显存峰值 1.438 GB = 基线的 **1.79 倍**，
   超了「≤1.25× 基线」这条硬门槛，所以**不能作为默认值**，只保留为显式开关。

### 1.3 为什么构建只快了 1.5%：三条更快的路都撞在硬门槛上

赛题的计分是 `14 × ln(复合加速比)/ln(3.0)`，但**任一硬门槛失败总分直接归零**。
实测下来的四条路：

| 方案 | 256 维窗口 | 加速 | recall@10 / @100（同轮基线 1.0000 / 0.9799） | 显存峰值 | 判定 |
|---|---|---|---|---|---|
| 参赛默认：`inter=256 niter=100` | 47 073～49 968 ms（同轮 4 轮） | **1.015x（同轮均值）** | 1.0000 / 0.9795（Δ 在 ±0.0010 内） | 0.803 GB = 1.00x | **✅ 全部门槛通过** |
| `inter=256 niter=80` | 40 863 ms | 1.237x（同轮 vs `nn_base`） | **0.9950** / 0.9753（**−0.0050** / −0.0046） | 0.803 GB | ❌ 召回差 = 门槛上限，零余量 |
| `inter=256 niter=64` | 35 899 ms | 1.409x（跨轮参照） | 0.9950 / 0.9706（−0.0093 / −0.0093） | 0.803 GB | ❌ 召回越界 |
| `algo=3` 精确 all-neighbors | 25 015 ms | **1.981x**（同轮 vs `ship_base`） | 1.0000 / 0.9998（更好） | **1.438 GB = 1.79x** | ❌ 显存越界 |
| `inter=128`（=输出度，省掉剪枝） | 18 673 ms（cuVS 层） | 2.28x（cuVS 层） | 带簇数据 recall@100 **0.9100**（−0.0900） | — | ❌ 召回大幅越界 |

`niter_sweep.log` 里只有参赛臂、没有基线臂，所以 `n256i100`/`n256i64` 的比值是拿轮 2 基线
（`ab_nn.log` 的 50 547 ms）做的跨时段参照——共享实例上这个口径只能用来排序，不能当成绩。
参赛默认值的**同轮**加速比在第 1.1 节的重复轮次表里单独给出。

另外三条试过但没有出路的路：

- **IVF-Flat 自搜索求精确 kNN**（第 3 种精确图来源）：256k×256 上 `total_ms=55 419~61 597`、
  峰值 **1.940 GB**，比 nn_descent 更慢且更吃显存，`recall@100 0.9998` 与 all-neighbors 相同——
  慢且不省内存，直接放弃。
- **all-neighbors + `n_clusters>1`**（压显存）：簇数 4/8/16/32/64 的峰值降到 0.598～0.785 GB
  （低于基线，能过显存门），但 `build_ms` 变成 34.7～88.5 s，且 `recall@100` 只有
  0.9758～0.9774（相对基线 −0.0025～−0.0041）——省下的显存是用「不再精确」换来的，
  召回余量被吃光，收益也只剩 1.2x 左右，不划算。
- **fp16 底库构图**：256 维 `total 20 433 ms` vs fp32 `20 374 ms`（无收益）。
- **提前释放设备端底库缓冲**：搜索直接 `rc=15`（`hipErrorInvalidValue` in
  `dataset_descriptor_host::state::eval`）——CAGRA 的图搜索要引用底库向量，
  OBCUVS03 快照里的底库必须留在显存里，省不掉。

所以本方案的立场是：**把图做得和基线一模一样（因此召回不可能越界），把窗口里所有非构图的
常数项（tmpfile 往返、重复序列化、每查一次的显存分配）拿掉**。
剩下的窗口里 87～92% 是 `cuvsCagraBuild` 本身（见 5.4 的打点拆分），在必须保住召回的前提下，
这版 hipVS 没给出更多可压的空间。

### 1.4 CPU / GPU 对照

同一实例、同一二进制树、同一份 250k×128 表，只把 DDL 里的 `lib=` 换掉：
CPU HNSW（`lib=vsag`）构建窗口 **157 163 ms**，GPU CAGRA（`lib=cuvs`）**45 663 ms**
——GPU 把窗口压到 CPU 的 **0.29 倍（3.44x）**，同时 recall@10 从 0.4680 升到 1.0000、
recall@100 从 0.3277 升到 0.9997。口径细节（同一条 DDL 的 `ef_search=64` 在两侧含义不同）
见第 5.5 节。

---

## 2. 背景与指标口径

seekdb 的向量索引 DDL：

```sql
CREATE VECTOR INDEX idx ON t(v)
WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
```

- `lib=vsag`（默认）走 CPU HNSW 构建，是本报告的 CPU 对照。
- `lib=cuvs` 走 GPU CAGRA 构建，通过 `libseekdb_cuvs_bridge.so` 调 hipVS 的 cuVS C API。

赛题把「构建完成」定义为：索引创建完成 + 数据扫描与 CPU/GPU 传输完成 + CAGRA 图构建与 GPU
同步完成 + 索引已经可以查询 + 第一条固定查询完成。所以端到端窗口 = DDL 墙钟 + 首查墙钟，
本报告的 `window_ms` 就是这个和。

门槛（任一失败 = 0 分）：

| 门槛 | 要求 |
|---|---|
| 召回绝对值 | SIFT250K recall@10/@100 ≥0.95；Hidden250K recall@10 ≥0.90、@100 ≥0.85 |
| 召回相对值 | 与同轮基线相比 recall 下降 ≤0.005 |
| 查询吞吐 | Top-10 / Top-100 QPS 比的几何平均 ≥0.90 |
| 显存 | 峰值 ≤1.25× 基线 **且** ≤45 GB |
| 单轮时间 | 整个 GPU 轮 ≤15 min |

计分：`build = 14 × clamp(ln(composite)/ln(3.0))`，`composite = r_sift^0.4 × r_hidden^0.6`；
`query = 2 × clamp(ln(qps_ratio)/ln(1.5))`。

---

## 3. 实现

### 3.1 bridge：构建旋钮 + 长驻搜索状态 + 图导出/导入

`tools/hipvs_bridge/seekdb_cuvs_bridge.c`

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SEEKDB_CUVS_BUILD_ALGO` | `0` = nn_descent | `1` = auto、`2` = ivf_pq、`3` = 精确 all-neighbors（**默认关闭，理由见 1.3**） |
| `SEEKDB_CUVS_GRAPH_DEGREE` | 128 | 落盘图的度数，也是快照体积系数 |
| `SEEKDB_CUVS_INTERMEDIATE_DEGREE` | 256（= 2×degree） | 构图期图宽，**不允许低于 degree**（见 5.2） |
| `SEEKDB_CUVS_NN_DESCENT_NITER` | 100 | 近邻细化轮次（80/64 实测越界，见 5.3） |
| `SEEKDB_CUVS_SEARCH_ITOPK` | `max(topk*4, 512)` | 搜索候选池（hipVS 硬上限 512） |
| `SEEKDB_CUVS_SEARCH_WIDTH` | 4 | 图扩展宽度 |

用环境变量而不是 DDL 参数，是为了让参数扫描不需要重新编译 seekdb；扫出来的最优值直接固化成
上面的默认值，DDL 语法因此完全不变。

副作用值得记下来：**hipVS 的 CAGRA 搜索把 `itopk_size` 硬限制在 512**
（`search_single_cta.cuh:139: itopk_size cannot be larger than 512`，实测触发过 `rc=15`）。
也就是说「构图省下来的召回，靠更宽的搜索补回来」这条路在本版 hipVS 上**不存在**，
候选池和扩展宽度只能在 512 以内调——这直接决定了构图参数必须自己扛住召回。

搜索状态（`cuvsCagraSearchParams_t` + query/结果缓冲）挂在索引句柄上按需扩容复用，
`seekdb_cuvs_search` 内对同一索引加锁。

新增两个快照入口：

- `seekdb_cuvs_export_graph(handle, &graph, &n, &degree)`：把邻接表经
  `cuvsCagraIndexGetGraph`（DLPack）拷到 host（malloc 缓冲，调用方用
  `seekdb_cuvs_free_host` 释放）。
- `seekdb_cuvs_import(base, n, dim, graph, degree)`：用 `cuvsCagraIndexFromArgs` 把 host 上的
  图和底库装回显存，直接得到可搜索索引——不跑 NN-descent，也不跑剪枝。
  注意：**底库必须一起交给 cuVS**（图搜索要引用向量），所以导入后的索引仍持有 `n×dim×4` 的
  设备端底库，这就是 1.3 里「提前释放缓冲」失败的原因。

### 3.2 adaptor：快照 v3

`src/oblib/lib/vector/ob_vsag_adaptor.cpp`

- 写侧 `ob_cuvs_serialize()`：必要时构图 → 导出邻接表 → 一次性写出
  `magic(8) + graph_bytes(8) + dim(4) + degree(4) + n(8) + ids + vectors + graph`，
  删掉 tmpfile 与 `cuvsCagraSerialize` 的整条往返链路。
- 读侧 `ob_cuvs_deserialize()`：校验头部（维度、`degree<=1024`、
  `graph_bytes == n*degree*4` 长度自洽）→ 读 ids/vectors/graph → `seekdb_cuvs_import` 重建索引
  → 注册句柄。校验是防脏数据/防越界的硬门，坏快照直接报错重建，不让 GPU 去读一段长度未知的邻接表。
- 构图/导出/写盘分段计时打进日志
  （`serialized GPU CAGRA snapshot(... built=... build_us=... total_us=... graph_bytes=...)`），
  重启恢复打 `deserialized GPU CAGRA snapshot(...)`。第 5.4/6 节的拆分直接读这两行。
- cuVS 的 build/import/search 会爆掉 OB worker 的小栈，统一挪到 32MB 栈的 pthread 上执行。

---

## 4. 实验方法

### 4.1 端到端 A/B 协议

`evidence/ab_run_ship.sh`（256 维）、`evidence/ab128_final.sh`（128 维）、
`evidence/ab_final_same_round.sh`（256 维、参赛默认值、零环境变量同轮 A/B）、
`evidence/final_ship.sh`（参赛默认值认证 + 重启校验）、`evidence/cpu_vsag.sh`
（DDL 契约 + CPU 对照）是同一套骨架，每轮做四件事：

1. **同轮相邻**：基线二进制（`/workspace/src_base/.../seekdb`，启动横幅
   `source revision 1-634888d9b037...`）与参赛二进制（`/workspace/src2/.../seekdb`，横幅
   `source revision 1-f310b91bb60a...`）在同一实例、同一时段各自起一个 observer，
   数据文件、observer 参数（`memory_budget=12G datafile_size=12G cpu_count=32`）、
   DDL 语句、查询集完全一致。revision 由日志头部留档。
2. **窗口计时**：`ddl_start/ddl_end` 两个 `CLOCK_MONOTONIC` 时间戳写进 `.marks`，
   `ddl_ms` = DDL 墙钟，`first_q_ms` = 紧随其后的第一条 `approximate limit 10`，
   `window_ms` = 两者之和。
3. **显存峰值**：`vram_sampler.sh` 每 0.25 s 采一次 `rocm-smi --showmeminfo vram`，
   落 `.vram`；离线按 `.marks` 的窗口区间取窗口内峰值（共享实例上别的租户也占卡，
   所以取**自己那一张卡**的峰值，脚本会打印各卡分布）。
4. **召回与 QPS**：索引建好后跑固定 100 条查询，`limit 100` 与 `limit 10` 各一遍，
   分别算 QPS；`limit 100` 的结果和 **seekdb 自己的精确 KNN**（`exact.txt`，无索引全表精确 top-100）
   比对得 recall@10/@100（`recall_ab.py`）。

代理数据：250K 行、FP32、L2。端到端用**标准高斯随机 + L2 归一化**的均匀分布——
所有点落在同一单位球面上、近邻距离高度集中，是 NN-descent 最难、召回最低的情形，
故意拿它当标定集；参数测绘阶段还用了 500 簇 + σ=0.35 的合成数据
（`evidence/scripts/gen_data.py`），两种分布的结论差异见 5.2。

### 4.2 cuVS 层探针

`evidence/scripts/bench.c` 与 bridge 同源、只调 cuVS（不含 seekdb），用来扫构图参数并复算召回；
`evidence/scripts/recall4.py` 用 numpy 暴力精确 top-100 作 GT；搜索参数固定
`itopk=512, search_width=4`（= bridge 线上默认），保证不同构图参数之间的召回差异只来自图本身。

### 4.3 复现命令

```bash
# 0) bridge 编译（官方命令，HIPVS_ROOT/ROCM_ROOT 就位）
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm tools/hipvs_bridge/build.sh /workspace/out_final

# 1) 端到端同轮 A/B（基线二进制 + 参赛二进制，含显存采样、QPS、召回）
bash evidence/ab_run_ship.sh        # 250k x 256
bash evidence/ab128_final.sh        # 250k x 128
bash evidence/ab_final_same_round.sh  # 250k x 256，参赛默认值（不设任何环境变量）

# 2) 参赛默认值认证 + 重启后快照恢复校验（全程不设构图环境变量）
bash evidence/final_ship.sh

# 3) DDL 契约用例 + CPU(lib=vsag) 同表对照
bash evidence/cpu_vsag.sh

# 4) cuVS 层构图参数扫描与召回复算
python3 evidence/scripts/gen_data.py /workspace/d256 256 cluster 23
hipcc -O2 -I tools/hipvs_bridge evidence/scripts/bench.c -o /workspace/bench \
  -L /workspace/out_final -lseekdb_cuvs_bridge -L /opt/hipvs/lib -lcuvs_c
bash evidence/scripts/sweep_build_params.sh /workspace/d256 256
python3 evidence/scripts/recall4.py /workspace/d256/gt.i32 /workspace/results/sweep/out_i256n80.bin
```

---

## 5. 实测结果

平台：AMD Radeon Cloud，`gfx1100`（Radeon PRO W7900），ROCm 7.2.4，
`hipVS` 位于 `/opt/hipvs`（组织方固定版本 `f96a663d0a4441d6f5344bd3ef552b71ae7b370b`）。
实例为共享实例：**绝对耗时不可跨时段比较**，只有同轮相邻运行的比值可用；
召回由同进程/同轮内的精确 GT 复算，不受节流影响。

### 5.1 端到端窗口（同轮 A/B）

见 1.1 的两张表。原始输出：`evidence/ff_ab.log` + `evidence/ff_rounds_summary.txt`
（参赛默认值的同轮重复 A/B，四轮的窗口/召回/QPS/显存逐轮复算）、
`evidence/ship_run.log`（256 维轮 1）、
`evidence/ab_nn.log`（256 维轮 2）、`evidence/niter_sweep.log`（参赛版各 niter 臂）、
`evidence/ab128.log`（128 维同轮）、`evidence/final_ship.log`（默认值认证 + 重启）。

参赛二进制在零环境变量下的认证运行（`final_ship.log`）：

```
ROWS=250000
RESULT ship ddl_ms=48358 first_q_ms=115 window_ms=48473
pre QPS top100 = 55.9 q/s
pre RECALL recall@10=0.9970 recall@100=0.9795 queries=100
pre QPS top10 = 70.2 q/s
VRAM samples=1312 ddl_peak=0.891GB cards=card2:0.891 card1:0.803 card7:0.026
```

（`card2:0.891` 是同一实例上别的租户，`card1:0.803` 才是本进程——与基线逐位相同。）

### 5.2 构图宽度：为什么 `intermediate` 必须保持 2×degree

250K 规模、4 份代理数据（128/256 维 × 均匀/带簇）的完整矩阵在
`evidence/recall_matrix.txt`，`evidence/sweep7_256d.txt` 是 256 维那轮的原始 sweep。要点：

| 数据集（250K 行） | `inter=256`（基线）recall@100 | `inter=128`（=输出度）recall@100 | 差值 |
|---|---|---|---|
| 128 维 · 均匀 | 0.9993 | 0.9975 | −0.0018 |
| 128 维 · 带簇 | 1.0000 | **0.8900** | **−0.1100** |
| 256 维 · 均匀 | 0.9804 | 0.9688 | −0.0116 |
| 256 维 · 带簇 | 1.0000 | **0.9200** | **−0.0800** |

带簇分布（更接近真实 embedding）上把构图宽度砍到 1× 会掉 0.08～0.11，远超 ±0.005。
原因不是细化轮次不够，而是 **cuVS 的剪枝阶段本身在改善图的可导航性**：构到两倍宽再剪，
剪掉的正是那些「太长、会把贪心搜索带偏」的边。降输出度同样危险：
`degree=96/64` 的组合在均匀 128 维上只有 0.9690 / 0.9265。

### 5.3 细化轮次 `niter`：80 与 64 都越界，只能保持 100

同一份 250k×256 表上的参赛版各 `niter` 臂；召回一律对同一份精确 GT 复算，基线召回取
`ab_nn.log` 轮 2 的 50 547 ms / 1.0000 / 0.9799（`niter=80` 臂与它真正同轮，
`n256i100`/`n256i64` 两臂来自 `niter_sweep.log`，与基线跨时段，加速比只作参照）：

| 臂 | niter | 窗口 (ms) | 加速 | recall@10 | recall@100 | Δ@10 | Δ@100 | 判定 |
|---|---|---|---|---|---|---|---|---|
| `n256i100` | 100 | 48 198 | 1.049x（参照） | 1.0000 | 0.9795 | 0.0000 | −0.0004 | ✅ 采用 |
| `nn_cand` | 80 | 40 863 | 1.237x（同轮） | 0.9950 | 0.9753 | **−0.0050** | −0.0046 | ❌ 零余量 |
| `n256i64` | 64 | 35 899 | 1.409x（参照） | 0.9950 | 0.9706 | −0.0093 | −0.0093 | ❌ 越界 |

`niter=80` 那一行是本轮最重要的负面结论：代理 cuVS 层 sweep 上它看起来是安全的
（均匀 128 维 −0.0003、带簇 256 维 −0.0000），但在**真实 seekdb 表 + 端到端口径**上
recall@10 恰好掉 0.0050，等于把 ±0.005 的门槛额度一次性用光。
隐藏集不是我们的代理集，这种配置不能提测。

同一份数据在 128 维上的同轮对照（45 663 / 45 212 ms）两侧 recall 都是 1.0000，
`@100` 差 0.0004——与 256 维 `niter=100` 臂一致，说明差距来自参数而不是链路。

关于抖动（这次用 4 轮同轮重复把它量下来了，见 1.1 主口径表）：两个二进制跑的是**同一套构图参数**，
每轮各自建一次图，于是同一个配置在不同轮里给出的图并不逐位相同。100 条查询下，
少命中 1 个邻居只移动平均值 0.001，因此：

- 参赛臂对**自己同轮基线**的差：`@10` 四轮分别是 +0.0010 / −0.0010 / +0.0010 / −0.0010，
  `@100` 是 +0.0014 / +0.0007 / −0.0010 / +0.0010 —— 全部在 ±0.0015 以内，离 0.005 的门槛
  还有 3 倍以上的余量。
- 基线臂**跨轮自己**的散布也不小：`@10` 取过 0.9980 / 0.9990 / 1.0000，`@100` 取过
  0.9782 / 0.9794 / 0.9815（跨轮极差 ±0.0017）。所以拿某一轮的基线召回当绝对基准本身就是
  一个 ±0.002 量级的读数，评测机上「同轮基线」才是唯一裁判（第 8 节）。
- 结论：召回这条门槛的余量不是靠「我们比基线更准」挣来的，而是靠**参赛默认值和基线同图**
  挣来的——两边的召回是同一个分布的两个样本，差值只会落在噪声带里。

### 5.4 窗口拆分（observer 打点）

参赛二进制自带分段计时，直接读日志就能把「构图 / 导出+落盘 / 其他」分开：

| 运行 | `build_us` | `total_us` | 导出+快照落盘 | DDL 墙钟 | 表扫描+SQL+其他 |
|---|---|---|---|---|---|
| 256 维 认证运行 | 43 351 029 | 43 663 360 | **0.31 s** | 48.36 s | **4.70 s** |
| 256 维 `n256i100` | 43 029 175 | 43 427 406 | 0.40 s | 48.13 s | 4.70 s |
| 128 维 `c128i100` | 41 844 214 | 42 052 322 | 0.21 s | 45.05 s | 3.00 s |
| 128 维基线 `b128base` | 42 007 276 | （无 total_us，走 tmpfile 往返） | — | 45.61 s | 3.60 s |

即：**窗口里 87～92% 是 `cuvsCagraBuild` 本身**，快照那一段只剩 0.2～0.4 s，
剩下 3～4.7 s 是表扫描 + SQL 层 + 索引切换，不属于向量索引构建可压的部分。
可选的精确图路径（`algo=3`）在同一条打点上 `build_us` 只有约 20 s（`b2.txt`：
`upload=123 build=20070 d2h=82 prune=77 import=19 total=20374` ms），这就是 1.98x 的来源，
代价是 1.438 GB 峰值。

### 5.5 CPU（`lib=vsag` HNSW）vs GPU（`lib=cuvs` CAGRA）

同一实例、同一份 250k×128 表、同一棵二进制树（参赛二进制，只是 DDL 里换 `lib=`）、
同一组 DDL 参数（`m=16, ef_construction=200, ef_search=64`）、同一 100 条查询：

| 路径 | DDL (ms) | 首查 (ms) | 窗口 (ms) | recall@10 / @100 | QPS@10 / @100 |
|---|---|---|---|---|---|
| CPU HNSW（`lib=vsag`，`evidence/cpu_vsag.log`） | 156 994 | 169 | **157 163** | 0.4680 / 0.3277 | 175.5 / 121.1 |
| GPU CAGRA（`lib=cuvs`，同表同轮基线 `b128base`，`evidence/ab128.log`） | 45 609 | 54 | **45 663** | 1.0000 / 0.9997 | 73.4 / 62.4 |
| **GPU / CPU** | | | **构建窗口 3.44x 更快** | **recall@10 +0.532、@100 +0.672** | CPU 单查更快（见下） |

两点如实说明：

- CPU 侧召回低不是 VSAG 不行，而是这条 DDL 把 `ef_search=64` 同时交给了两条路径：
  VSAG 拿它当搜索 `ef`，对 top-100 来说远远不够；CAGRA 侧 `ef_search` 不建图参数，
  搜索候选池走 bridge 的 `itopk = max(topk*4, 512)`（且 hipVS 硬上限 512，见 3.1）。
  所以「同参 DDL」下 GPU 召回高、CPU 单查快（CPU 每查询只扫 64 个候选，GPU 扫 512 个），
  两者是同一个搜索代价旋钮的两端，不是免费的午餐。
- 本方案的查询门槛比较对象是**同轮的 GPU 基线**，不是 CPU（见 1.1 的 QPS 列），
  这一节的 CPU 数字是赛题要求的 CPU/GPU 构建对照。

### 5.6 显存（第 4 条硬门槛）

窗口内本进程所在卡的峰值（`evidence/*.vram` 离线按 `.marks` 复算）：

| 配置 | 256 维峰值 | 相对基线 | 128 维峰值 | 判定 |
|---|---|---|---|---|
| 基线 `634888d` | 0.803 GB | 1.00x | 0.684 GB | — |
| 参赛默认（nn_descent，同图） | 0.803 GB | **1.00x** | 0.684 GB | ✅ 远好于 ≤1.25x，且 ≪45 GB |
| `algo=3` 精确 all-neighbors | 1.438 GB | **1.79x** | 未单独采样 | ❌ 256 维越界，故只做显式开关 |
| IVF-Flat 精确 kNN | 1.940 GB | 2.42x | — | ❌ 越界且更慢 |
| all-neighbors + `n_clusters=8/32` | 0.605 / 0.785 GB | 0.75x / 0.98x | — | 显存过，但召回余量被吃光（见 1.3） |

同轮重复的四轮（1.1 主口径表）逐轮复核了这条：参赛臂 862 367 744 / 862 384 128 / 862 392 320 /
862 392 320 B，基线臂 862 412 800 / 862 388 224 / 862 392 320 / 862 412 800 B
——**参赛臂在每一轮都不高于基线**（比值 ≤1.0000），两边都是 0.8032 GB。
这四轮的 `.vram` 里混进了一条作废轮次未清理的采样进程（第 2 轮 `FAIL start` 后采样器没被杀掉），
所以严格 `ddl_start..ddl_end` 窗口内的峰值只有 0.624～0.803 GB、随采样对齐抖动；
整臂峰值不受影响，干净窗口以 `evidence/vram_peaks.txt` 的 `ship_base` / `nn_base` / `n256i100` /
`final_ship` 四条为准（都是 0.803 GB，重复轮次与之一致）。

### 5.7 已否决的替代方案汇总

| 方案 | 结果 | 否决原因 |
|---|---|---|
| `intermediate == degree`（省掉剪枝） | cuVS 层 2.28x | 带簇数据 recall@100 −0.08～−0.11 |
| `niter=80 / 64` | 1.24x / 1.41x | 端到端 recall@10 −0.0050 / −0.0093 |
| `degree=96 / 64` | 8.6～11.4 s（cuVS 层） | 均匀 128 维 recall@100 0.9690 / 0.9265 |
| `algo=3` 精确 all-neighbors | **1.98x，召回还更高** | 显存 1.79x > 1.25x 门槛 |
| IVF-Flat 自搜索求精确 kNN | 55.4～61.6 s | 比 nn_descent 更慢，显存 2.42x |
| all-neighbors 分簇（4/8/16/32/64） | 34.7～88.5 s | 显存达标但不再精确，recall@100 只剩 0.9758～0.9774 |
| fp16 底库构图 | 20 433 ms vs 20 374 ms | 无收益 |
| 提前释放设备端底库 | 搜索 `rc=15` | CAGRA 图搜索引用底库向量，省不掉 |
| `cuvsBruteForce*` | 符号未从 `libcuvs_c.so` 导出 | 头文件有声明但链接不到，这条路在本版 hipVS 上不存在 |
| 提高搜索 `itopk` 补召回 | `rc=15` | hipVS 硬限制 `itopk_size ≤ 512` |

---

## 6. 正确性验证

| 验证 | 做法 | 状态 |
|---|---|---|
| bridge 能编译、能链接 hipVS | AMD 平台上用官方命令 `HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm tools/hipvs_bridge/build.sh` 实编译 | ✅ 通过 |
| 交付件与实测二进制同源 | 仓库文件 md5 `40da67e51fb7c394b60b9c953bbd9250`；实例上按此文件重建出 `libseekdb_cuvs_bridge.so`（md5 `fb0c860c969df9cd9082477b610e7cae`），1.1 的认证运行与 128 维 A/B 都用这份 `.so`；`evidence/ship_bridge/` 留了实例上的副本与逐字节比对（仅注释里一个 `±` 字符差异） | ✅ 通过 |
| bridge 符号齐全 | `nm -D` 检查导出符号 + `RTLD_NOW` dlopen 自检 | ✅ 通过 |
| DDL 契约（基线自带 mysqltest） | `evidence/cpu_vsag.log` 用参赛二进制逐条重跑 `vector_index_cuvs_ddl.test` 的 6 条语句：`lib=cuvs + l2` 接受，`cosine / inner_product / ivf_flat` 各报 `1235`，`lib=vsag + cosine / inner_product` 照旧接受 | ✅ 与基线期望完全一致 |
| 快照落盘 + 重启恢复 | `evidence/final_ship.log`：建索引 → 查询 → `kill -TERM` → 同一 data-dir 重启 → 不重新建索引（日志里 `built GPU CAGRA index` 0 次、`deserialized GPU CAGRA snapshot(n64=250000, dim32=256, degree32=128, graph_bytes=128000000)` 1 次），重启后 100 条查询的 top-100 / top-10 结果与重启前**逐行相同**，recall 同为 0.9970 / 0.9795 | ✅ 通过 |
| 端到端功能（250K 规模、真实 DDL、真实 SQL 查询） | 上述全部 A/B 轮次 | ✅ 通过 |
| `lib=vsag` 路径无回归 | 同一份 250k×128 表上跑通 CPU HNSW 建索引 + 查询（`evidence/cpu_vsag.log`） | ✅ 通过 |
| adaptor 改动可编译 | 改动区域抽出来配桩头文件做 `-fsyntax-only` 类型检查（`evidence/syntax_check/`），g++ 15.2 | ✅ 通过 |
| 评测机用旧桥接也能跑 | adaptor 不再直接引用 `seekdb_cuvs_export_graph / _import / _free_host`，改成启动时 `dlsym` 解析；解析不到就退回基线的 v2 快照（cuVS 序列化文件 + tmpfile）。`nm` 确认改动区域只剩 `build/search/free/serialize/deserialize` 四个基线就有的符号 | ✅ 通过 |
| 评测机 hipVS 没有 all-neighbors | 桥接里 `<cuvs/neighbors/all_neighbors.h>` 改成 `__has_include` 护栏，头文件缺失时 `algo=3` 编不进、自动回落 nn_descent；出厂默认本来就是 nn_descent，实测口径不受影响 | ✅ 预处理双分支验证 |
| `OB_BUILD_CUVS=ON` 但没传 `CUVS_BRIDGE_LIB` | `cmake/Env.cmake` 依次探测 `/work/bridge`、`tools/hipvs_bridge`，再不行就现场调用 `build.sh` 造一个；都拿不到时把 GPU 后端关掉（而不是留一个必挂的链接） | ✅ cmake 五种分支实测 |

---

## 7. 评分推算（按本报告的实测数字）

只用**同轮**比值，跨轮参照不参与推算。

- 构建加速：`r_hidden = 1.0147`（1.1 主口径表四轮均值）、`r_sift = 1.008`（128 维两轮同轮
  1.010 / 1.006 的均值）→
  `composite = 1.008^0.4 × 1.0147^0.6 ≈ 1.0120` → `build = 14 × ln(1.0120)/ln(3.0) ≈ 0.15 分`。
  把这四轮的散布当成不确定度：最差一轮 `r_hidden = 0.996` → composite ≈ 1.001 → 约 0.01 分；
  最好一轮 `r_hidden = 1.053` → composite ≈ 1.035 → 约 0.44 分。**期望 0.15 分、区间 0～0.5 分**。
- 查询吞吐：256 维四轮 QPS 几何比均值 1.021（最差 0.990）、128 维 1.001 →
  复合 ≈1.013 → `query = 2 × ln(1.013)/ln(1.5) ≈ 0.06 分`（门槛 0.90 有 8% 以上余量）。
- 四条硬门槛：召回（每一轮 |Δ| ≤0.0015，门槛 0.005）、QPS（最差轮 0.990，门槛 0.90）、
  显存（逐轮 ≤1.000x，门槛 1.25x）、单轮时间（GPU 臂 ≤ 5 min，门槛 15 min）→ `4 分`。
- **合计 ≈ 4.2 分**（区间 4.0～4.5）。

这就是「把图做成和基线一样」这条路的真实分数上限，也是我们把数字写小而不是写大的原因。
对照：如果默认取 `algo=3`（精确 all-neighbors，同轮 1.981x、召回还更高），构建分是
`14 × ln(1.98)/ln(3) ≈ 8.7 分`、合计 ≈ 12.8 分——但显存门槛 1.79x 一失败就是 **0 分**。
取 `niter=80` 是约 2.9 分的构建分（同轮 1.237x），同样在召回门槛上零余量、一失败就 0。
在「门槛失败 = 0 分」的规则下，本方案选的是**余量为正的那一点**，并把 1.98x 的路径
完整保留在 `SEEKDB_CUVS_BUILD_ALGO=3` 上，等显存口径放宽（1.25x 是唯一挡住它的门槛，
绝对量 1.44 GB 相对 45 GB 上限毫无压力）即可直接启用。

---

## 8. 局限

- 只支持 `distance=l2`（cuVS CAGRA 侧的限制，DDL 层已拦 `lib=cuvs` + 非 L2）。
- 单卡；底库必须驻留显存（`n×dim×4` 底库 + 图 `n×degree×4` + 搜索缓冲）。
  250k×256 实测峰值 0.803 GB，距离 45 GB 上限很远，但方案本身不做分片，规模再大要另配
  多卡/分片策略。
- 快照体积随 `degree` 线性增长：`degree=128`、`n=250K` 时邻接表 128 MB。
- 构建旋钮是进程级环境变量，不是 per-index 的 DDL 参数；多租户场景要固化默认值或改成
  per-table 配置。当前默认值已经固化成「与基线同图」，所以不设变量就是安全行为。
- 共享实例的绝对耗时受节流影响，跨时段不可比；本报告的加速比全部来自同轮相邻运行。
  256 维的参赛默认值同轮重复有 4 轮有效（另有 1 轮因基线 observer 启动失败作废），
  128 维的同轮重复只来得及 2 轮（实例到 4 小时销毁时间，后面排队的轮次没跑完），
  所以 1.015x 这个均值的置信度有限——第 7 节把它当期望值并同时给出 0～0.5 分的区间，
  而不是当成分数本身。
- **本方案的构建收益天生就小**：默认参数刻意与基线同图，窗口里 87～92% 的时间是两边都要跑的
  `cuvsCagraBuild`。真正能把构建分做大的三条路（精确 kNN 图、更少的细化轮次、更窄的构图宽度）
  各撞一条硬门槛，这是这版 hipVS + 这套门槛下的实际形状，不是实现没做到位。
- 召回余量取决于数据分布。`@10` 在 100 条查询上有 ±0.003 量级的轮间抖动（5.3），
  这个底噪对所有配置一致；**评测机上的同轮基线召回才是唯一裁判**。
