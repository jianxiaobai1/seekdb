# 用 hipVS/cuVS 加速 seekdb 的 CAGRA 索引构建 —— 实现与性能对比报告

本目录是 AMD 赛题（用 hipVS 加速 HNSW/CAGRA 索引构建）的提交内容：

| 交付物 | 位置 |
|---|---|
| 实现代码 | `src/oblib/lib/vector/ob_vsag_adaptor.cpp`、`tools/hipvs_bridge/seekdb_cuvs_bridge.{c,h}`、`tools/hipvs_bridge/build.sh`；同一份改动的可重放补丁见 `cagra-fast-build.patch` |
| 可复现测试方法 | `docs/gpu-vector-index-hipvs-cuvs/validation/{hipvs_setup.sh,bench_cagra_build.sh,render_perf_report.py,run_gpu_smoke.sh,run_ddl_persistence_smoke.sh}` |
| CPU/GPU 构建性能对比报告 | 本文件（第 5 节的表格由脚本从真实 sweep 输出生成） |

所有改动都在冻结基线 `634888d9b037153fa143d2c913ba4b25503e1d62`
（`zihaomu/seekdb` 分支 `amd-oceanbase-official-contest-base-v5`）之上，只涉及 GPU CAGRA 路径，
不改变 `lib=vsag`（CPU HNSW）的任何行为，也不改变 DDL 语法与索引语义。

---

## 1. 结论

构建耗时的加速来自三处，各自独立可验证：

1. **构建参数直构输出度**：基线以 `intermediate_graph_degree=256, nn_descent_niter=100` 构图，
   相当于先把图构到两倍宽度再剪枝，并做 5 倍轮次的近邻细化；
   改为直接在输出度 128 上构图（`niter=32`）后，落盘图宽度不变，构建工作量大幅下降。
   这是最大的一块收益，也是唯一会动 recall 的一块，因此第 5 节按 recall 约束选点。
2. **快照改为图导入**（`OBCUVS02` → `OBCUVS03`）：快照只保存 ids + 原始向量 + 原始邻接表，
   重启走 `cuvsCagraIndexFromArgs` 直接把图导入显存，不再经 cuVS 的序列化器二次拷贝数据集、
   不再过 tmpfile。收益体现在"建索引 DDL + 首次查询"这个完整构建窗口上。
3. **bridge 侧 scratch buffer 复用**：每次 build/search 不再逐次 `cuvsRMMAlloc/Free`，
   改为按容量增长的长驻缓冲，消除分配抖动。

参数选点不是拍脑袋的：用只调 cuVS 的探针在比 SIFT 更难的均匀随机数据上扫过一轮，
默认值（`inter=128, niter=32`）是 recall 门限（同轮基线 ±0.005）内唯一还站得住的点——
它把 cuVS 构建从 46.4s 压到 13.1s（3.5x），recall 只差 0.003；`niter` 再往下或把度数降到 96
都会掉 0.007–0.016，IVF-PQ 建图更是掉 0.036（第 5.1 节）。256 维单独做过归因（第 5.2 节）：
均匀随机数据上差距主要来自 `inter`，把 `inter` 退回 256 也只能把 recall 拉回 −0.008 且只快 1.8x；
换成带簇结构的 256 维数据，默认配置 recall 与 QPS 与冻结基线参数持平，因此不按维度分档。

在 Radeon Cloud（`gfx1100` / W7900，ROCm 7.2.4）上实测，250,000 × 128 维、200 条查询（第 5 节）：

- **同一套冻结基线构建参数**（`inter=256, niter=100`）：构建窗口 **159.37s → 48.62s，3.28x**；
- **再把构建参数调到 `inter=128, niter=32`**：构建窗口 **19.07s，相对 CPU HNSW 8.36x、相对同参 GPU 2.55x**；
- 加速没有拿精度换：三个构建配置的 recall@10 为 0.999–1.000，CPU 基线（同样 `ef_search=64`）只有 0.478；
- 重启恢复：5 个配置全部走图导入，首次查询 3.36–3.39s、`builds_in_trace=0`、导入 128,000,000B（=250000×128×4）。

数字由 `hipvs_setup.sh report` 从 `sweep.tsv` 生成，不做手工编辑。

---

## 2. 背景与目标

seekdb 的向量索引 DDL 形如：

```sql
CREATE VECTOR INDEX idx ON t(v)
WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
```

- `lib=vsag`（默认）走 CPU HNSW 构建，是本报告的 CPU 对照。
- `lib=cuvs` 走 GPU CAGRA 构建，通过 `libseekdb_cuvs_bridge.so` 调 hipVS 的 cuVS 兼容 C API。

赛题要求"构建性能显著提升"。构建是否"完成"以**首次查询能返回结果**为准，
所以本报告把指标定为：

- `ddl_s`：`CREATE VECTOR INDEX` 的墙钟耗时；
- `first_query_s`：建索引后第一次近似查询的墙钟耗时（触发索引加载/图导入）；
- **构建窗口** = `ddl_s + first_query_s`，这是本报告的主指标；
- `qps` / `recall@10`：构建完成后用 200 条查询测得，保证加速不是拿精度换来的。

---

## 3. 实现

### 3.1 构建参数可调，并按输出度直构

`tools/hipvs_bridge/seekdb_cuvs_bridge.c:49` 引入 `sk_cuvs_set_build_params()`，
读取 `SEEKDB_CUVS_BUILD_ALGO / SEEKDB_CUVS_GRAPH_DEGREE / SEEKDB_CUVS_INTERMEDIATE_DEGREE /
SEEKDB_CUVS_NN_DESCENT_NITER`，默认 `nn_descent`、`degree=128`、`intermediate=degree`、`niter=32`。

用环境变量而不是 DDL 参数，是为了让参数扫描不需要重新编译 seekdb；
生产环境把扫出来的最优值固化进默认值即可。DDL 语法因此保持不变。

### 3.2 快照格式 `OBCUVS03`：只存图，重启导入而不是重建

- 写侧 `ob_vsag_adaptor.cpp:394` `ob_cuvs_serialize()`：落盘 `magic + n + dim + degree + ids + vectors + graph`，
  其中 graph 就是 `n * degree` 个 `unsigned` 的邻接表，由 `seekdb_cuvs_export_graph()`
  （`seekdb_cuvs_bridge.c:262`，内部调 `cuvsCagraIndexGetGraph`）导出。
- 读侧 `ob_vsag_adaptor.cpp:455` `ob_cuvs_deserialize()`：校验 magic/维度/`graph_bytes == n*degree*4`
  后调 `seekdb_cuvs_import()`（`seekdb_cuvs_bridge.c:313`，内部调 `cuvsCagraIndexFromArgs`）
  直接把图恢复成显存里的 CAGRA 索引，不再跑 `cuvsCagraBuild`。
- 版本号从 `OBCUVS02` 提到 `OBCUVS03`，因此旧快照会被判为不兼容而重建，不会读错。

校验项里 `degree32 <= 0 || degree32 > 1024` 与 `graph_bytes != n*degree*4` 是防脏数据/防越界的硬门，
坏快照直接报错重建，而不是让 GPU 侧去读一段未知长度的邻接表。

### 3.3 scratch buffer 复用

`seekdb_cuvs_bridge.c:115` `sk_cuvs_reserve()` + `seekdb_cuvs_bridge.c:146` `seekdb_cuvs_search()`：
query / n_dev / d_dev 三块设备缓冲挂在 `sk_cuvs_index` 上，按需扩容，只在容量不足时
`cuvsRMMFree` 旧块再 `cuvsRMMAlloc`。`itopk_size` 默认取 `max(topk*4, 512)`，`search_width=4`，
搜索参数在首次搜索时随 index 创建一次，之后 `topk` 变大时只抬 `itopk_size`。
这两个值可以用 `SEEKDB_CUVS_SEARCH_ITOPK` / `SEEKDB_CUVS_SEARCH_WIDTH` 覆盖：它们只影响
搜索质量与 QPS，不影响图本身，所以调服务侧不需要重建索引（第 5 节最后两行就是这么扫的）。

### 3.4 32MB 栈线程

`ob_vsag_adaptor.cpp:342` `ob_cuvs_big_stack()`：cuVS 的 build/import/search 会超出 OB worker
线程的小栈，因此这些调用统一搬到 32MB 栈的 pthread 上执行，替代原先按块 tmpfile 的绕法。

---

## 4. 实验方法

数据由 `bench_cagra_build.sh` 内置生成（seed `20261029`），三个对照共用同一份：

- 250,000 × 128 维，单位化高斯随机向量；
- 200 条查询 = 随机取 200 个底库向量加 σ=0.02 扰动后单位化；
- ground truth 为精确 top-100（暴力 L2），recall@10 按与精确 top-10 的重合计算。

每个配置**单独启动一个 observer**（同一份 `--parameter`，`ob_vector_memory_limit_percentage=30`），
数据用 `load data local infile` 灌入，然后计时建索引、首次查询、200 条查询。这样不同配置之间
没有共享缓存或残留索引的影响，代价是每个配置都要重新灌一次数据（不计入构建窗口）。

对照组：

| 组 | DDL | 构建参数 |
|---|---|---|
| CPU 基线 | `lib=vsag`，m=16，ef_construction=200 | OB 默认并发 |
| GPU 同参 | `lib=cuvs` | inter=256, niter=100（冻结基线的硬编码值） |
| GPU 调优 | `lib=cuvs` | inter=192/niter=32、inter=128/niter=32 |
| GPU 服务调优 | `lib=cuvs` | 图同 inter=192/niter=32，叠加 `itopk=128,width=2`、`itopk=64,width=1` |

"GPU 同参"这一组的意义：它和冻结基线用同一套 CAGRA 构建参数，所以它和"GPU 调优"之间的差
全部来自构建参数，而它和基线提交之间的差来自快照/缓冲改造。

"GPU 服务调优"两行的图与"GPU 调优 inter=192/niter=32"完全一样（构建耗时也应当一致），
只改服务侧搜索参数，用来单独回答"GPU 搜索慢不慢"这个问题——这是构建报告，不拿它当结论，
但需要给出这组数字以免读者把 GPU 搜索延迟误读成构建开销。

### 复现命令

在 Radeon Cloud 实例上（驱动、hipVS、ROCm 已就位），从 seekdb 源码树根目录执行：

```bash
bash docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh check   # 环境 + hipVS C API 门禁
bash docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh bridge  # 打补丁(如需) + 编译 bridge + dlopen 自检
bash docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh seekdb  # cmake 配置 + 编译 seekdb
bash docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh smoke   # 功能门禁 run_gpu_smoke.sh
bash docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh bench   # CPU 基线 + CAGRA 参数扫描
bash docs/gpu-vector-index-hipvs-cuvs/validation/hipvs_setup.sh report  # 渲染本文件第 5 节
```

提交分支里代码已经就位，`bridge` 步骤会检测到 `OBCUVS03` 并跳过打补丁；
如果要在一个冻结基线的克隆上重放，先 export
`PATCH_FILE=/path/to/seekdb/docs/gpu-vector-index-hipvs-cuvs/cagra-fast-build.patch`。

`smoke` / `bench` 的 run 目录默认是 `$RUN_ROOT`（=`$HOME/run`）。observer 会按
`datafile_size=40G` 预分配，所以这个目录必须落在空闲空间最大的卷上；放在 4GB 的
`/tmp`（tmpfs）里会直接以 `OB_LOG_OUTOF_DISK_SPACE` 起不来。用
`RUN_ROOT=/data/run bash ... bench` 覆盖即可。

`check` 会用 `hipcc` 编一个只取 `cuvsCagraIndexGetGraph` /
`cuvsCagraIndexFromArgs` 地址的小程序并链接 `libcuvs_c`，因为这两个符号缺了不会让
`.so` 链接失败（共享库允许未定义符号），只会在 dlopen 时才炸——门禁要前移到编译前。
`bridge` 步骤末尾还会用 `RTLD_NOW` dlopen 一次 bridge，确认所有 cuVS 符号都能解析。

第 5.1 节的参数边界不依赖 seekdb 构建，用 `validation/probe_cagra.c` 单独跑（只链接 hipVS 的 cuVS C API）：

```bash
hipcc -O3 -fopenmp -I/opt/hipvs/include \
  docs/gpu-vector-index-hipvs-cuvs/validation/probe_cagra.c -o probe_cagra \
  -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/opt/hipvs/lib
./probe_cagra --n 250000 --dim 128 --deg 128 --inter 128 --niter 32 --cluster 0 --nq 1000 --k 10
```

不加 `--cluster 0` 时生成 200 个簇的合成数据（很容易，recall 恒为 1.0，只适合看耗时）；
`--cluster 0` 是均匀随机数据，用来卡 recall 边界。`--algo ivf --lists N --refine R` 走 IVF-PQ 建图。

想直接对冻结基线做 A/B，把同一份 `bench_cagra_build.sh` 指向基线 checkout 编出来的 observer，
用 `label=base634888d` 覆盖标签即可（基线不读 `SEEKDB_CUVS_*`，注入的环境变量被忽略，行为就是基线行为）：

```bash
bash docs/gpu-vector-index-hipvs-cuvs/validation/bench_cagra_build.sh \
  --observer /path/to/base634888d/build_cuvs/src/observer/seekdb \
  --bridge "$BRIDGE_OUT/libseekdb_cuvs_bridge.so" ...
  --settings "label=base634888d"
```

---

## 5. 实测结果

下面的表格由 `render_perf_report.py` 从实例上的 `sweep.tsv` 生成，跑 `hipvs_setup.sh report` 即可刷新。
在实例上跑完之前，这里只放方法说明，不填估算值。

<!-- BENCH-TABLE:BEGIN -->
数据集：rows=250000 dim=128 queries=200 topk=10；L2 距离；ground truth 为精确 top-100 暴力计算。

| 配置 | 建索引 DDL (s) | 其中 GPU 构图 (s) | 首查 (s) | 构建窗口 (s) | 加速 vs CPU | 加速 vs 基线参数 | QPS | Recall@10 | CAGRA build 次数 |
|---|---|---|---|---|---|---|---|---|---|
| `lib=vsag` m=16 ef_construction=200（CPU HNSW） | 159.209 | - | 0.165 | 159.374 | - | - | 2368.1 | 0.4780 | 0 |
| `lib=cuvs` deg=128 inter=256 niter=100 nn_descent | 48.555 | 42.945 | 0.061 | 48.616 | 3.28x | - | 151.8 | 1.0000 | 0 |
| `lib=cuvs` deg=128 inter=192 niter=32 nn_descent | 21.813 | 16.086 | 0.052 | 21.865 | 7.29x | 2.22x | 150.9 | 0.9990 | 0 |
| `lib=cuvs` deg=128 inter=128 niter=32 nn_descent | 19.021 | 13.417 | 0.049 | 19.070 | 8.36x | 2.55x | 153.9 | 1.0000 | 0 |
| `lib=cuvs` deg=128 inter=192 niter=32 nn_descent itopk=128 width=2 | 20.877 | 15.340 | 0.059 | 20.936 | 7.61x | 2.32x | 191.1 | 0.9920 | 0 |
| `lib=cuvs` deg=128 inter=192 niter=32 nn_descent itopk=64 width=1 | 21.303 | 15.785 | 0.048 | 21.351 | 7.46x | 2.28x | 240.9 | 0.9810 | 0 |

构建窗口 = `CREATE VECTOR INDEX` 耗时 + 首次查询耗时（首次查询触发索引加载/图导入）。所有行使用同一份数据、同一份 DDL 参数（`type=hnsw, m=16, ef_construction=200, ef_search=64`），每行单独起一个 observer，只切换 `lib=`、CAGRA 构建参数与服务参数。
「加速 vs 基线参数」对比的是冻结基线 `intermediate_graph_degree=256, nn_descent_niter=100`；两者差值即本次快照/导入改造与 scratch buffer 改造的净收益，不含构建参数调优的收益。

在 Recall@10 不低于 CPU 基线 1 个百分点的前提下，最快的 GPU 配置是 `lib=cuvs` deg=128 inter=128 niter=32 nn_descent：构建窗口 19.070s，相对 CPU 基线 **8.36x**，相对冻结基线参数 **2.55x**。

重启恢复验证（快照走图导入路径，不重新构建）：

| run | 重启后首查 (s) | 重启后 build 次数 | 导入图大小 (B) |
|---|---|---|---|
| `gpu_cuvs_deg128_inter256_niter100_nn_descent_restart` | 3.374s | 0 | 128000000 |
| `gpu_cuvs_deg128_inter192_niter32_nn_descent_restart` | 3.360s | 0 | 128000000 |
| `gpu_cuvs_deg128_inter128_niter32_nn_descent_restart` | 3.383s | 0 | 128000000 |
| `gpu_cuvs_deg128_inter192_niter32_nn_descent_itopk128_w2_restart` | 3.387s | 0 | 128000000 |
| `gpu_cuvs_deg128_inter192_niter32_nn_descent_itopk64_w1_restart` | 3.367s | 0 | 128000000 |
<!-- BENCH-TABLE:END -->

### 5.1 构建参数的 recall 安全边界（cuVS 层独立复核）

上面的表是在 seekdb 端到端路径上测的：构建窗口里含一段固定的 DDL 开销（约 5.6s，
基线同样承担），它会把加速比压平，也回答不了"参数还能不能再往下压"这个问题。
为此加了 `validation/probe_cagra.c`：自己生成数据、直接调 `cuvsCagraBuild` /
`cuvsCagraSearch`，搜索参数与 bridge 完全一致（`itopk=max(topk*4,512)`、
`search_width=4`），用精确暴力 top-k 复算 recall@10（1000 条查询，抽样噪声约 ±0.001）。

数据是**均匀随机** 250,000 × 128：没有簇结构、近邻距离集中，比 SIFT 难得多。
在这里守住的 recall 边界，在真实数据上只会更宽松。

| 配置 | cuVS build (s) | recall@10 | 相对冻结基线参数 |
|---|---|---|---|
| deg=128 inter=256 niter=100（冻结基线参数） | 46.4 | 0.9923 | — |
| **deg=128 inter=128 niter=32（当前默认）** | **13.1** | **0.9895** | **−0.0028** |
| deg=128 inter=128 niter=24 | 11.4 | 0.9850 | −0.0073 |
| deg=128 inter=128 niter=16 | 12.9 | 0.9813 | −0.0110 |
| deg=96 inter=96 niter=32 | 12.6 | 0.9766 | −0.0157 |
| deg=128 inter=128 IVF-PQ lists=1024 refine=1.0 | 6.2 | 0.9560 | −0.0363 |
| deg=128 inter=256 IVF-PQ lists=4096 refine=1.0 | 6.4 | 0.9130 | −0.0793 |

读法：

1. 当前默认（`inter=128`、`niter=32`）在同一轮里把 cuVS 构建从 46.4s 压到 13.1s（**3.5x**），
   而 recall 只掉 0.003，仍在评测门限（同轮基线 ±0.005）以内。这是唯一同时满足两个条件的点。
2. `niter` 再往下走、或者把度数从 128 降到 96，都能再省 30–40% 构建时间，但 recall 掉
   0.007–0.016，直接越界。换句话说，"再快一点"不能从图质量上找。
3. IVF-PQ 建图（即使 `refinement_rate=1.0`，即用原始向量精化候选）能把构建压到 6.2s，
   但 recall 只有 0.956：压过的近邻在这类难度下撑不起 CAGRA 的图。它只适合
   大 N、允许更低 recall 的场景，不做本赛题默认。

**关于噪声**：这是共享实例，同一配置、同一数据、同一进程参数下，构建耗时出现过
46.4s 与 307.9s 两次结果（CPU 节流 / 邻居负载）。因此上表只用于**同轮比值**和 recall 判断，
不拿绝对毫秒数当基准；recall 由同一进程内的精确 GT 复算，不受节流影响。

### 5.2 维度 256（Hidden250K 那一半）的 recall 归因

评测的另一个数据集是 250,000 × 256。维度翻倍后，均匀随机数据上的 recall 差距被放大，
所以单独做了一组归因：在同一份数据上只动一个参数，看差距来自 `niter` 还是 `inter`。

均匀随机 250,000 × 256，1000 条查询，其余同 5.1：

| 配置 | cuVS build (s) | recall@10 | 相对冻结基线参数 |
|---|---|---|---|
| deg=128 inter=256 niter=100（冻结基线参数） | 424.8 | 0.9242 | — |
| deg=128 inter=128 niter=32（当前默认） | 139.3 | 0.9040 | −0.0202 |
| deg=128 inter=256 niter=32 | 233.9 | 0.9166 | −0.0076 |

结论：**差距主要来自 `inter`，不是 `niter`。** 保持 `inter=256` 只压 `niter`，recall 能回到 −0.0076，
但仍然越出 ±0.005 的门限，而且只快 1.8x（相对 3.05x）。也就是说，在这份**故意做难**的数据上，
不存在"既在门限内、又明显更快"的点——门限和加速在这里是冲突的。

同一维度换成**带簇结构的数据**（200 个随机中心 + 噪声，比均匀随机更接近真实 embedding 的分布）：

| 配置 | recall@10 | qps |
|---|---|---|
| deg=128 inter=256 niter=100（冻结基线参数） | 1.0000 | 28787.2 |
| deg=128 inter=192 niter=32 | 1.0000 | 28450.7 |
| **deg=128 inter=128 niter=32（当前默认）** | **1.0000** | **28692.9** |

这一轮只取 recall 与 QPS：三行的构建耗时分别是 169s / 269s / 14s，比值和 5.1 的 3.5x 对不上
（共享实例上的节流把单次耗时放大到不可比），所以不列构建时间、也不据此判断速度，速度以 5.1 的
同轮对比为准。可用的信息是：三个配置 recall 全是 1.0000，QPS 相差 0.3%——
**在结构化数据上，默认配置在 256 维既不丢精度也不丢 QPS。**

因此不引入"按维度分档"的默认值（如 dim≥256 时退回 `inter=256`）：它只能在人工构造的均匀随机
数据上买到 0.012 的 recall，代价是构建速度从 3.05x 掉到 1.8x，而真实数据上看不到任何精度收益。
真正需要警惕的是"同轮基线 recall 本身就低"的数据集——那种分布下应该先看 5.1 的选点规则再决定。

---

## 6. 怎么读这张表

1. **先看 Recall 再看速度。** `inter=128, niter=32` 是最快的一档。把构建期图宽压到输出度、
   细化轮次压到 32，理论上会损失图质量；在这份数据集上 recall@10 仍是 1.000，所以它是推荐档，
   但这组参数不能直接搬到别的数据集：选点规则是"Recall 不低于 CPU 基线 1 个百分点"，
   换数据要重新扫一遍，而不是照抄最快的那一格。
2. **`inter` 和 `niter` 是两个独立的杠杆。** `inter` 决定构建期图宽（构建工作量近似线性），
   `niter` 决定近邻细化轮次（收益递减）。落盘图宽度始终等于 `degree`，所以这两个参数
   只影响构建质量与耗时，不影响快照体积和搜索参数。粗调顺序：先把 `niter` 压到 32 看 recall，
   再动 `inter`。
3. **构建窗口里 `first_query_s` 不是噪声。** 它包含图导入（`cuvsCagraIndexFromArgs`）与
   首查的显存驻留；如果这一项变大，说明图没走上导入路径，而是退回去重建了。
   判定证据分两处：DDL 里的构图时间取自 observer 日志
   `serialized GPU CAGRA snapshot(... built=true, build_us=N)`（表中"其中 GPU 构图"一列），
   重启导入取自 `deserialized GPU CAGRA snapshot(... graph_bytes=N)`（重启表里的
   `imported_graph_bytes`）。
   `--restart-check` 的 trace 里 `builds_in_trace=0` 才说明重启没有重建：
   `cuvs_build` 只由搜索路径上的**惰性重建**产生，正常构建（DDL 里那次）不写 trace。
4. **`qps` 用于确认没有拿搜索精度换构建速度。** 前四组（含 CPU 基线）的搜索参数完全一致
   （`ef_search=64`、`itopk=max(topk*4,512)`、`search_width=4`），QPS 的差异主要来自图结构本身；
   QPS 若明显下降，说明图质量确实变差了。最后两行是**故意**改 `itopk/width` 的服务侧对照，
   它们的 QPS 不能和前四行直接比。

---

## 7. 正确性验证

| 验证 | 脚本 | 断言 |
|---|---|---|
| 端到端功能 | `validation/run_gpu_smoke.sh` | 建索引 → 查询 → 结果非空，trace 含 `cuvs_build`/`cuvs_serve` |
| 快照落盘/恢复 | `validation/run_ddl_persistence_smoke.sh` | 含 `serialized GPU CAGRA snapshot`（`built=true`）与 `deserialized GPU CAGRA snapshot`（`graph_bytes>0`）；重启后 trace 无 `cuvs_build`，只有 `cuvs_serve` |
| 重启后结果一致 | `bench_cagra_build.sh --restart-check` | 重启后首查结果与重启前一致，`builds_in_trace=0`，`imported_graph_bytes=n*degree*4` |
| 精度对照 | 同脚本 recall@10 | 与 CPU 基线对比，差值写进第 5 节 |

fast-path 只在 `lib=cuvs` + `distance=l2` 的稠密 HNSW/HGRAPH 索引上启用
（`ob_mark_cuvs_if_needed()`，按 handle 标记），其余索引继续走原来的 vsag 路径。

---

## 8. 局限

- 只支持 `distance=l2`（cuVS CAGRA 侧的限制，DDL 层已拦 `lib=cuvs` + 非 L2）。
- 单卡；数据集必须能放进显存（`n*dim*4` 底库 + 图 `n*degree*4` + 搜索缓冲）。
- 快照体积随 `degree` 线性增长：`degree=128`、`n=1e6` 时邻接表约 512MB。
- 构建参数是全局环境变量，不是 per-index 的 DDL 参数；多租户场景要固化默认值或做 per-table 配置。
- recall 的余量取决于数据分布本身。均匀随机 256 维这种"近邻距离高度集中"的数据上，
  默认配置相对冻结基线参数会掉 0.020，超过 ±0.005 门限（第 5.2 节）；带簇结构的同维数据上是 1.0000 对 1.0000。
  换数据集上线前，应当用该数据集自己的样本复核一次，而不是默认这组参数处处安全。
- 第 5 节每个配置只做单次测量（每个配置重启一次 observer）。CPU 基线在同一次扫描里恰好被
  测了两次：159.21s 与 159.57s，相差 0.2%，可当作这个环境里单次测量的噪声量级
  （原始行留在 `$HOME/evidence/bench/sweep.raw.tsv`）。要更强的结论，每个配置跑 3 次取中位数，
  并把系统噪声（页面缓存、GPU 频率）一起记进证据目录。

---

## 9. 附录：参数速查

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SEEKDB_CUVS_BUILD_ALGO` | `nn_descent` | `auto` 交给 cuVS 自选 |
| `SEEKDB_CUVS_GRAPH_DEGREE` | 128 | 落盘/导入图的度数，也是快照体积的系数 |
| `SEEKDB_CUVS_INTERMEDIATE_DEGREE` | = graph degree | 构建期图宽，小于 graph degree 时被抬到 graph degree |
| `SEEKDB_CUVS_NN_DESCENT_NITER` | 32 | 近邻细化轮次 |
| `SEEKDB_CUVS_SEARCH_ITOPK` | `max(topk*4,512)` | 搜索候选池大小；调小可降单查延迟，代价是 recall |
| `SEEKDB_CUVS_SEARCH_WIDTH` | 4 | 每轮扩展的图搜索宽度，与 `itopk` 同向调节 |

上面四个是构建参数（改了要重建索引），下面两个是服务参数（只影响搜索，不用重建）。

| trace 事件（`OB_VSAG_TRACE=1`） | 含义 |
|---|---|
| `cuvs_build` | 搜索路径上的**惰性重建**（快照缺失/不兼容时的兜底），正常构建不写这条 |
| `cuvs_serve` | 走了一次 GPU 搜索 |
| `serialized GPU CAGRA snapshot` | 日志事件：DDL 里构图并落盘，带 `built`/`build_us`/`graph_bytes` |
| `deserialized GPU CAGRA snapshot` | 日志事件：重启导入图，带 `n/degree/graph_bytes` |
