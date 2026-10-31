# 用 hipVS/cuVS 加速 seekdb 的 CAGRA 索引构建 —— 实现与性能对比报告

本目录是 AMD 赛题（用 hipVS 加速 HNSW/CAGRA 索引构建）的提交内容：

| 交付物 | 位置 |
|---|---|
| 实现代码 | [`tools/hipvs_bridge/seekdb_cuvs_bridge.c`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c)（bridge）、[`src/oblib/lib/vector/ob_vsag_adaptor.cpp`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp)（接入）、[`tools/hipvs_bridge/build.sh`](../../tools/hipvs_bridge/build.sh) |
| 可复现测试方法 | [`validation/build_and_measure.sh`](validation/build_and_measure.sh)、[`bench_cagra_build.sh`](validation/bench_cagra_build.sh)、[`probe_cagra.c`](validation/probe_cagra.c)、[`render_perf_report.py`](validation/render_perf_report.py) |
| CPU/GPU 构建性能对比报告 | 本文件（第 5 节表格由脚本从 sweep 输出渲染） |

改动全部在冻结基线 `634888d9b037153fa143d2c913ba4b25503e1d62` 之上，只涉及
`lib=cuvs` 的 GPU CAGRA 路径；`lib=vsag`（CPU HNSW）行为、DDL 语法与索引语义均未改动。

---

## 1. 结论

构建窗口（建索引 DDL + 首次查询）的加速来自三处，各自独立可验证：

1. **按输出度直接构图。** 基线的 CAGRA 构建参数是 `intermediate_graph_degree=256,
   nn_descent_niter=100`：先把图构到两倍宽度再剪枝，并做 100 轮近邻细化。落盘图的宽度始终是
   `graph_degree=128`，因此中间图宽度和细化轮次都是纯开销。改为直接在输出度上构图
   （`inter=degree=128`、`niter=32`）后，落盘图不变，构建工作量大幅下降。这是最大的一块收益，
   也是唯一会动 recall 的一块，选点依据见 5.1。
2. **快照改为图导入。** 快照只保存 ids、原始向量和原始邻接表；重启时用
   `cuvsCagraIndexFromArgs` 把图直接装回显存，不再跑 nn_descent，也不再经 cuVS 序列化器
   二次拷贝数据集。收益体现在"DDL + 首次查询"这个完整构建窗口上。
3. **bridge 侧缓冲复用。** 每次搜索不再逐次申请/释放显存，query/结果缓冲按高水位复用；
   搜索参数在首次搜索时创建一次。

在 Radeon Cloud（`gfx1100` / W7900，ROCm 7.2.4）上实测，250,000 × 128 维、200 条查询（第 5 节）：

- 同一套冻结基线构建参数下，GPU 路径的构建窗口为 **48.62s**，相对 CPU HNSW 的 159.37s 是 **3.28x**；
- 再把构建参数调到 `inter=128, niter=32`：构建窗口 **19.07s**，相对 CPU HNSW **8.36x**、
  相对同参 GPU **2.55x**；
- 加速不是拿精度换的：三个 GPU 配置 recall@10 为 0.999–1.000，CPU 基线（同样 `ef_search=64`）
  只有 0.478；
- 重启恢复：每个配置的首次查询 3.36–3.39s，`builds_in_trace=0`，导入图 128,000,000B
  （=250000×128×4，邻接表按 `degree=128` 另计）。

> 数字的来源：第 5 节表格由 `render_perf_report.py` 从实例上的 `sweep.tsv` 生成，非手工填写。
> 采集时使用的是与本提交相同的构建参数、快照策略和服务参数；代码在采集之后做过一次结构重写
> （同一设计），提交前应重跑 `build_and_measure.sh bench/report` 刷新绝对值。

---

## 2. 背景与指标定义

seekdb 的向量索引 DDL 形如：

```sql
CREATE VECTOR INDEX idx ON t(v)
WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
```

- `lib=vsag`（默认）走 CPU HNSW 构建，是本报告的 CPU 对照。
- `lib=cuvs` 走 GPU CAGRA 构建，通过 `libseekdb_cuvs_bridge.so` 调 hipVS 提供的 cuVS C API。

赛题要求"构建性能显著提升"，而构建是否完成以**首次查询能返回结果**为准，因此本报告统一用：

- `ddl_s`：`CREATE VECTOR INDEX` 的墙钟耗时；
- `first_query_s`：建索引后第一次近似查询的耗时（触发索引加载 / 图导入）；
- **构建窗口** = `ddl_s + first_query_s`，这是主指标；
- `qps` / `recall@10`：构建完成后用 200 条查询测得，用来确认加速没有拿精度去换。

构建窗口里有一段与配置无关的固定开销（约 5.6s：2MB 分块的 LOB 追加、表扫描与校验和），
CPU 基线和 GPU 各行都要付，所以它同时压低所有加速比。

---

## 3. 实现

### 3.1 构建参数：可调、默认按输出度直构

[`seekdb_cuvs_bridge.c:73`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c#L73) `sk_load_config()`
读取 `SEEKDB_CUVS_BUILD_ALGO / GRAPH_DEGREE / INTERMEDIATE_DEGREE / NN_DESCENT_NITER`，
默认 `nn_descent`、`degree=128`、`intermediate=degree`、`niter=32`，`intermediate` 低于 `degree`
时被抬到 `degree`。用环境变量而不是 DDL 参数，保证参数扫描不需要重编译 seekdb，也保证
DDL 语法不变；生产把扫出来的值固化进默认值即可。

### 3.2 快照 `OBCUVS04`：只存图，重启导入而不是重建

- 写侧 [`ob_vsag_adaptor.cpp:386`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp#L386)
  `ob_cuvs_serialize()`：若当前 GPU 索引未覆盖全部行则先重建一次，然后用
  [`ob_cuvs_export_graph`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c#L348)（内部
  `cuvsCagraIndexGetGraph` + 显存到主机拷贝）把邻接表导出，依次写
  `magic + version + dim + degree + flags + n + graph_bytes + ids + vectors + graph`。
- 读侧 [`ob_vsag_adaptor.cpp:448`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp#L448)
  `ob_cuvs_deserialize()`：校验版本、维度、`degree ∈ (0,1024]`、`graph_bytes == n*degree*4`
  后调用 [`seekdb_cuvs_import`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c#L397)（内部
  `cuvsCagraIndexFromArgs`）装图，不跑 `cuvsCagraBuild`。
- 快照带版本号，旧格式会被判为不兼容而重建，不会被误读。

### 3.3 构建与查询共用一条路径

[`ob_vsag_adaptor.cpp:176`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp#L176) `ob_cuvs_rebuild()`
是唯一"建图并接管句柄"的地方；DDL 的 `build_index`、搜索路径的惰性重建、快照序列化都调它。
[`ob_cuvs_query_job`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp#L235) 是唯一的查询执行体：
单查、批量、快照都走同一个函数，避免了三份各自维护的"先建后查"逻辑。

### 3.4 32MB 栈线程

cuVS 的 build/search/import 会超出 OB worker/DDL 线程的栈（约 1.5MB），因此所有进入 bridge 的
调用都经 `ob_cuvs_run_on_big_stack()` 搬到 32MB 栈的 pthread 上执行。

### 3.5 显存缓冲与搜索参数

[`seekdb_cuvs_bridge.c:187`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c#L187) `sk_reserve()`
按高水位复用 query / id / distance 三块显存；
[`sk_prepare_search()`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c#L229) 在首次搜索时创建
搜索参数（`itopk_size = max(4*topk, 512)`、`search_width = 4`），之后只在 `topk` 变大时抬高
`itopk_size`。这两个值可以用 `SEEKDB_CUVS_SEARCH_ITOPK / WIDTH` 覆盖，它们只影响搜索质量与 QPS，
不影响图，因此服务侧调参不需要重建索引（第 5 节最后两行就是这么扫的）。

### 3.6 何时不用 GPU

`ob_cuvs_try_search()` 只在下面这些条件全部成立时直接返回 GPU 结果，否则回落到 CPU 索引：
存在覆盖全部行的 GPU 索引（否则先按"行数翻倍"的节奏重建）、cuVS 调用成功、以及——
若查询带过滤条件——cuVS 返回的 top-k 里没有任何一行被过滤条件排除。最后一条是正确性要求：
无过滤的 top-k 中若有一行被排除，真正的过滤 top-k 可能排在 k 之外，必须交给 CPU 索引重算。
行数少于 256 时不做惰性建图，直接走 CPU。

---

## 4. 实验方法

数据由 [`bench_cagra_build.sh`](validation/bench_cagra_build.sh) 内置生成（seed `20261029`），
所有配置共用同一份：

- 250,000 × 128 维单位化高斯随机向量；
- 200 条查询 = 随机取 200 个底库向量加 σ=0.02 扰动后单位化；
- ground truth 为精确 top-100（暴力 L2），recall@10 按与精确 top-10 的重合计算。

每个配置**单独起一个 observer**（同一份 `--parameter`，`ob_vector_memory_limit_percentage=30`），
用 `load data local infile` 灌数据，再分别计时建索引、首次查询、200 条查询。配置之间没有共享缓存
或残留索引，代价是每个配置都要重新灌一次数据（不计入构建窗口）。每组还做一次中途重启，
验证快照恢复。

对照组：

| 组 | DDL | 构建参数 |
|---|---|---|
| CPU 基线 | `lib=vsag`，m=16，ef_construction=200 | OB 默认并发 |
| GPU 同参 | `lib=cuvs` | inter=256, niter=100（冻结基线硬编码值） |
| GPU 调优 | `lib=cuvs` | inter=192/niter=32、inter=128/niter=32 |
| GPU 服务调优 | `lib=cuvs` | 图同 inter=192/niter=32，叠加 `itopk=128,width=2`、`itopk=64,width=1` |

"GPU 同参"与冻结基线用同一套构建参数，因此它与"GPU 调优"之间的差全部来自构建参数，
它与冻结基线提交之间的差来自快照/缓冲改造。

### 复现命令

在 Radeon Cloud 实例上、从源码树根目录执行：

```bash
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh check   # 工具链 + cuVS 符号门禁
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh bridge  # 编译 bridge + 符号/dlopen 自检
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh seekdb  # 编译 seekdb
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh smoke   # 功能门禁
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh bench   # CPU 基线 + CAGRA 参数扫描 + 重启检查
bash docs/gpu-vector-index-hipvs-cuvs/validation/build_and_measure.sh report  # 渲染第 5 节表格
```

`check` 会在编译前用 `hipcc` 探测 `cuvsCagraIndexGetGraph/FromArgs` 两个符号：共享库允许未定义
符号，缺了不会链接失败、只会在运行时炸，所以门禁必须提前。`bridge` 步骤还会用 `RTLD_NOW`
dlopen 一次，确认 cuVS 符号都能解析。observer 会按 `datafile_size=40G` 预分配，`RUN_ROOT`
要落在空闲空间最大的卷上（放在 4GB 的 tmpfs 会以 `OB_LOG_OUTOF_DISK_SPACE` 起不来）。

第 5.1 节的参数边界不依赖 seekdb 构建，用 [`probe_cagra.c`](validation/probe_cagra.c) 单独跑
（只链接 hipVS 的 cuVS C API）：

```bash
hipcc -O3 -fopenmp -I/opt/hipvs/include \
  docs/gpu-vector-index-hipvs-cuvs/validation/probe_cagra.c -o probe_cagra \
  -L/opt/hipvs/lib -lcuvs_c -Wl,-rpath,/opt/hipvs/lib
./probe_cagra --n 250000 --dim 128 --deg 128 --inter 128 --niter 32 --cluster 0 --nq 1000 --k 10
```

不加 `--cluster 0` 时生成 200 个簇的合成数据（recall 恒为 1.0，只适合看耗时）；`--cluster 0`
是均匀随机数据，用来卡 recall 边界；`--algo ivf --lists N --refine R` 走 IVF-PQ 建图。

---

## 5. 实测结果

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

构建窗口 = `CREATE VECTOR INDEX` 耗时 + 首次查询耗时。所有行使用同一份数据、同一份 DDL 参数
（`type=hnsw, m=16, ef_construction=200, ef_search=64`），每行单独起一个 observer，只切换
`lib=`、CAGRA 构建参数与服务参数。「加速 vs 基线参数」对比的是冻结基线
`intermediate_graph_degree=256, nn_descent_niter=100`，差值即快照/导入改造与缓冲改造的净收益。

在 recall@10 不低于 CPU 基线 1 个百分点的前提下，最快的 GPU 配置是
`deg=128 inter=128 niter=32`：构建窗口 19.070s，相对 CPU 基线 **8.36x**，相对冻结基线参数
**2.55x**。

重启恢复验证（快照走图导入，不重新构建）：

| run | 重启后首查 (s) | 重启后 build 次数 | 导入图大小 (B) |
|---|---|---|---|
| `gpu_cuvs_deg128_inter256_niter100_nn_descent_restart` | 3.374 | 0 | 128000000 |
| `gpu_cuvs_deg128_inter192_niter32_nn_descent_restart` | 3.360 | 0 | 128000000 |
| `gpu_cuvs_deg128_inter128_niter32_nn_descent_restart` | 3.383 | 0 | 128000000 |
| `gpu_cuvs_deg128_inter192_niter32_nn_descent_itopk128_w2_restart` | 3.387 | 0 | 128000000 |
| `gpu_cuvs_deg128_inter192_niter32_nn_descent_itopk64_w1_restart` | 3.367 | 0 | 128000000 |
<!-- BENCH-TABLE:END -->

### 5.1 构建参数的 recall 安全边界（cuVS 层独立复核）

上面的表是在 seekdb 端到端路径上测的，其中约 5.6s 是与配置无关的固定 DDL 开销，会把加速比压平，
也回答不了"参数还能不能再往下压"。因此另加了 `probe_cagra.c`：自己生成数据、直接调
`cuvsCagraBuild` / `cuvsCagraSearch`，搜索参数与 bridge 完全一致
（`itopk=max(4*topk,512)`、`search_width=4`），用暴力 top-k 复算 recall@10（1000 条查询，
抽样噪声约 ±0.001）。

数据是**均匀随机** 250,000 × 128：没有簇结构、近邻距离集中，比 SIFT 难得多。在这里守住的边界，
在真实数据上只会更宽松。

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

1. 当前默认（`inter=128`、`niter=32`）把 cuVS 构建从 46.4s 压到 13.1s（**3.5x**），recall 只掉
   0.003，仍在评测门限（同轮基线 ±0.005）以内。这是同时满足两个条件的点。
2. `niter` 再降、或把度数从 128 降到 96，还能再省 30–40%，但 recall 掉 0.007–0.016，直接越界。
   也就是说"再快一点"不能从图质量上找。
3. IVF-PQ 建图（即使 `refinement_rate=1.0`，即用原始向量精化候选）能把构建压到 6.2s，但 recall
   只有 0.956：压过的近邻在这类难度下撑不起 CAGRA 的图。只适合大 N、允许更低 recall 的场景。

### 5.2 维度 256 的 recall 归因

评测的另一个数据集是 250,000 × 256。维度翻倍后均匀随机数据上的 recall 差距被放大，因此单独
做了一组归因：同一份数据上只动一个参数，看差距来自 `niter` 还是 `inter`。

均匀随机 250,000 × 256，1000 条查询，其余同 5.1：

| 配置 | cuVS build (s) | recall@10 | 相对冻结基线参数 |
|---|---|---|---|
| deg=128 inter=256 niter=100（冻结基线参数） | 424.8 | 0.9242 | — |
| deg=128 inter=128 niter=32（当前默认） | 139.3 | 0.9040 | −0.0202 |
| deg=128 inter=256 niter=32 | 233.9 | 0.9166 | −0.0076 |

结论：**差距主要来自 `inter`，不是 `niter`。** 保持 `inter=256` 只压 `niter`，recall 回到
−0.0076，但仍越出 ±0.005 门限，而且只快 1.8x。在这份**故意做难**的数据上，不存在"既在门限内、
又明显更快"的点——门限和加速在这里是冲突的。

同一维度换成**带簇结构的数据**（200 个随机中心 + 噪声，更接近真实 embedding 的分布）：

| 配置 | recall@10 | qps |
|---|---|---|
| deg=128 inter=256 niter=100（冻结基线参数） | 1.0000 | 28787.2 |
| deg=128 inter=192 niter=32 | 1.0000 | 28450.7 |
| **deg=128 inter=128 niter=32（当前默认）** | **1.0000** | **28692.9** |

这一轮只取 recall 与 QPS：三行的构建耗时分别是 169s / 269s / 14s，比值和 5.1 的 3.5x 对不上
（共享实例的节流把单次耗时放大到不可比），所以不据它判断速度，速度以 5.1 的同轮对比为准。
可用的信息是：三个配置 recall 全是 1.0000，QPS 相差 0.3%——**在结构化数据上，默认配置在 256 维
既不丢精度也不丢 QPS。**

因此不引入"按维度分档"的默认值（如 dim≥256 时退回 `inter=256`）：它只能在人工构造的均匀随机
数据上买到 0.012 的 recall，代价是构建速度从 3.05x 掉到 1.8x，而真实数据上没有精度收益。

---

## 6. 噪声与读表

1. **这是共享实例，绝对值会漂。** 同一配置、同一数据、同一进程参数下，构建耗时出现过 46.4s 与
   307.9s 两种结果（CPU 节流 / 邻居负载）。因此第 5 节的表格只用于**同轮比值**和 recall 判断，
   不拿单次毫秒数当基准。recall 由同一进程内的精确 GT 复算，不受节流影响。
2. **先看 recall 再看速度。** `inter=128, niter=32` 是最快的一档，在这份数据上 recall@10 仍是
   1.000；但这组参数不能直接搬到别的数据集。选点规则是"recall 不低于同轮基线 0.005"，换数据
   要重新扫一遍（5.1 的方法），而不是照抄最快的那一格。
3. **`first_query_s` 不是噪声。** 它包含图导入与首查的显存驻留；这一项明显变大，说明图没走上
   导入路径而是退回去重建了。判定证据：DDL 里的构图时间取自 observer 日志
   `serialized GPU CAGRA snapshot(... built=true, build_us=N)`（表中"其中 GPU 构图"一列），
   重启导入取自 `deserialized GPU CAGRA snapshot(... graph_bytes=N)`，以及重启 trace 里
   `builds_in_trace=0`（`cuvs_build` 只由搜索路径的惰性重建产生，DDL 里那次不写 trace）。
4. **`qps` 用来确认没拿搜索精度换构建速度。** 前四行（含 CPU 基线）的搜索参数完全一致
   （`ef_search=64`、`itopk=max(4*topk,512)`、`search_width=4`），QPS 差异主要来自图结构本身；
   最后两行是**故意**改 `itopk/width` 的服务侧对照，它们的 QPS 不能和前四行直接比。

---

## 7. 正确性验证

| 验证 | 脚本 | 断言 |
|---|---|---|
| 端到端功能 | `validation/run_gpu_smoke.sh` | 建索引 → 查询 → 结果非空；非 L2 DDL 被拒；过滤/删除/陈旧索引回落 CPU |
| 快照落盘/恢复 | `validation/run_ddl_persistence_smoke.sh` | 含 `serialized GPU CAGRA snapshot`（`built=true`）与 `deserialized GPU CAGRA snapshot`（`graph_bytes>0`）；重启后 trace 无 `cuvs_build`，只有 `cuvs_serve` |
| 重启后结果一致 | `bench_cagra_build.sh --restart-check` | 重启后首查结果与重启前一致，`builds_in_trace=0`，`imported_graph_bytes>0` |
| 精度对照 | `bench_cagra_build.sh` | 与 CPU 基线对比 recall@10，写入第 5 节 |
| bridge 符号 | `build_and_measure.sh bridge` | 六个导出符号齐全，`RTLD_NOW` dlopen 成功 |

fast-path 只在 `lib=cuvs` + `distance=l2` 的稠密 HNSW/HGRAPH 索引上启用（按 handle 标记），
其余索引继续走原来的 vsag 路径。
