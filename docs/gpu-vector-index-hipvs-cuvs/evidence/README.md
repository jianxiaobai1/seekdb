# evidence 索引

所有文件都来自 AMD Radeon Cloud 实例（`gfx1100` / Radeon PRO W7900 / ROCm 7.2.4 /
`/opt/hipvs`）上的真实运行，没有本地估算。平台、口径与复现命令见 `../PERF_REPORT.md` 第 4 节。

## 运行器（脚本）

| 文件 | 用途 | 报告引用处 |
|---|---|---|
| `ab_run_ship.sh` | 250k×256 同轮 A/B：基线二进制（`634888d`）vs 参赛二进制（`f310b91`），参赛臂用当时默认建图器 | 1.1、5.1 |
| `ab_run_nn.sh` | 同上，参赛臂 `algo=0 niter=80` | 1.3、5.3 |
| `ab_final_same_round.sh` | 参赛默认参数（零环境变量 + 认证过的 bridge）对基线的同轮 A/B | 1.1、5.3 |
| `ff_repeat.sh` | 把上面那个同轮 A/B 连着跑 4 次并归档每轮的样本（1.1 主口径表就是它产出的） | 1.1、5.3、5.6 |
| `niter_sweep.sh` | 参赛版单臂扫 `niter=100/64`（256 维）与 128 维各档 | 5.3 |
| `ab128_final.sh` | 250k×128 同轮 A/B（bridge 用交付件 `out_final`），每个臂跑完删 store | 1.1、5.3 |
| `final_ship.sh` | 认证运行：不设任何 `SEEKDB_CUVS_*` 构图变量 + 干净重启后校验快照恢复 | 1.1、5.1、6 |
| `cpu_vsag.sh` | 基线自带 DDL 契约用例逐条重跑 + 同表 CPU（`lib=vsag`）构建对照 | 5.5、6 |
| `cl_sweep.sh` | 精确 all-neighbors 建图器按 `n_clusters` 扫描（分段耗时 / 峰值显存 / 召回） | 1.3、5.7 |
| `vram_sampler.sh` | 每 0.25s 采一次 `rocm-smi` 显存占用，落 `<arm>.vram` | 5.6 |
| `vram_prof.py`, `vram_report.py` | 显存样本的峰值/时间线复算（脚本里那行 `awk PEAK` 有 printf bug，故离线重算） | 5.6 |
| `recall_ab.py` | 结果 id 列表 vs seekdb 自身精确 KNN 复算 recall@10/@100 | 5.1 |
| `scripts/` | cuVS 层探针（`bench.c`）、代理数据生成（`gen_data.py`）、召回复算（`recall4.py`）、构图参数扫描（`sweep_build_params.sh`）、同轮重复轮的离线复算（`ff_round_report.py`） | 4.2、5.2、5.3 |

## 端到端运行记录

| 文件 | 内容 |
|---|---|
| `ff_rounds_summary.txt` | **1.1 主口径的出处**：参赛默认值（零环境变量）对基线的 4 轮同轮 A/B 逐轮复算（窗口、召回、QPS 几何比、显存），并写明作废的那一轮和 `.vram` 采样器被残留进程污染的坑 |
| `ff_ab.log` | 上述重复中第 1 轮的完整原始输出（含脚本里那行 `awk PEAK` 的 printf bug 现场） |
| `ship_run.log` | 256 维轮 1：基线窗口 49 546 ms（1.0000/0.9797）vs 精确 all-neighbors 参赛臂 25 015 ms（1.0000/0.9998，显存 1.438 GB） |
| `ab_nn.log` | 256 维轮 2：基线 50 547 ms（1.0000/0.9799）vs `niter=80` 臂 40 863 ms（0.9950/0.9753，Δ@10 = −0.0050） |
| `niter_sweep.log` | 参赛版各 `niter` 臂：`n256i100` 48 198 ms（1.0000/0.9795）、`n256i64` 35 899 ms（0.9950/0.9706）、128 维各档 |
| `ab128.log` | 128 维同轮：基线 45 663 ms vs 参赛 45 212 ms，两侧 recall@10 均 1.0000 |
| `final_ship.log` | 零环境变量认证运行 48 473 ms + 重启后 top-10/top-100 逐行一致、`deserialized GPU CAGRA snapshot` 1 次、构图 0 次 |
| `cpu_vsag.log` | DDL 契约 6 条语句的结果 + CPU HNSW 同表窗口 157 163 ms（0.4680/0.3277） |
| `endtoend_250k128.txt` | 上一轮 128 维同轮四臂（含 `build_us`/`total_us` 打点原样摘录） |
| `vram_peaks.txt` | 各臂窗口内本进程显存峰值复算表（门槛 1.25× 的判定依据） |
| `cl_sweep.log` | all-neighbors 分簇扫描原始输出 |
| `probe_exact.out`, `pe.out` | IVF-Flat 自搜索求精确 kNN 的分段耗时与峰值（1.94 GB） |
| `b1.txt`, `b2.txt`, `bench_final.txt` | 认证 bridge 的 cuVS 层探针：默认参数 42 471 ms / 精确 all-neighbors 20 401 ms（分段计时见 `b2.txt`） |
| `recall_matrix.txt`, `sweep7_256d.txt` | 四份 250K 代理数据的完整召回矩阵、256 维构图参数 sweep 原始输出 |

## `raw/` —— 原始采样与启动横幅

`raw/<dir>-<arm>.vram` 是显存采样点，`.marks` 是同轮的 `load_start/load_end/ddl_start/ddl_end`
单调时钟时间戳（`vram_peaks.txt` 就是按这两个时间戳切窗口算的），
`.observer.log` 是 observer 的启动横幅（含 `source revision`，用来证明两条臂跑的是哪两个二进制：
`1-634888d9b037...` 是冻结基线，`1-f310b91bb60a...` 是参赛分支）。

同轮重复的 4 轮原始 `.vram`（约 45 KB/臂）随实例一起销毁了，没有拉回本地：它们的窗口/整臂峰值已经
逐轮换算进 `vram_peaks.txt` 与 `ff_rounds_summary.txt`，本地保留的是同口径的另一批样本
（`ab256-*` / `ab-*`）。要复算就照 `ff_repeat.sh` + `scripts/ff_round_report.py` 跑。

## `ship_bridge/` —— 实例上实际编译的那份 bridge 源码

认证 `.so`（md5 `fb0c860c969df9cd9082477b610e7cae`）由这个目录里的源码在实例上用官方
`build.sh` 编出来。与仓库 `tools/hipvs_bridge/seekdb_cuvs_bridge.c`
（md5 `40da67e51fb7c394b60b9c953bbd9250`）逐字节 diff 只有一处：注释里的 `±` 换成了 `0.005`
（上传通道对非 ASCII 的处理），代码完全相同。

## `syntax_check/`

adaptor 改动区域的 `-fsyntax-only` 类型检查脚手架（改动区域 + 桩头文件），
用于在完整 seekdb 构建之外单独验证类型正确性。
