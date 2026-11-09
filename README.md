# MiniVectorDB（可运行实现 v0.2）

赛题 `2026-competition-round2` 的单机向量数据库实现：**SQL 解析 → 行存 → 精确/过滤 TopK → 事务 → WAL 崩溃恢复** 主链路已打通，并补上了超内存数据溢出和 WAL 阈值 checkpoint。

本地已复跑通过：示例 SQL、错误码与类型校验、过滤 TopK、并列主键排序、事务读己写/回滚、语句失败原子性、重启持久化、`kill -9` 恢复、WAL 撕裂尾部丢弃、1 MiB 预算下 2 万行溢出后的精确检索与恢复。全部用例见 `tests/`，`bash tests/run_tests.sh` 可复跑（7 组全过）。

## 已实现

| 模块 | 状态 |
| --- | --- |
| CREATE TABLE | BIGINT 主键、VARCHAR(n)、VECTOR(n)；单主键、单向量列、重复列名/多主键校验 |
| CREATE VECTOR INDEX | 元数据持久化；检索本身仍是精确扫描（见限制 2） |
| INSERT | 多行、可选列清单、主键去重（语句内 + 表内）、缺主键报 1364、整条语句原子 |
| SELECT | `*`、普通列投影、`= < <= > >= !=`、`AND`、`ORDER BY`（列/别名/`L2_DISTANCE(...)` 表达式，ASC/DESC）、`LIMIT`、`L2_DISTANCE` |
| 向量检索 | 过滤后 TopK（先 WHERE 再取 K）；距离并列按主键升序；`LIMIT 0` 返回空；检索缓存与记录修改按代际失效 |
| UPDATE / DELETE | 条件更新/删除；拒绝主键更新；整条语句原子 |
| 事务 | `BEGIN` / `START TRANSACTION` / `COMMIT`（`END`）/ `ROLLBACK`；事务内读己写；语句失败只回退该语句；DDL 隐式提交 |
| 内存预算 | 解码行超出 `--mem-budget` 时整表溢出到 mmap 追加文件 `*.spool`；点查、扫描、TopK、更新、删除、恢复仍返回精确结果 |
| 持久化 | 快照 + 每事务一条 CRC WAL；提交 `fdatasync` 后才应答；WAL 超过 `--wal-checkpoint-mb` 自动 checkpoint；关闭时全量 checkpoint 并截断 WAL |
| 恢复 | `kill -9` 后重放 WAL；未提交事务不生效；撕裂尾部按长度/CRC 丢弃并**截断文件**，避免后续提交被坏记录挡住；重复恢复幂等 |
| 协议 | stdin/stdout、READY 标识、逐语句哨兵、stderr 日志、SIGTERM 优雅关闭 |

类型约束按题意校验：VARCHAR 长度按 Unicode 码点计；字符串比较按 UTF-8 字节序（等价 Unicode 码点序）；向量必须是可表示为有限 float32 的数值，拒绝 NaN/Infinity/维度不符/溢出；距离按存储后的 float32 计算，用 double 累加。

## 构建与运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/minivectordb --data-dir /tmp/mvdb < tests/sample.sql
```

Docker（正式环境为 Linux x86_64，容器内前台运行）：

```bash
docker build -t minivectordb:latest .
docker run --rm -i -v minivectordb-data:/data --cpus=2 --memory=2g minivectordb:latest
```

启动参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--data-dir <dir>` | `/data` | 数据目录；快照、WAL、spool 都写在这里 |
| `--mode exact\|ann` | `exact` | 当前 ann 仍复用 FlatIndex（精确扫描），结果合法但无加速 |
| `--sentinel <str>` | `__MVDB_END__` | 每条语句输出后的结束标记；传空串关闭 |
| `--ready <str>` | `READY` | 恢复完成后的就绪标识 |
| `--echo-ok` | 关 | DML/DDL 输出 `OK <affected>`，便于本地调试 |
| `--mem-budget <MB>` | cgroup 限额的 40%（夹在 64 MiB–1 GiB） | 解码行内存上限；超出后整表溢出到 spool，`0` 表示交给自动推导 |
| `--wal-checkpoint-mb <MB>` | `128` | WAL 超过该大小就在语句间 checkpoint；`0` 只在关闭时 checkpoint |

## 临时协议（重点：需按题包替换）

> 官方题包尚未发布。协议、错误码、结果格式全部集中在 `src/main.cpp` 和 `src/common.h`，引擎不感知输出格式；拿到题包后只改这两处即可。

当前约定：

- 启动恢复完成后打印 `READY`（一行）。
- stdin 按顶层 `;` 分句；支持字符串内分号、`''` 与 `\'` 转义、`--`/`#`/`/* */` 注释。
- `SELECT`：先表头，再逐行，列间 `\t`；空结果只输出表头。
- DML/DDL：默认无输出；`--echo-ok` 时输出 `OK n`。
- 错误：`ERROR <code>: <message>`，错误码暂用 MySQL 风格（1062 主键重复、1406 超长、1265 取值非法、1364 缺主键、1264 越界、1146 表不存在、1054 列不存在、1050 表已存在、1061 索引已存在、1235 不支持等）。
- 每条语句响应以 `__MVDB_END__` 结尾（可关闭）。
- EOF、`exit;`、`quit;`、`\q` 触发优雅关闭：回滚未提交事务 + checkpoint。

示例输出：

```
READY
id	content	distance
1	事务与恢复	0.141421
2	向量索引	1.272792
__MVDB_END__
```

## 存储、事务与恢复设计

文件（全部位于 `--data-dir`）：

- `tbl_<sanitized>.snap`：表快照，`MVSNAP01` + schema + 活行；写入走 `tmp → fsync → rename → fsync(dir)`，rename 原子。
- `wal.log`：每个事务（或自动提交语句）一条记录：`[u32 payload_len][u32 crc32][payload]`，payload 是操作列表（CREATE_TABLE / CREATE_INDEX / INSERT / UPDATE / DELETE，整行镜像）。
- `tbl_<sanitized>.spool`：可重建的溢出缓存，mmap 追加写；快照/WAL 才是事实来源。

提交路径：内存就地应用修改并记录 undo/redo → 追加 WAL → `fdatasync` → 应答。未收到成功响应的事务允许整体丢弃。

回滚：逆序回放 undo（旧行镜像），恢复记录内容与可见集合，向量检索随之一致。

恢复：先载入快照，再顺序重放 WAL。记录独立且幂等（INSERT/UPDATE 按主键覆盖，DELETE 删除，CREATE 跳过已存在），因此“checkpoint 写完快照但没来得及截断 WAL”后重复恢复也安全。长度越界或 CRC 不匹配即停止重放，并立即把文件截断到最后一条好记录之后——否则下次启动的新提交会接在垃圾后面，再次崩溃时这些已确认事务会被静默丢掉。

正常关闭：回滚未提交事务 → 全量快照 → 截断 WAL → fsync。运行期间 WAL 超过阈值时，在一条语句结束、事务已提交后触发同样的 checkpoint，避免长时间写入留下一个需要重放很久的巨日志。

内存预算：解码后的行常驻内存；当某张表的驻留行超出预算时，`enforce_budget()` 把最大的表整表编码进 spool，并把行引用改为 `(offset, len)`。扫描、TopK、点查都在需要时从 spool 解码，因此结果仍是精确的；启动加载也按 4096 行一批执行预算检查，避免恢复阶段先把内存打满。

向量检索：`L2_DISTANCE` 使用 float32 存储值、double 累加后开方；TopK 在 WHERE 候选集内计算；排序键为 `(距离, 主键)`，普通 ORDER BY 并列也用主键升序兜底。距离计算走 AVX2/FMA 四累加器内核（运行时探测，无 AVX2 时退回标量）。`FlatIndex` 是当前唯一实现，`ann` 模式复用同一实现，因此 Recall@10 = 1.0。

## 测试

```bash
bash tests/run_tests.sh          # 7 组：4 组 SQL + 崩溃恢复 + 溢出 + 语义
python3 scripts/bench.py          # 10k×128 维、1000 次 TopK 的冒烟 benchmark
```

- `tests/sample.sql`：赛题示例 + ROLLBACK 后数据核对。
- `tests/errors_txn.sql`：错误码、VARCHAR 码点长度、语句失败原子性、事务读己写、回滚、主键更新拒绝。
- `tests/ties.sql`：过滤 TopK、距离并列主键升序、过滤后取 K、DESC、`LIMIT`、字符串内分号和注释。
- `tests/restart_phase*.sql`：正常关闭 + 重启快照。
- `tests/crash_test.py`：未提交事务 rollback、已确认 COMMIT 保留、WAL 撕裂尾部（恢复后再写入再崩溃，验证尾部已截断）、WAL 阈值 checkpoint 后的快照 + 日志尾部恢复。
- `tests/spill_test.py`：1 MiB 预算 + 2 万行强制溢出，验证点查、过滤/非过滤 TopK、更新、删除、事务与重启恢复仍精确。
- `tests/semantics_test.py`：转义与注释、Unicode 长度、LIMIT 边界、排序并列、不支持语法、插入原子性、事务状态机、向量校验、TopK 语义、空表、大小写/反引号、重启回读。

本地 benchmark（WSL Ubuntu、Release、10k 行数据量，仅供趋势参考）：

```
rows=10000 dim=128 batch=100 queries=1000 K=10
write      :    0.309 s  (32,355 rows/s)
query      :    0.177 s  (5,650 q/s, 0.18 ms/query)
peak RSS   :     13.6 MiB
close+ckpt :    0.007 s
recovery   :    0.013 s
```

## 提交材料

- `src/`、`CMakeLists.txt`：完整实现与构建文件。
- `Dockerfile`：多阶段构建，运行阶段离线可用，入口 `--data-dir /data`。
- `agent/conversation.json`：Agent prompt 学习材料；根目录 `conversation.json` 为同一份内容，兼容评测入口的不同路径约定。
- `README.md`：本运行说明。

以上文件须位于同一个 commit；提测时提供仓库 HTTPS URL、分支 `2026-competition-round2` 和完整 40 位 SHA。

## 已知限制（按优先级）

1. **协议未对齐题包**：结果格式、错误码、就绪标识、正常退出方式都是临时约定，拿到官方题包后需要按实际驱动调整。
2. **无真正 ANN**：`CREATE VECTOR INDEX` 只持久化元数据；`ann` 模式仍是精确扫描，Recall@10 = 1.0 但没有索引加速。性能榜要拿分需要实现 IVF-PQ / HNSW，并保证 filter + 增量维护正确。
3. **每事务一次 `fdatasync`**：写入 benchmark 的持久性正确，但批量写可合并组提交以优化 I/O 与耗时。
4. **spool 只增不减**：溢出文件是追加写，更新/删除后的旧编码会留在文件里；重启加载会重建引用，但不会压缩文件本身。长时间大量改写的场景需要重写/压实。
5. **溢出表的过滤靠流式解码**：没有为溢出表建立持久化二级索引，带 WHERE 的查询需要扫完候选行；正确性不受影响。
6. **不支持项**（赛题不要求，均返回错误）：NULL、JOIN、聚合、子查询、ALTER、外键、嵌套事务、主键更新。
7. **DDL 事务语义**：CREATE TABLE / CREATE INDEX 会隐式提交当前事务（MySQL 风格），题包若另有约定需调整。
8. **Dockerfile 未在本机构建**：开发环境没有 Docker，只做了静态检查；首次平台预检时应确认镜像能构建并启动。
