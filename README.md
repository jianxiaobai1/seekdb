# MiniVectorDB

单机、单进程、单客户端的轻量向量数据库，实现赛题要求的 SQL 子集、持久化、
事务与崩溃恢复，通过 stdin/stdout 的 JSON Lines 协议与判题驱动交互。

## 1. 构建与运行

- 语言/标准：C++17，无第三方运行时依赖；JSON 解析使用仓库内自带的
  `src/nlohmann/json.hpp`（3.11.3）。
- 构建：`make`（或 `g++ -O3 -std=c++17 -DNDEBUG -o minivectordb src/main.cpp`）。
- 容器镜像：`docker build -t minivectordb:latest .`，多阶段构建，运行阶段仅包含
  可执行文件。镜像已实际构建并以 `--image` 方式通过官方检查器（见第 4 节）。
- 启动入口：`/app/minivectordb --data-dir /data`。数据、日志等全部文件位于
  `/data`；程序不监听网络端口，调试信息只写 stderr。
- 启动时先完成 WAL 恢复，再向 stdout 输出一行 `{"type":"ready"}` 并 flush；
  收到 `{"op":"shutdown"}` 时回滚未提交事务、回 `{"id":...,"ok":true}` 后以
  退出码 0 结束。SIGTERM/SIGINT 同样回滚未提交事务后退出。

本地运行：

```bash
make
rm -rf /tmp/mvd-demo
./minivectordb --data-dir /tmp/mvd-demo <<'EOF'
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16), embedding VECTOR(2));"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1, '事务', '[1,0]'), (2, '索引', '[0,1]');"}
{"id":3,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding, '[0.9,0.1]') AS d FROM docs ORDER BY d LIMIT 2;"}
{"id":4,"op":"execute","sql":"BEGIN;"}
{"id":5,"op":"execute","sql":"DELETE FROM docs WHERE id = 1;"}
{"id":6,"op":"execute","sql":"ROLLBACK;"}
{"id":7,"op":"execute","sql":"SELECT id FROM docs ORDER BY id;"}
{"id":8,"op":"shutdown"}
EOF
```

预期输出中 `id=3` 返回两行及各自距离，`id=6` 回滚后 `id=7` 仍能看到 `[1, 2]`。

重启验证：复用同一 `--data-dir` 再次启动即可，已提交数据完整保留，未提交事务
不生效。

## 2. SQL 支持范围

- 表结构：`CREATE TABLE [IF NOT EXISTS]`；`BIGINT`、`VARCHAR(n)`、`VECTOR(n)`；
  恰好一个 `BIGINT PRIMARY KEY`（列级或表级均可），每表至多一个向量列。
  列级 `NOT NULL` 接受（语义上恒为非空）。
- 数据操作：`INSERT`（多行、可选列清单）、`SELECT`、`UPDATE`、`DELETE`；
  按主键或 `WHERE` 条件更新/删除。
- 查询：列投影、`SELECT *`、`= < <= > >= <>`、`AND`、`ORDER BY`（多键、
  `ASC/DESC`、别名/表达式）、`LIMIT`。
- 向量检索：`L2_DISTANCE(col, '[...]')`，精确 TopK；带普通列过滤时先在过滤
  集合内计算 TopK；`ORDER BY L2_DISTANCE(...)` 无需先投影该列。
- 索引：`CREATE VECTOR INDEX name ON table(vector_col)`。
- 事务：`BEGIN`、`COMMIT`、`ROLLBACK`；不显式开启事务时，数据修改语句自动
  提交；DDL 隐式提交当前事务。
- 不支持：`NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、嵌套事务、
  主键更新、`OR`、`DROP TABLE` 等；一律返回错误响应，不静默执行。

错误响应形如 `{"id":N,"ok":false,"error":{"code":"duplicate_key","message":"..."}}`；
`duplicate_key`、`dimension_mismatch`、`invalid_vector`、`value_too_long`、
`not_null_violation`、`table_not_found`、`column_not_found`、`table_exists`、
`index_exists`、`unsupported_sql`、`syntax_error` 等代码用于区分失败类别。

## 3. 实现概要

### 3.1 存储布局

只有一个追加式 WAL 文件 `/data/minivectordb.wal`，文件头为 8 字节 magic
`MVDWAL1\n`，其后是若干记录：

```
[u32 payload_len][u32 crc32][payload]
```

payload 首字节为记录类型：

- `C`：建表（表 id、表名、列名/类型/长度/主键标记），提交时以 `fsync` 落盘。
- `X`：创建向量索引的元数据。
- `T`：一批数据变更（表 id、操作类型 1=upsert/3=delete、主键、行编码长度、
  行编码），用于自动提交语句和事务提交。

行编码：BIGINT 8 字节；VARCHAR 为 `[u32 字节长度][UTF-8 字节]`；VECTOR 为
`[u32 维度][float32 × 维度]`。向量按 float32 存储，距离以存储后的值计算。
内存中维护主键到 WAL 绝对偏移的映射用于定位当前版本；最近行数据有 32 MiB
行缓存，溢出后查询改为顺序扫描 WAL，保证数据量超过内存预算时仍可工作。

### 3.2 事务与恢复

事务期间修改保存在内存 overlay（`table_id -> pk -> optional<Row>`，空值即
删除）。`COMMIT` 将整个事务编码为单条 `T` 记录，追加并 `fdatasync` 成功后才
返回成功响应；`ROLLBACK`、关闭、SIGTERM 直接丢弃 overlay。语句失败在应用
overlay/WAL 之前完成全部校验，因此不会留下部分修改，事务内先前成功的语句仍可
继续提交或回滚。

恢复时顺序读取 WAL，校验长度与 CRC：只接受完整记录，残尾记录（以及其后的
字节）被丢弃并 `ftruncate` 到最后一个有效边界，因此恢复过程可被反复中断，
已确认提交的前缀不会丢失，未完整写出的提交不会部分生效。索引元数据可从 WAL
恢复；本实现的向量检索不依赖索引加载，恢复完成后即可就绪。

### 3.3 向量检索

`L2_DISTANCE` 返回欧氏距离；TopK 使用大小为 K 的最大堆，距离相同按主键升序，
普通 `ORDER BY` 的并列项同样按主键升序。带过滤的 TopK 在满足 `WHERE` 的记录
集合内计算，不会先取全表 K 条再过滤。`search_mode` 为 `exact` 或 `ann` 时均
使用同一精确扫描路径，ANN 召回率为 1.0；`CREATE VECTOR INDEX` 仅记录元数据，
新增、更新、删除天然反映到检索结果。

## 4. 验证记录（本地）

- 官方公开用例：20/20 通过（`protocol_check/run.py --command ...`）。
- Docker 镜像：本机 Docker 29.1.3（WSL2 Ubuntu 26.04）实际执行
  `docker build -t minivectordb:local .` 成功；随后用官方检查器
  `python3 protocol_check/run.py --image minivectordb:local ...` 验证容器内启动、
  JSONL 握手、卷持久化与重启，公开用例 20/20、本地对抗用例 18/18 全部通过。
- 本地对抗用例：18/18 通过（`devtests/adv`，含多行语句、引号/裸向量、
  `ORDER BY` 表达式/DESC/无 LIMIT/LIMIT 0、失败语句原子性、事务内向量可见性、
  非法向量、嵌套 BEGIN、主键更新、Unicode 码点长度与排序、BIGINT 边界、
  过滤后 TopK、事务内删除后重插同主键等）。
- 崩溃恢复：8 个场景通过（`devtests/crash_test.py`）：未提交事务强杀回滚、
  自动提交/显式提交强杀后保留、提交过程中强杀的全有或全无、恢复途中反复强杀、
  WAL 残尾截断、首启文件头被截断、正常关闭回滚。
- 协议边界：`devtests/protocol_edge.py` 通过（多行 SQL、反引号标识符、
  字符串转义、注释、约 0.6 MiB 单行请求、非数字 id、未知 op、SIGTERM、
  shutdown 后无额外输出）。
- 随机差分测试：`devtests/fuzz.py` 共 6 组（400 步 + 5×800 步，不同随机种子）
  通过；用独立的 Python 模型逐语句对照混合 INSERT/UPDATE/DELETE/SELECT、
  显式事务与回滚、重启、过滤、多形态投影、排序/DESC/LIMIT 的结果，并校验失败
  多行 INSERT 不留下部分数据。
- 消毒器构建：`-fsanitize=address,undefined` 版本通过全部 20 个公开用例和
  300 步随机差分测试，无告警。
- 小型基准（WSL2，tmpfs，非正式评测环境）：10,000 行 × 128 维，每批 100 行
  提交，写入 0.63 s（约 15.9k 行/s）；1,000 次 TopK（K=10，一半带过滤）
  0.92 ms/查询；峰值 RSS 约 17.5 MiB；WAL 5.22 MiB；重启恢复 0.031 s。
- 超缓存路径：80,000 行 × 128 维（WAL 41.7 MiB，超过 32 MiB 行缓存），
  加载 4.83 s；过滤/不过滤 TopK 与独立暴力计算完全一致；峰值 RSS 66.3 MiB；
  重启 0.14 s；5,000 行事务回滚后无残留。

以上为本地开发验证，正式成绩以平台对锁定版本的评测为准。

## 5. 已知限制

- 单机单进程单连接；不支持多客户端并发。
- 不支持 `NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、外键、嵌套事务、
  主键更新与 `OR` 条件。
- WAL 只追加、不做压缩；大量删除后文件不会收缩。
- 向量索引仅保存定义，`exact` 与 `ann` 都走精确扫描；数据规模很大时查询为
  线性扫描。
- 大事务在提交时整体编码为一条记录，事务极大时提交需要相应内存。
- 镜像已在本地 Docker 中构建并跑通官方检查器与公开用例；尚未在组织方的正式
  评测容器与资源限额下运行，正式成绩以平台对锁定版本的评测为准。
