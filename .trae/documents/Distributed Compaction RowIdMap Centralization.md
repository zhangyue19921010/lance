## 现状定位
- 明确 row_id_map 目前在每个 task 执行阶段计算，并且在计算前会先 reserve 新 fragment id（否则 new RowAddress 无法生成）。参考 [optimize.rs](file:///Users/bytedance/workproject/lance/rust/lance/src/dataset/optimize.rs#L1107-L1128)。
- commit 阶段会把各 task 的 row_id_map 合并成一个大 HashMap 再进行 index remap。参考 [commit_compaction](file:///Users/bytedance/workproject/lance/rust/lance/src/dataset/optimize.rs#L1351-L1393)。

## 方案2（集中 reserve + 集中 row_id_map）性能评估切入点
- 计算量：按 task 使用 transpose 算法重建 mapping，时间复杂度近似 O(P)（P=被 compact 的旧 fragments 物理总行数），且需要为“缺失行”写入 None。参考 [remapping.rs](file:///Users/bytedance/workproject/lance/rust/lance/src/dataset/optimize/remapping.rs#L167-L196)。
- 内存量：row_id_map 的 entry 数量近似等于旧物理行数（保留行 K + 删除行 D ≈ P），commit 必须同时持有全量 mapping 才能调用 remap。评估在典型规模（1e7/1e8/1e9 行）下的内存上界。
- 序列化/网络：对比分布式场景下（worker→coordinator）传 HashMap vs 传 RoaringTreemap bytes（changed_row_addrs）的大小与 CPU 开销。

## 风险与瓶颈确认
- 识别 coordinator 单点：集中计算会把 O(P) 的 CPU + 内存压力集中到 commit 节点；评估是否需要并行化（按 task 并发 transpose）以及是否会放大 GC/内存碎片问题。
- 判断是否可跳过：若 dataset uses stable row ids 或启用 defer_index_remap，则不需要在 commit 阶段构造全量 row_id_map（分别对应 `needs_remapping=false` 或 `defer_index_remap=true` 路径）。

## 优化方向（如决定继续推进）
- 方向A：不落地全量 HashMap，改 IndexRemapper 接口为“流式/分片式 remap”，按 fragment 或按 task 逐批处理，降低峰值内存。
- 方向B：采用 defer_index_remap（写 Fragment Reuse Index + changed_row_addrs），把 transpose 推迟到后续 remap/读取阶段，避免 commit 端全量 HashMap。
- 方向C：若必须集中计算，增加 commit 端并行 transpose，并直接写入最终全局 map（避免 per-task map 再 extend）。

## 验证方式
- 用代表性数据集规模跑基准：记录 commit 阶段 CPU 时间、峰值 RSS、以及网络传输字节（分布式时）。
- 用回归测试覆盖：确保集中 reserve 后生成的新 RowAddress 与原逻辑一致，并验证 index remap/查询结果正确。