# C006 Spec Repository 结果

- 人类编写格式只有 YAML；每文件一个 `apiVersion/kind/metadata/spec` 文档。
- AnalysisPlan 通过仓库相对路径与版本显式引用其余 Spec；Workload 继续显式引用 Model。
- Pydantic Schema 全部 `extra=forbid`；重复 YAML key 在构造映射前拒绝。
- 引用解析后核验 kind、metadata.version 与可选 SHA-256；路径必须位于 repository root。
- 每个有效源记录 repo-relative path、kind、name、version 与内容 SHA-256。
- TDD：首次 RED 为公开模块不存在；最终 C006 5 passed，全量 6 passed。
