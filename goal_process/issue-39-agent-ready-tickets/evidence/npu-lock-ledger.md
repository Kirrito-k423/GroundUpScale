# Ascend host-lock ledger

所有权威 NPU 行为均由 `/home/t00906153/.groundupscale/bin/with-ascend-lock` 覆盖完整 session；wrapper SHA-256 为 `22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787`，visibility 固定为 `0`。

| Issue / session | Owner / 时间（Asia/Shanghai） | Cohort | 结论 |
|---|---|---|---|
| #42 `issue42-20260813-v1` | PID 2504805；20:08:11–20:17:54 | `ascend-npu-23b93a89d5fecc79` | 30/30 source PASS；最终 0/5 structured unknown（缺第二 eligible candidate） |
| #43 `issue43npu01` | PID 2858804；21:34:34–21:35:12 | `ascend-npu-23b93a89d5fecc79` | mean_scale/search dispersion >10% 后有界停止；无重试；authority 0/7 unknown |
| #44 `1850Z` | 完整锁 session | `ascend-npu-89022e60525d608f` | timing lanes 不稳定；非权威 |
| #44 `1900Z` | PID 2142475；18:55:33 起 | `ascend-npu-89022e60525d608f` | timing lanes 不稳定；非权威 |
| #44 `1910Z` | PID 2226102；18:59:55–19:03:24 | `ascend-npu-89022e60525d608f` | 10/10 timing PASS，但缺真实链 operand；旧 numeric 非权威，最终 unknown-v2 |
| #45 `2300Z` | 18:28:07–18:37:33 | `ascend-npu-23b93a89d5fecc79` | bundle 级锁元数据缺失；整批非权威 |
| #45 `1850Z` | PID 2132689；18:46:19–18:55:33 | `ascend-npu-23b93a89d5fecc79` | 30/30 PASS；2 域 qualified、3 域 structured unknown |
| #50 holdout v1 | PID 3839132；01:52:42–01:53:44 | `ascend-npu-23b93a89d5fecc79` | 24/24 PASS；发布后补 lock artifact，保留 immutable 历史但非权威 |
| #50 holdout v2 | PID 3984228；02:30:01–02:31:01 | `ascend-npu-23b93a89d5fecc79` | 合规 authority；24/24 PASS；median 1,927,420 ns；IQR 11,340 ns |

## 等待、失败与终态

- #44 在 #45 释放锁的同秒取得锁；无证据 session 交错。
- #42 首次 wrapper 尝试 PID 2493562 在任何 NPU 初始化和 artifact 写入前因 `uv` 依赖网络超时退出；0 bundle，随后以相同预注册计划和显式可信 Python 合法重试。
- #50 首次以非 root 用户打开公共锁时 permission denied；0 NPU、0 bundle，随后由 root 运行相同预注册 session。
- 任何早期绕 wrapper 的 pytest/import 尝试均被终止或在初始化前失败，未进入 authority。
- 收尾只读检查：`FLOCK_FREE`、`OWNER_ABSENT`。
