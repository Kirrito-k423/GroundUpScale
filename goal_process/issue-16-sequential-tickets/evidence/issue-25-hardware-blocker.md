# Issue #25 真实硬件 blocker 证据

- **采集时间：** 2026-08-08T03:00–03:12+08:00
- **执行者：** `/root/issue_25`
- **结论：** BLOCKED；未进入 TDD/code-review，未写入实现、提交或关闭 #25。

## 工单前置状态

只读 `gh issue view` 核验 #22、#23、#24 均为 CLOSED；#25 自身仍为 OPEN。当前 HEAD `a5c117d` 已包含三个 blocker 的交付。

## 本机 M4

系统身份显示 Darwin arm64、Apple M4、Mac16,12。设备身份在对外记录中仅保留 SHA-256 前 12 位 `595686a95512`。

执行：

```text
uv run groundupscale preflight --sample-interval-seconds 0.2 --process-samples 3 --json
```

结果：exit 2，`eligible=false`；原因：

```text
load-above-policy
total-competing-cpu-above-policy
```

因此当前窗口的本机测量证据不能晋级为可信 cohort 证据。

## 本地 Run Bundle

对 `.groundupscale/runs/*/run.manifest.json` 做只读计数：

- manifest 总数：27；
- `environment_validity=passed`：0；
- 非 `apple-m4-*` cohort：0。

不存在可回放的第二真实硬件原始 Bundle。

## 远端候选硬件

对 `/Users/Zhuanz/Documents/autoresearch/config/config.yaml` 仅做脱敏读取，不记录主机名、地址、用户名、密码或密钥内容。4 台配置候选机的身份文件均存在，TCP 端口可达，但：

- 4/4 SSH 连接均在密钥交换前被远端关闭，exit 255；
- 4/4 只读 Redfish 探测均在 TLS 建立前关闭。

由此无法执行 capability discovery、fingerprint、preflight、timing plan、correctness、Completion Boundary，亦无法采集第二 cohort 的 digest-verifiable Run Bundle。

## 不可采用的替代

仓库存在 `MeasurementAdapter` Protocol；可见具体 Adapter 仅有测试中的 `_RecordedFixtureAdapter`。#25 acceptance criteria 明确规定“无法满足时明确 blocked/unknown，不以测试 Adapter 代替真实验收”，故不能用合成/测试证据绕过硬件门禁。

## 队列动作

- #25 保持 OPEN；无实现、测试、review、commit 或 close。
- 未创建 #26 子代理，#26 保持 OPEN。
- 严格按用户规则在首个真实 blocker 处停止。
