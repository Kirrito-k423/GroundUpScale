# C002：统一远端 Ascend 全机锁

- **阶段：** PROBE
- **动作类型：** PROBE / INTEGRATE

## 预注册

- **micro-goal：** 确认远端目标、固定 device visibility、安装并验证用户指定语义的公共 wrapper。
- **已有证据：** #30 冻结 Run Bundle；#36/#38 collector。
- **证据等级：** E1。
- **预期观察：** wrapper 在所有 issue 工作目录之外持有 flock，owner 生命周期正确。
- **风险：** 不得在 smoke test 初始化 NPU。

## 结果

- 连接目标：`root@192.168.9.225`；`/home/t00906153` 属主 `tsj:tsj`。
- 固定可见性：`ASCEND_RT_VISIBLE_DEVICES=0` / `npu:0`，由 #30/#36/#38 交叉确认。
- 安装：`/home/t00906153/.groundupscale/bin/with-ascend-lock`，SHA-256 `22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787`。
- 非 NPU smoke：命令内 owner 存在，结束后 owner 清理；通过。
- 初次尝试指定不存在的 Unix 用户 `t00906153` 安装时安全失败；没有初始化 NPU，随后保留实际目录属主并由 root 管理公共锁目录。

## 结论

- wrapper 可供各 issue 排队；首次真实完整 session 后再把 H-02 提升到 E3。
