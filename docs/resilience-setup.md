# 可选独立服务的安全配置

所有日常运行仍在 GitHub；不用每天手动启动。新增服务并不等于已经启用，配置前不会把已有密钥发送给新供应商。

## 独立模型接口

同一个接口下切换模型不能抵御整个接口停机。要启用已实现的跨接口备援，在[仓库 Secrets](https://github.com/RigenMo/eat-pubmed-paper-daily/settings/secrets/actions/new)新增 `LLM_BACKUP_API_KEY`，在[仓库 Variables](https://github.com/RigenMo/eat-pubmed-paper-daily/settings/variables/actions)填写 `LLM_BACKUP_BASE_URL` 与 `LLM_BACKUP_MODEL`。必须使用该备用供应商提供的独立凭证和实际支持的模型名；不在聊天、代码或日志中粘贴密钥。不配置时继续现有四模型路径。主、备接口共享每次请求总预算，备用不是无限重试。

## 独立定时监督

Cloudflare 监督器已经提供代码和部署工作流，但本会话没有提供 Cloudflare 账号及部署凭证，未宣称在外部运行。在上述 Secrets 界面设置 `CF_API_TOKEN`、`CF_ACCOUNT_ID`、`GH_WATCHDOG_TOKEN`，然后一次性运行仓库 `Deploy external watchdog`。权限和流程见 [scheduler/README.md](../scheduler/README.md)。此服务由 Cloudflare 独立计时，只唤醒已有 GitHub 监督器；发送仍由同仓库账本决定。

若 GitHub 执行平台自身不可用，即使外部时钟触发也不能保证完成发信；SMTP 接受也不等于核实了收件箱展示。SMTP 结果不明时停止自动重发，先对账。

## 无需新增凭证的部分

后台库存、加密检查点、Europe PMC/PubMed备援、少量匿名OpenAlex检索、合法OA深读和周综合复用现有配置。`OPENALEX_API_KEY` 可选，用于其当前服务允许的额外额度；缺失或服务失败都不会阻断早间发送。
