请你评估我的建议，提出一个改动计划

1. “find the OAR” with no body region can still make Qwen guess structures instead of asking which region. Add a deterministic gate: OAR without a confirmed site must ask the user before Qwen runs.

2. 前端出现的Ask segagent下的liver示例不需要
3. 前端视图界面的loading标识容易卡顿并且对视图有一定遮挡
4. 对于生成的mask，agent调用segment model时直接在视图中显示，不再让user review mask，相关逻辑都删除
5. enter键直接run，不要cmd+enter
6. **Some important new architecture is not connected.**
   `EvidenceLedger`, evidence statuses and typed `Claim` objects are good designs, but runtime execution still uses ordinary observations. The entity-keyed retrieval method is also only exercised by tests. These are currently foundations, not completed features.

7. 右侧的对话界面参照一个典型的agent界面，类似codex，claude code，展示思考过程，类似对话形式，有意图检测，plan待办事项，tool us等等的显示不要做成现在的一个一个板块冒出来。
8. 删除qc功能，现有的qc功能没有实际作用

9. confidence是人工规定的没有意义