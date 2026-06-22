# PSS/E 新能源小网数据理解说明

这份说明用于解释 `real-data-new` 里的两个新能源小网如何类比我们现有的 `M1+M2+EMT` framework。当前分析是 **static inspection**：只读文件、脚本和 DYR model headers，不执行 PSS/E，不加载 vendor DLL。

## 一句话总结

现有 `M1+M2+EMT` 是 research/test framework：用 `pandapower`、`ANDES` 和 `SCR proxy` 做可控、可复现的简化评估。

`real-data-new` 是 PSS/E real-data bundle：包含真实风光/新能源场站小网的 `.sav`、`.dyr`、`DLL` 和 `psspy scripts`。它更像实际 interconnection study 里的小型等值网，里面有真实的 `PPC`、`inverter`、`STATCOM` 和 vendor user-defined model。

## 两个小网

### 1. `2026_05_17`

这是更接近真实项目的 PIF6 小网。

核心文件：

- `PIF6_POC2_KLS_V9_updt.sav`: PSS/E static case，等价于 real M1 base network。
- `PIF6_POC2_Disaggregated.dyr`: PSS/E dynamic data，定义 plant controller、inverter、STATCOM 等 dynamic models。
- `sippc_pif6_v36_5_2_5_5.dll`: plant controller / PPC user model library。
- `SG1100UD_PSSE36_V0131101_260420.dll`: Sungrow inverter user model library。
- `NWSTAT01_V2_20250829_V36.dll`: STATCOM user model library。
- `run.py`: PSS/E script，加载 `.sav`、`.dyr`、DLL，并启动 dynamic simulation。

`run.py` 显示它由 `PSS(R)E release 36.03.02` 生成；同时目录里的 shortcut 指向 `PSS/E 36.5`。这说明未来真跑时要注意 PSSE minor version 和 DLL ABI compatibility。

DYR 结构：

- `SIPIF6`: 1 个 plant-level PPC，在 bus 2。
- `SGCVTF0131101`: 240 个 Sungrow inverter models。
- `NWSTAT01`: 10 个 STATCOM models，在 buses 800-809。
- `SIAUX1`: 250 条 auxiliary links，把 PPC 连接到 inverter/STATCOM。
- `GENCLS`: 1 条 classical generator placeholder/model。

可以理解成：

```text
PPC / plant controller: SIPIF6 @ bus 2
  -> SIAUX1 auxiliary links
    -> 240 Sungrow inverter models: SGCVTF0131101
    -> 10 STATCOM models: NWSTAT01
  -> POC / monitored branches / voltage-reactive control logic
```

这不是“generic solar at bus 10”。这是一个具体新能源场站模型，包含 controller hierarchy、dynamic response、Q/P limits、voltage/frequency ride-through-like parameters 和 vendor DLL behavior。

### 2. `test_cases _v36`

这是更小的 PPC/PQ control benchmark。

核心文件：

- `psse_ppc_test_bench_PMIN_QMAX.sav`: 小型 PSS/E static case。
- `psse_ppc_test_dynamic.dyr`: dynamic model file。
- `sippc_r5gz_v36_5_2_6_test.dll`: test PPC user model library。
- `step_pmin_qmax.py`: 调整 P/Q 的 test script。
- `qref_PMAX_QMAX_v3_analysis.png`: original vs recompiled DLL 的响应对比图。

DYR 结构：

- `SIR5GZ`: 1 个 PPC / plant controller，在 bus 700。
- `REGCAU1`: 4 个 generator/converter interface models。
- `REECAU1`: 4 个 renewable electrical control models。
- `SIAUX1`: 4 条 PPC auxiliary links。

Script 目标：

- POC branch: `800-900`, circuit `1`
- Wind buses: `40`, `41`
- STATCOM-like buses: `200`, `201`
- `TARGET_P = 200 MW`
- `TARGET_Q = 330 Mvar`
- `STAT_LIMIT = 150 Mvar`

这个小网更像一个 controller unit test：看 `P_POC`、`Q_POC`、`PMAX/QMAX`、wind Q 和 STATCOM Q 如何共同跟踪 POC target。

## 和现有 M1+M2+EMT 的类比

| 现有 framework | Real-data analogue | 差异 |
|---|---|---|
| `M1`: `pandapower` power flow / CIA / N-1 | PSS/E `.sav` static case | 从 IEEE toy/public case 变成真实 PSS/E 小网 |
| `M2`: `ANDES` transient stability | PSS/E `.dyr + DLL` RMS dynamic simulation | 从 `static PQ approximation` 变成实际 PPC/inverter/STATCOM dynamic models |
| `EMT`: SCR proxy | 暂无真正 EMT waveform；可先导出 POC/SCR/dynamic metrics | PSS/E RMS 仍不是 PSCAD/EMT waveform，但比当前 M2 更真实 |
| `ToolRegistry` outputs | PSSE run output JSON | agent 不需要直接理解 `.sav/.dyr/.dll`，只需要结构化 observation |
| frozen oracle / benchmark | frozen PSSE simulation results | 可作为更真实的 eval/RL labels |

## 为什么说“小网如大网”

这两个 PSS/E 小网规模不大，但结构像真实大网研究的一部分：

- 有 `POC`，即 point of connection。
- 有 `PPC`，负责 plant-level voltage/reactive/active power control。
- 有大量 inverter 或 renewable unit models。
- 有 `STATCOM` 或 reactive support。
- 有 monitored branch、remote voltage、P/Q target、limits。
- 有 dynamic initialization 和 simulation script。
- 有 vendor DLL，说明控制逻辑不是开源 Python，而是实际工程模型。

所以它们适合用来训练我们理解真实 study workflow：

```text
large grid study
  -> extract/equivalent renewable plant area
  -> build small PSS/E study case
  -> load SAV/DYR/DLL
  -> run load flow and dynamic checks
  -> export POC metrics and pass/fail decision
```

## 当前 AI/代码能做什么

现在已经可以做：

- 生成 file manifest 和 SHA256。
- 识别 `.sav`、`.dyr`、`.dll`、`script`。
- 统计 DYR model counts。
- 找出 PPC、inverter、STATCOM、SIAUX link 层级。
- 提取 script 里的 PSSE version、`psspy` calls、case/dyr/dll paths。
- 把 real data 映射到 `M1+M2+EMT` analogy。

命令：

```bash
PYTHONPATH=Code python3 Code/scripts/inspect_real_data.py --summary-only
```

查看单个 bundle：

```bash
PYTHONPATH=Code python3 Code/scripts/inspect_real_data.py \
  --bundle "real-data-new/2026_05_17" \
  --summary-only
```

## 当前不能做什么

当前本地不能：

- 直接 load `.sav`。
- 执行 `psspy`。
- 调用 Windows `.dll` user models。
- 跑真正的 PSS/E dynamic simulation。
- 从 `.sav` binary 里完整解码 topology 和 operating point。
- 声称已经验证 dynamic pass/fail。

这些需要：

- Windows
- PSS/E 36
- `psspy`
- matching PSSE user-model DLL ABI
- 合适的 runtime libraries

## 推荐下一步

不要马上把这些文件硬塞进 `pandapower` 或 `ANDES`。更稳的路线是分三层：

1. `real-data understanding layer`
   已开始：静态解析、manifest、model hierarchy、framework analogy。

2. `PSSE execution layer`
   在有 PSS/E 的机器上跑 `psspy`，输出 compact JSON：
   - load-flow convergence
   - bus voltage range
   - branch loading
   - POC P/Q/V/frequency
   - dynamic channel metrics
   - controller/inverter/STATCOM response
   - pass/fail/reason codes

3. `GridMind adapter layer`
   把 PSSE JSON 映射成现有 `run_integrated_assessment` 风格的 observation：

```json
{
  "tool": "run_real_psse_assessment",
  "stage_reports": [
    {"stage": "m1_psse_powerflow", "status": "pass"},
    {"stage": "m2_psse_dynamic", "status": "borderline"},
    {"stage": "f4_scr_or_emt", "status": "not_run"}
  ],
  "summary": {
    "poc_p_mw": 200.0,
    "poc_q_mvar": 330.0,
    "m2_backend": "psse_rms_dynamic"
  }
}
```

这样 agent 层不用变太多：它仍然看 structured tool observation，只是 backend 从 simplified public framework 升级为 real PSSE data。
