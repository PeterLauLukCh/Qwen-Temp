import os
os.chdir(r"E:\opencode_test_enviroment\test_cases _v36")

import psse36, psspy
psspy.psseinit(100000)

SAV_OUT = r"E:\opencode_test_enviroment\test_cases _v36\psse_ppc_test_bench_PMIN_QMAX.sav"
TARGET_P = 200.0
TARGET_Q = 330.0
STAT_LIMIT = 150.0

def set_wind(bus, pg, q):
    ig = [1,0,0,0,0,1,0]
    rl = [pg, q, q, q, 9999, -9999, 520, 0,1,0,0,1,1,0,0,0,1]
    return psspy.machine_data_4(bus, '1', ig, rl)

def set_stat(bus, q):
    ig = [1,0,0,0,0,1,0]
    rl = [0.0, q, q, q, 9999, -9999, 1000, 0,1,0,0,1,1,0,0,0,1]
    return psspy.machine_data_4(bus, '1', ig, rl)

def poc():
    _, c = psspy.brnflo(800, 900, '1')
    return c.real, c.imag

def vbus(b):
    return psspy.busdat(b, 'PU')[1]

# Load base SAV
psspy.case(r"psse_ppc_test_bench.sav")
psspy.fdns([0,0,0,1,1,0,99,0])
p, q = poc()
print(f"Base: P_POC={p:.0f} Q_POC={q:.1f}")

# ==== Step A: 调整有功 P 到 200 MW ====
print("\n--- Step A: Reduce P_POC to 200 ---")
# 从 500 MW/台 降低
wp = 500.0
for it in range(15):
    set_wind(40, wp, 0); set_wind(41, wp, 0)
    psspy.fdns([0,0,0,1,1,0,99,0])
    p, q = poc(); e = p - TARGET_P
    print(f"  Iter{it}: Wind_P={wp:.1f} → P_POC={p:.1f}  err={e:.1f}")
    if abs(e) <= 0.5:
        print(f"  P converged!")
        break
    wp -= e / 2

# 记录此时的 Q_POC
p, q = poc()
print(f"After P adjust: P_POC={p:.1f} Q_POC={q:.1f}")

# ==== Step B: Wind Q=0 (已为0, 确认) ====
print("\n--- Step B: Confirm Wind Q=0 ---")
# 重新锁定 Q=0 (可能被 Q 迭代覆盖)
set_wind(40, wp, 0); set_wind(41, wp, 0)
psspy.fdns([0,0,0,1,1,0,99,0])
p, q = poc()
print(f"  Wind P={wp:.0f}, Q=0 → Q_POC={q:.1f}")

# ==== Step C: STATCOM 满发 150 Mvar ====
print("\n--- Step C: STATCOM Q=150 ---")
set_stat(200, 150); set_stat(201, 150)
psspy.fdns([0,0,0,1,1,0,99,0])
p, q = poc()
print(f"  STATCOM=150 → Q_POC={q:.1f}  target={TARGET_Q}  need={TARGET_Q-q:.1f}")

# ==== Step D: 如果 Q < 330, 提高 Wind Q ====
if q < TARGET_Q - 0.1:
    print(f"\n--- Step D: Wind Q increase ---")
    wq = 0.0
    for it in range(15):
        set_wind(40, wp, wq); set_wind(41, wp, wq)
        psspy.fdns([0,0,0,1,1,0,99,0])
        p, q = poc(); e = q - TARGET_Q
        print(f"  Iter{it}: Wind_Q={wq:+.2f} → Q_POC={q:+.4f}  err={e:+.4f}")
        if abs(e) <= 0.1: break
        wq -= e / 2

p, q = poc()
print(f"\nFinal: P_POC={p:.1f} MW, Q_POC={q:+.4f} Mvar")
print(f"  V_POC={vbus(900):.4f}  V_bus700={vbus(700):.4f}  V_bus200={vbus(200):.4f}  V_bus40={vbus(40):.4f}")
psspy.save(SAV_OUT)
print(f"Saved: {SAV_OUT}")
