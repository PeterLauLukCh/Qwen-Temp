# File:"D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\run.py", generated on MON, MAY 18 2026  10:19, PSS(R)E release 36.03.02
import psse36
import psspy
psspy.psseinit()
psspy.case(r"""D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\PIF6_POC2_KLS_V9_updt.sav""")
psspy.dyre_new_2([1,1,1,1],r"""D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\PIF6_POC2_Disaggregated.dyr""")
psspy.addmodellibrary(r"""D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\sippc_pif6_v36_5_2_5_5.dll""")
psspy.addmodellibrary(r"""D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\SG1100UD_PSSE36_V0131101_260420.dll""")
psspy.addmodellibrary(r"""D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\NWSTAT01_V2_20250829_V36.dll""")
psspy.dynamics_solution_param_2([50,_i,_i,_i,_i,_i,_i,_i],[0.3,_f,0.001,_f,_f,_f,_f,_f])
psspy.cong(0)
psspy.conl(0,1,1,[0,0],[100.0,0.0,0.0,100.0])
psspy.conl(0,1,2,[0,0],[100.0,0.0,0.0,100.0])
psspy.conl(0,1,3,[0,0],[100.0,0.0,0.0,100.0])
psspy.strt_2([1,1],r"""D:\siemens ppc project\Projects\pif6_project\20260518\error_2026_05_17\1.out""")
