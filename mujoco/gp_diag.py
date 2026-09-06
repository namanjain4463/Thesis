"""Diagnostic: does each family DISCRIMINATE the grasp choice? Sweep gy and yaw on body S and T."""
import numpy as np
from gp_core import scene, run_episode

for body in ("S", "T"):
    print(f"\n############ BODY {body} ############")
    for fam in ("A", "B", "C"):
        _, meta = scene(fam, body)
        print(f"\n=== family {fam}  CoM_y={meta['comy']*1000:.0f}mm "
              f"pocket=({meta['pocket_hx']*1000:.0f},{meta['pocket_hy']*1000:.0f})mm ===")
        # sweep grasp location (yaw=0) then orientation (gy=comy)
        def lab(r): return "OK  " if r["placed"] else r["label"]
        loc = []
        for gy in (0.0, 0.02, 0.035, 0.05, 0.062):
            r = run_episode(fam, body, dict(gy=gy, yaw=0.0, close=meta["frange"], tspeed=0.4))
            loc.append(f"gy={gy*1000:.0f}:{lab(r)}")
        print("   loc(yaw=0): " + "  ".join(loc))
        yw = []
        for yaw in (0.0, np.pi/4, np.pi/2):
            r = run_episode(fam, body, dict(gy=meta["comy"], yaw=yaw, close=meta["frange"], tspeed=0.4))
            yw.append(f"yaw={np.degrees(yaw):.0f}:{lab(r)}")
        print("   yaw(gy=CoM): " + "  ".join(yw))
