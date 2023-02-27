#!/usr/bin/env python3
import os, sys

def main():
    params = [sys.argv[1]]
    seed = 0
    np = 1
    for item in sys.argv[2:]:
        key, val = item.split("=")
        if key == "seed":
            seed = int(val)
        elif key == "np":
            np = int(val)
        else:
            params.append(item)

    cmds = []
    for i in range(np):
        cmd = ["partrac"] + params + [f"seed={seed+i}"]
        cmds.append(cmd)

    with open("parallel_runs.txt", "w") as ofile:
        ofile.write("\n".join(" ".join(cmd) for cmd in cmds))

    os.system(f"parallel -j {np} :::: parallel_runs.txt")
    
if __name__ == "__main__":
    main()