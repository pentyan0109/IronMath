"""Record running Python/Julia processes into <folder>/processes_<tag>.json (never overwrites)."""
import datetime, json, subprocess, sys
from pathlib import Path
folder, tag = Path(sys.argv[1]), sys.argv[2]
q = subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python|julia|wsl.exe'} | "
                    "Select-Object Name,ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3"],
                   capture_output=True, text=True)
p = json.loads(q.stdout) if q.stdout.strip() else []
p = [p] if isinstance(p, dict) else p
out = folder / f"processes_{tag}.json"
with out.open("x", encoding="utf-8") as f:
    json.dump({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "processes": p}, f, ensure_ascii=False, indent=1)
print(out.name, [(x["ProcessId"], (x["CommandLine"] or "")[-70:]) for x in p])
