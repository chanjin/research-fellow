#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$project_root/.venv/bin/python" ]]; then
  default_python="$project_root/.venv/bin/python"
else
  default_python="python3"
fi
python_bin="${PYTHON_BIN:-$default_python}"
agent_dir="$HOME/Library/LaunchAgents"
agent_path="$agent_dir/com.research-fellow.nightly-summary.plist"
mkdir -p "$agent_dir"

cat > "$agent_path" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.research-fellow.nightly-summary</string>
  <key>ProgramArguments</key><array><string>$python_bin</string><string>$project_root/scripts/run_nightly_summary.py</string><string>--model</string><string>gpt-oss:20b</string></array>
  <key>WorkingDirectory</key><string>$project_root</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>$project_root/src</string></dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>0</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$project_root/data/nightly_summary.log</string>
  <key>StandardErrorPath</key><string>$project_root/data/nightly_summary.error.log</string>
</dict></plist>
PLIST

launchctl bootout "gui/$(id -u)" "$agent_path" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$agent_path"
echo "Installed: $agent_path"
