#!/bin/bash

pid_folder="/path/to/pid"

for pid_file in "$pid_folder"/*.pid; do
  if [ -f "$pid_file" ]; then
    content=$(cat "$pid_file")

    IFS=':' read -ra parts <<< "$content"
    pid="${parts[1]}"

    echo "Killing process with PID: $pid"
    kill -9 "$pid"
  fi
done
