#!/bin/bash
# Stream live macOS system logs to a file that Vector can monitor

LOG_FILE="/Users/vovkes/loging/logs/live_system.log"

echo "🔄 Starting live macOS system log streaming to $LOG_FILE"

# Create the log file if it doesn't exist
touch "$LOG_FILE"

# Stream live system logs
log stream --style compact >> "$LOG_FILE" &
LOG_PID=$!

echo "✅ Live log streaming started (PID: $LOG_PID)"
echo "📄 Logs are being written to: $LOG_FILE"
echo "🛑 To stop: kill $LOG_PID"

# Keep the script running
wait $LOG_PID
