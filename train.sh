TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOCK_DIR="log"

if [ ! -d "$LOCK_DIR" ]; then
  mkdir -p "$LOCK_DIR"
fi

LOG_FILE="${LOCK_DIR}/train_${TIMESTAMP}.log"
echo "Training started. Logs are being written to ${LOG_FILE}"

nohup python main_pl.py --run_name axis_rot --epochs 250 --devices 6,7 --num_traj 300 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process ID: $PID"