#!/bin/bash
# =============================================================================
# Submit Time-Frequency Map Computation Jobs to Slurm
# =============================================================================
# This script submits TF map computation jobs to the Slurm cluster.
#
# Usage:
#   # Generate session list and submit array job
#   bash scripts/submit_tf_jobs.sh --eeg-folder /path/to/eeg --output-dir /path/to/output
#
#   # With custom parameters
#   bash scripts/submit_tf_jobs.sh --eeg-folder /path/to/eeg --output-dir /path/to/output \
#       --partition medium --memory 16G --time 02:00:00
#
#   # Dry run (don't actually submit)
#   bash scripts/submit_tf_jobs.sh --eeg-folder /path/to/eeg --output-dir /path/to/output --dry-run
#
#   # Force recomputation of existing files
#   bash scripts/submit_tf_jobs.sh --eeg-folder /path/to/eeg --output-dir /path/to/output --force
# =============================================================================

set -e

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$HOME/envs/epimicro"  # Adjust to your environment

# Slurm parameters (can be overridden via command line or environment)
PARTITION="${TF_PARTITION:-short}"
MEMORY="${TF_MEMORY:-16G}"
CPUS="${TF_CPUS:-4}"
TIME_LIMIT="${TF_TIME:-02:00:00}"
MAX_PARALLEL="${TF_MAX_PARALLEL:-50}"

# Processing parameters
TARGET_FS=128
CONTACT_MIN=1
CONTACT_MAX=5

# Parse command line arguments
EEG_FOLDER=""
OUTPUT_DIR=""
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --eeg-folder|-e)
            EEG_FOLDER="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --target-fs)
            TARGET_FS="$2"
            shift 2
            ;;
        --contact-min)
            CONTACT_MIN="$2"
            shift 2
            ;;
        --contact-max)
            CONTACT_MAX="$2"
            shift 2
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --eeg-folder, -e PATH   Path to EEG folder containing session subfolders"
            echo "  --output-dir, -o PATH   Directory to save TF map results"
            echo ""
            echo "Slurm Options:"
            echo "  --partition PART        Slurm partition (default: short)"
            echo "  --memory MEM            Memory per job (default: 16G)"
            echo "  --cpus N                CPUs per job (default: 4)"
            echo "  --time TIME             Time limit (default: 02:00:00)"
            echo "  --max-parallel N        Max concurrent jobs (default: 50)"
            echo ""
            echo "Processing Options:"
            echo "  --target-fs HZ          Target sampling frequency (default: 128)"
            echo "  --contact-min N         Minimum contact number (default: 1)"
            echo "  --contact-max N         Maximum contact number (default: 5)"
            echo "  --force, -f             Force recomputation of existing files"
            echo ""
            echo "Other:"
            echo "  --dry-run               Show what would be done without submitting"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$EEG_FOLDER" ]]; then
    echo "Error: --eeg-folder is required"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    exit 1
fi

if [[ ! -d "$EEG_FOLDER" ]]; then
    echo "Error: EEG folder does not exist: $EEG_FOLDER"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/slurm_outputs/tf_jobs"

# Session file
SESSION_FILE="$PROJECT_DIR/slurm_outputs/tf_jobs/sessions.txt"

echo "=============================================="
echo "Time-Frequency Map - Job Submission"
echo "=============================================="
echo "Project:      $PROJECT_DIR"
echo "EEG Folder:   $EEG_FOLDER"
echo "Output Dir:   $OUTPUT_DIR"
echo "Partition:    $PARTITION"
echo "Memory:       $MEMORY"
echo "CPUs:         $CPUS"
echo "Time:         $TIME_LIMIT"
echo "Target FS:    $TARGET_FS Hz"
echo "Contact Range: $CONTACT_MIN - $CONTACT_MAX"
echo ""

# Scan for sessions (folders containing .ncs files)
echo "Scanning for EEG sessions..."

> "$SESSION_FILE"  # Clear file

SKIP_COUNT=0
for session_dir in "$EEG_FOLDER"/*/; do
    if [[ -d "$session_dir" ]]; then
        # Check if folder contains .ncs files
        ncs_count=$(find "$session_dir" -maxdepth 1 -name "*.ncs" 2>/dev/null | wc -l)
        if [[ $ncs_count -gt 0 ]]; then
            session_name=$(basename "$session_dir")
            output_file="$OUTPUT_DIR/${session_name}_tf.npz"
            
            # Check if already exists (unless --force)
            if [[ -f "$output_file" && "$FORCE" == false ]]; then
                SKIP_COUNT=$((SKIP_COUNT + 1))
            else
                echo "${session_dir%/}" >> "$SESSION_FILE"
            fi
        fi
    fi
done

N_SESSIONS=$(wc -l < "$SESSION_FILE" | tr -d ' ')

echo "Found $N_SESSIONS sessions to process"
if [[ $SKIP_COUNT -gt 0 ]]; then
    echo "Skipped $SKIP_COUNT sessions (already processed, use --force to recompute)"
fi
echo ""

if [[ $N_SESSIONS -eq 0 ]]; then
    echo "No sessions to process"
    exit 0
fi

# Show sessions
echo "Sessions to process:"
head -10 "$SESSION_FILE" | while read -r line; do
    echo "  $(basename "$line")"
done
if [[ $N_SESSIONS -gt 10 ]]; then
    echo "  ... and $((N_SESSIONS - 10)) more"
fi
echo ""

# Calculate array range
MAX_INDEX=$((N_SESSIONS - 1))

# Build array job script
ARRAY_SCRIPT="$PROJECT_DIR/slurm_outputs/tf_jobs/array_job.sh"
cat > "$ARRAY_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=tf_compute
#SBATCH --partition=$PARTITION
#SBATCH --mem=$MEMORY
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME_LIMIT
#SBATCH --array=0-${MAX_INDEX}%${MAX_PARALLEL}
#SBATCH --output=$PROJECT_DIR/slurm_outputs/tf_jobs/job_%A_%a.log
#SBATCH --error=$PROJECT_DIR/slurm_outputs/tf_jobs/job_%A_%a.log

# Configuration
PROJECT_DIR="$PROJECT_DIR"
SESSION_FILE="$SESSION_FILE"
OUTPUT_DIR="$OUTPUT_DIR"
TARGET_FS=$TARGET_FS
CONTACT_MIN=$CONTACT_MIN
CONTACT_MAX=$CONTACT_MAX
FORCE_FLAG="${FORCE:+--force}"

# Read session path from file
SESSION_PATH=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\$SESSION_FILE")
SESSION_NAME=\$(basename "\$SESSION_PATH")

echo "=============================================="
echo "Job Array ID: \$SLURM_ARRAY_JOB_ID"
echo "Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Node: \$HOSTNAME"
echo "Session: \$SESSION_NAME"
echo "=============================================="
echo ""

# Load modules (adjust for your cluster)
module load python/3.12 2>/dev/null || true

# Activate virtual environment
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
fi

# Change to project directory
cd "\$PROJECT_DIR"

# Run computation
python scripts/compute_tf_single.py \\
    --session-path "\$SESSION_PATH" \\
    --output-dir "\$OUTPUT_DIR" \\
    --target-fs \$TARGET_FS \\
    --contact-min \$CONTACT_MIN \\
    --contact-max \$CONTACT_MAX \\
    --workers $CPUS \\
    \$FORCE_FLAG

echo ""
echo "Job completed at \$(date)"
EOF

echo "Generated job script: $ARRAY_SCRIPT"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN - Would submit: ==="
    echo ""
    cat "$ARRAY_SCRIPT"
    echo ""
    echo "=== End of job script ==="
else
    # Submit the job
    JOB_ID=$(sbatch --parsable "$ARRAY_SCRIPT")
    echo "=============================================="
    echo "Submitted array job: $JOB_ID"
    echo "Tasks: 0-$MAX_INDEX (${N_SESSIONS} sessions)"
    echo "Max parallel: $MAX_PARALLEL"
    echo "=============================================="
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  sacct -j $JOB_ID"
    echo ""
    echo "View logs:"
    echo "  tail -f $PROJECT_DIR/slurm_outputs/tf_jobs/job_${JOB_ID}_*.log"
    echo ""
    echo "When complete, view results with:"
    echo "  python view_tf_maps.py $OUTPUT_DIR"
fi
