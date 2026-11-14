# More tokens per micro-batch at the cost of CPU teacher transfers; retains the same global batch 128.
source "$(dirname "${BASH_SOURCE[0]}")/balanced.env.sh"
export MAX_TOKEN_LEN_PER_GPU=36864 TEACHER_CPU_OFFLOAD=true
