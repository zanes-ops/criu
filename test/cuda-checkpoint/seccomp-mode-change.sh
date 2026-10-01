#!/bin/sh

set -eu

# Verify that criu dump refuses a CUDA task that uses seccomp in ways that
# CRIU cannot handle, with both backends, and that the CUDA plugin then
# rolls the CUDA state back, so that the task keeps running with its
# filter.
#
# In the "window" case, the target installs a filter during the mocked
# checkpoint action, while the CUDA plugin lets its restore thread run.
# CRIU decides from the seccomp mode that it collected when it seized a
# thread whether to suspend seccomp for the parasite and whether to dump
# the thread's seccomp filters, so it must refuse the dump.
#
# In the "preinstalled" case, the target installs the filter before the
# dump. Without CAP_SYS_ADMIN in the initial user namespace, the kernel
# refuses to suspend seccomp, so CRIU must fail to seize the task, which
# the CUDA plugin has locked by then.
#
# CRIU runs with --unprivileged as root of a new user namespace, like
# unprivileged.sh; no GPU is required. When CRIU can suspend seccomp, for
# example as root in CI, the "window" case also runs without a user
# namespace and without --unprivileged: CRIU must refuse the dump there as
# well, because it would not dump the new filter. The "preinstalled" case
# needs a tracer without CAP_SYS_ADMIN, so it runs only in the user
# namespace.

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
CRIU="$ROOT/criu/criu"
PLUGIN_DIR="$ROOT/plugins/cuda"
MOCK_DIR="$ROOT/test/cuda-checkpoint"

# Runs as pid 1 of a new pid and mount namespace, created by run_case().
# The remaining arguments are passed to criu dump.
if [ "${1:-}" = "--in-ns" ]; then
	WORK_DIR=$2
	BACKEND=$3
	CASE=$4
	shift 4
	if [ "$CASE" = preinstalled ]; then
		# The target installs its filter as soon as the API marker records
		# a checkpoint action.
		echo "checkpoint 0 0" >"$WORK_DIR/api"
	fi
	setsid "$MOCK_DIR/seccomp-mode-change" "$WORK_DIR/api" "$WORK_DIR/filtered" \
		</dev/null >/dev/null 2>&1 &
	TARGET_PID=$!
	if [ "$CASE" = preinstalled ]; then
		ATTEMPTS=0
		while [ ! -e "$WORK_DIR/filtered" ]; do
			ATTEMPTS=$((ATTEMPTS + 1))
			if [ "$ATTEMPTS" -eq 300 ]; then
				echo "The target did not install its filter"
				exit 1
			fi
			sleep 0.1
		done
	fi
	if timeout 30s env \
		CRIU_FAULT=138 \
		CRIU_CUDA_MOCK_STATE_FILE="$WORK_DIR/state" \
		CRIU_CUDA_MOCK_API_MARKER="$WORK_DIR/api" \
		CRIU_CUDA_MOCK_CHECKPOINT_WAIT="$WORK_DIR/filtered" \
		PATH="$MOCK_DIR:$PATH" \
		LD_LIBRARY_PATH="$MOCK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
		"$CRIU" dump --tree "$TARGET_PID" \
		--no-default-config \
		--images-dir "$WORK_DIR" \
		--log-file dump.log \
		--verbosity=4 \
		--libdir "$PLUGIN_DIR" \
		--plugin-option=cuda_plugin.backend="$BACKEND" "$@"; then
		echo "criu dump did not fail"
		exit 1
	fi
	LOG="$WORK_DIR/dump.log"
	if [ "$CASE" = preinstalled ]; then
		if ! grep -q "suspending seccomp failed: Operation not permitted" "$LOG"; then
			echo "criu dump did not fail to suspend seccomp"
			exit 1
		fi
	elif ! grep -q "Seccomp mode of thread $TARGET_PID changed from 0 to 2" "$LOG"; then
		echo "criu dump did not detect the seccomp mode change"
		exit 1
	fi
	if ! grep -q "resuming devices on pid $TARGET_PID" "$LOG" ||
	   grep -q "Unable to restore CUDA state" "$LOG"; then
		echo "CUDA plugin did not roll the CUDA state back"
		exit 1
	fi
	if ! kill -0 "$TARGET_PID" 2>/dev/null; then
		echo "criu dump left the target dead"
		exit 1
	fi
	if [ "$(awk '/^Seccomp:/ { print $2 }' "/proc/$TARGET_PID/status")" != "2" ]; then
		echo "The target lost its seccomp filter"
		exit 1
	fi
	kill "$TARGET_PID"
	exit 0
fi

WORK_DIR=$(mktemp -d)
cleanup()
{
	status=$?
	# Keep CRIU logs from a failed run for inspection.
	if [ "$status" -eq 0 ]; then
		rm -rf "$WORK_DIR"
	else
		echo "Keeping $WORK_DIR" >&2
	fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

make -C "$ROOT" cuda_plugin
make -C "$MOCK_DIR"

if ! unshare -Ur true 2>/dev/null; then
	echo "SKIP: user namespaces are not available"
	exit 0
fi

# Runs a case with both backends, in namespaces created with the given
# unshare flags. The remaining arguments are passed to criu dump.
run_case()
{
	FLAGS=$1
	CASE=$2
	shift 2
	for BACKEND in driver-api cuda-checkpoint; do
		DIR="$WORK_DIR/$BACKEND-$CASE$FLAGS"
		mkdir "$DIR"
		if ! unshare "$FLAGS" --mount-proc "$0" --in-ns "$DIR" "$BACKEND" "$CASE" "$@"; then
			grep -h "Error" "$DIR"/*.log >&2 || true
			echo "CUDA mock seccomp $CASE test (unshare $FLAGS) failed with the $BACKEND backend"
			exit 1
		fi
	done
}

run_case -Urpf window --unprivileged
run_case -Urpf preinstalled --unprivileged

if "$CRIU" check --no-default-config --feature seccomp_suspend >/dev/null 2>&1; then
	echo "CRIU can suspend seccomp: running the window case as root too"
	run_case -pf window
else
	echo "CRIU cannot suspend seccomp: not running the window case as root"
fi

echo "CUDA mock seccomp mode change PASS"
