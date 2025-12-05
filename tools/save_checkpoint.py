#!/usr/bin/env python
"""
Save checkpoint for a running training process.

Usage:
    python tools/save_checkpoint.py           # Auto-find training process
    python tools/save_checkpoint.py <pid>     # Specify PID manually

This sends SIGUSR1 signal to the training process, which triggers
the signal handler in train.py to save a checkpoint.
"""

import os
import sys
import signal
import subprocess


def find_training_pid():
    """Find the PID of the running train.py process"""
    try:
        # Try to find process with pgrep
        result = subprocess.run(
            ['pgrep', '-f', 'python.*train.py'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            # Filter out this script's own PID
            my_pid = str(os.getpid())
            pids = [p for p in pids if p and p != my_pid]
            if pids:
                return pids
    except FileNotFoundError:
        pass

    # Fallback: use ps
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        pids = []
        for line in result.stdout.split('\n'):
            if 'python' in line and 'train.py' in line and 'save_checkpoint' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pids.append(parts[1])
        if pids:
            return pids
    except Exception:
        pass

    return []


def send_save_signal(pid):
    """Send SIGUSR1 to the training process"""
    try:
        os.kill(int(pid), signal.SIGUSR1)
        print(f"[OK] Sent SIGUSR1 to process {pid}")
        print(f"     Check the training terminal for checkpoint save confirmation.")
        return True
    except ProcessLookupError:
        print(f"[ERROR] Process {pid} not found")
        return False
    except PermissionError:
        print(f"[ERROR] Permission denied to send signal to process {pid}")
        print(f"        Try running with sudo: sudo python tools/save_checkpoint.py {pid}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send signal: {e}")
        return False


def main():
    print("=" * 50)
    print("OpenSceneFlow - Manual Checkpoint Saver")
    print("=" * 50)

    if len(sys.argv) > 1:
        # PID provided as argument
        pid = sys.argv[1]
        print(f"\nUsing provided PID: {pid}")
        send_save_signal(pid)
    else:
        # Auto-find training process
        print("\nSearching for running train.py process...")
        pids = find_training_pid()

        if not pids:
            print("[ERROR] No training process found!")
            print("\nMake sure train.py is running. You can also specify PID manually:")
            print("    python tools/save_checkpoint.py <pid>")
            print("\nTo find the PID manually:")
            print("    ps aux | grep train.py")
            sys.exit(1)

        if len(pids) == 1:
            print(f"Found training process: PID {pids[0]}")
            send_save_signal(pids[0])
        else:
            print(f"Found multiple training processes: {pids}")
            print("\nSending signal to all of them...")
            for pid in pids:
                send_save_signal(pid)

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
