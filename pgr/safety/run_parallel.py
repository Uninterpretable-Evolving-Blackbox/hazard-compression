#!/usr/bin/env python3
"""
Parallel experiment runner for PGR Safety experiments.

Auto-detects hardware (CUDA/MPS/CPU) and maximizes parallelism.
Skips already-completed runs. Resumes gracefully.

Usage:
    # Auto-detect everything
    python safety/run_parallel.py

    # Override parallelism
    python safety/run_parallel.py --max_parallel 4

    # Quick test (1 epoch, 1 seed)
    python safety/run_parallel.py --epochs 1 --seeds 42 --envs cheetah-run-v0

    # Full run
    python safety/run_parallel.py --epochs 500 --results_dir ./safety_results
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ── Hardware detection ────────────────────────────────────────────────────────

def detect_hardware():
    """Return (device, max_parallel, threads_per_worker, description)."""
    try:
        import torch
    except ImportError:
        cpu_count = os.cpu_count() or 4
        return 'cpu', max(1, cpu_count // 2), 2, f'CPU ({cpu_count} cores, no torch)'

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # PGR uses ~4-6GB per run on GPU; SAC ~2GB
        runs_per_gpu = max(1, int(gpu_mem_gb / 8))
        max_parallel = gpu_count * runs_per_gpu
        desc = f'CUDA ({gpu_name}, {gpu_mem_gb:.0f}GB x {gpu_count})'
        return 'cuda', max_parallel, 4, desc

    cpu_count = os.cpu_count() or 4

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS + CPU hybrid: cap MPS workers to avoid GPU contention; route
        # non-diffusion (SAC) runs to CPU. See MAX_MPS_WORKERS below.
        max_parallel = min(9, max(1, cpu_count // 2))
        desc = f'MPS GPU + {cpu_count} CPU cores (hybrid)'
        return 'hybrid', max_parallel, 2, desc

    threads = 2
    max_parallel = max(1, cpu_count // threads)
    desc = f'CPU ({cpu_count} cores)'
    return 'cpu', max_parallel, threads, desc


# ── Live progress parsing (for the worker dashboard) ─────────────────────────

def _parse_run_progress(log_path, total_env_steps):
    """
    Parse a worker's log file to extract:
      - latest env_step (from the most recent epoch tabular dump)
      - whether we are currently INSIDE a diffusion retrain
      - if so, how far through it (parsed from `[N/M] loss` lines)

    Returns None if the log doesn't exist or is empty.
    """
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except IOError:
        return None
    if not content.strip():
        return None

    env_matches = re.findall(r'TotalEnvInteracts\s*\|\s*([\d.eE+\-]+)', content)
    env_step = int(float(env_matches[-1])) if env_matches else 0

    # We are mid-retrain iff the latest 'Retraining diffusion model' line is
    # AFTER the latest epoch-dump end marker (a row of dashes).
    last_retrain_pos = content.rfind('Retraining diffusion model at step')
    last_epoch_end_pos = content.rfind('---------------------------------------')
    in_diffusion = (last_retrain_pos != -1 and last_retrain_pos > last_epoch_end_pos)

    diff_progress = None
    if in_diffusion:
        diff_section = content[last_retrain_pos:]
        diff_matches = re.findall(r'\[(\d+)/(\d+)\]\s*loss', diff_section)
        if diff_matches:
            d_now, d_max = int(diff_matches[-1][0]), int(diff_matches[-1][1])
            diff_progress = (d_now, d_max)

    return {
        'env_step': env_step,
        'env_pct': 100.0 * env_step / max(1, total_env_steps),
        'in_diffusion': in_diffusion,
        'diff_progress': diff_progress,
    }


def _format_duration(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN guard
        return '?'
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f'{h}h{m:02d}m'


def _print_status_dashboard(active, log_dir, total_env_steps, pending_count,
                            wall_start):
    """Print a one-block status dashboard for all active workers."""
    now = time.time()
    print()
    print(f'[{time.strftime("%H:%M:%S")}] === Worker dashboard '
          f'(wall {_format_duration(now - wall_start)}, '
          f'{len(active)} active, {pending_count} queued) ===')
    items = sorted(active.items(), key=lambda kv: kv[1][1]['label'])
    for pid, (proc, exp, t0, lf, on_mps) in items:
        elapsed = now - t0
        log_path = os.path.join(log_dir,
                                f"{exp['label'].replace('/', '_')}.log")
        prog = _parse_run_progress(log_path, total_env_steps)
        label = exp['label']
        if prog is None or prog['env_step'] == 0:
            print(f"  {label:42s}  startup            "
                  f"elap={_format_duration(elapsed)}")
            continue

        if prog['in_diffusion']:
            if prog['diff_progress']:
                d_now, d_max = prog['diff_progress']
                phase_str = f"diff[{d_now:>6}/{d_max}]"
            else:
                phase_str = "diff[starting]"
        else:
            phase_str = "training"

        # Naive linear ETA based on env-step rate. Slightly underestimates
        # because diffusion retrains add overhead — refine after first retrain.
        eta_str = '?'
        if elapsed > 30 and prog['env_step'] > 0:
            rate = prog['env_step'] / elapsed
            if rate > 0:
                eta_str = _format_duration(
                    (total_env_steps - prog['env_step']) / rate)

        print(f"  {label:42s}  step={prog['env_step']:>6}/{total_env_steps} "
              f"({prog['env_pct']:5.1f}%)  {phase_str:20s}  "
              f"elap={_format_duration(elapsed)}  ETA={eta_str}")
    if pending_count > 0:
        print(f"  ... {pending_count} more queued")
    print()
    sys.stdout.flush()


# ── Experiment generation ─────────────────────────────────────────────────────

ENV_DEFAULTS = {
    'cheetah-run-v0': '--velocity_threshold 7.0',
    'walker-walk-v0': '--velocity_threshold 3.0 --height_threshold 1.0',
}

METHOD_CONFIGS = {
    'sac': {
        'gin': 'config/online/sac.gin',
        'gin_params': 'redq_sac.disable_diffusion = True',
        'flags': '--sac_only',
    },
    'pgr': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': '',
    },
    'pgr_lagrangian': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': '--use_lagrangian',
    },
    'pgr_buffer': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': '--use_rare_buffer',
    },
    'pgr_lb': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': '--use_rare_buffer --use_lagrangian',
    },
    # ── Anti-windup variants for the Walker integral-windup study ──
    # All three include rare-event buffer + Lagrangian; differ only in
    # how the lambda update is regularized to defuse integral windup.
    'pgr_lb_warmup': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': ('--use_rare_buffer --use_lagrangian '
                  '--lambda_warmup_episodes 20 '
                  '--method_name pgr_lb_warmup'),
    },
    'pgr_lb_clip': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': ('--use_rare_buffer --use_lagrangian '
                  '--lambda_grad_clip 0.1 '
                  '--method_name pgr_lb_clip'),
    },
    'pgr_lb_warmclip': {
        'gin': 'config/online/sac_cond_synther_dmc.gin',
        'gin_params': 'redq_sac.cond_top_frac = 0.25',
        'flags': ('--use_rare_buffer --use_lagrangian '
                  '--lambda_warmup_episodes 20 '
                  '--lambda_grad_clip 0.1 '
                  '--method_name pgr_lb_warmclip'),
    },
}


def is_run_complete(result_file):
    """Check if a result file exists and is marked complete."""
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        return data.get('complete', False)
    except (json.JSONDecodeError, KeyError):
        return False


def generate_experiments(envs, methods, seeds, epochs, results_dir):
    """Generate list of experiment configs, skipping completed runs."""
    experiments = []
    skipped = 0

    for env in envs:
        env_flags = ENV_DEFAULTS.get(env, '')
        env_results = os.path.join(results_dir, env)
        os.makedirs(env_results, exist_ok=True)

        for method_name in methods:
            mc = METHOD_CONFIGS[method_name]

            for seed in seeds:
                result_file = os.path.join(env_results,
                                           f'{method_name}_seed{seed}_results.json')

                if is_run_complete(result_file):
                    skipped += 1
                    continue

                gin_params = [f'redq_sac.epochs = {epochs}',
                              mc['gin_params']]
                cmd_parts = [
                    sys.executable, 'safety/online_cost_cond.py',
                    '--env', env,
                    '--seed', str(seed),
                    '--gin_config_files', mc['gin'],
                    '--gin_params'] + gin_params + [
                    '--results_folder', env_results,
                ]
                # Add env-specific flags
                if env_flags:
                    cmd_parts.extend(env_flags.split())
                # Add method-specific flags
                if mc['flags']:
                    cmd_parts.extend(mc['flags'].split())

                experiments.append({
                    'cmd': cmd_parts,
                    'env': env,
                    'method': method_name,
                    'seed': seed,
                    'result_file': result_file,
                    'label': f'{env}/{method_name}/seed{seed}',
                })

    return experiments, skipped


# ── Run a single experiment ───────────────────────────────────────────────────

def run_single(experiment, device, threads_per_worker, log_dir):
    """Run one experiment as a subprocess. Returns (label, success, duration)."""
    label = experiment['label']
    log_file = os.path.join(log_dir, f'{label.replace("/", "_")}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads_per_worker)
    env['MKL_NUM_THREADS'] = str(threads_per_worker)
    env['TORCH_NUM_THREADS'] = str(threads_per_worker)
    # Set device for each worker
    env['PGR_DEVICE'] = device

    start = time.time()
    try:
        with open(log_file, 'w') as lf:
            result = subprocess.run(
                experiment['cmd'],
                stdout=lf, stderr=subprocess.STDOUT,
                env=env, timeout=7200,  # 2 hour timeout per run
            )
        duration = time.time() - start
        success = result.returncode == 0
        return label, success, duration
    except subprocess.TimeoutExpired:
        return label, False, time.time() - start
    except Exception as e:
        return label, False, time.time() - start


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Parallel PGR Safety experiment runner')
    parser.add_argument('--envs', nargs='+',
                        default=['cheetah-run-v0', 'walker-walk-v0'],
                        help='Environments to evaluate')
    parser.add_argument('--methods', nargs='+',
                        default=['sac', 'pgr', 'pgr_lagrangian',
                                 'pgr_buffer', 'pgr_lb'],
                        help='Methods to run')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 43, 44, 45, 46],
                        help='Random seeds (5 seeds: C(10,5)=252 perms, min p=0.0039)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs (x1000 = steps)')
    parser.add_argument('--results_dir', type=str, default='./safety_results',
                        help='Results directory')
    parser.add_argument('--max_parallel', type=int, default=None,
                        help='Max concurrent runs (auto-detect if not set)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print experiments without running')

    args = parser.parse_args()

    # Detect hardware
    device, auto_parallel, threads, hw_desc = detect_hardware()
    max_parallel = args.max_parallel or auto_parallel

    print('=' * 70)
    print('PGR Safety Experiments — Parallel Runner')
    print('=' * 70)
    print(f'Hardware:      {hw_desc}')
    print(f'Device:        {device}')
    print(f'Max parallel:  {max_parallel}')
    print(f'Threads/worker:{threads}')
    print(f'Environments:  {args.envs}')
    print(f'Methods:       {args.methods}')
    print(f'Seeds:         {args.seeds}')
    print(f'Epochs:        {args.epochs} ({args.epochs}K steps)')
    print(f'Results:       {args.results_dir}')

    # Generate experiments
    experiments, skipped = generate_experiments(
        args.envs, args.methods, args.seeds, args.epochs, args.results_dir)

    total = len(experiments) + skipped
    print(f'\nTotal runs:    {total}')
    print(f'Already done:  {skipped}')
    print(f'To run:        {len(experiments)}')

    if not experiments:
        print('\nAll experiments complete!')
        return

    if args.dry_run:
        print('\n--- DRY RUN ---')
        for exp in experiments:
            print(f'  {exp["label"]}')
            print(f'    {" ".join(exp["cmd"][:6])} ...')
        print(f'\nWould run {len(experiments)} experiments, '
              f'{max_parallel} at a time')
        return

    # Run
    log_dir = os.path.join(args.results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    print(f'\nStarting {len(experiments)} runs, {max_parallel} in parallel...')
    print('-' * 70)

    completed = 0
    failed = 0
    start_time = time.time()

    # Direct Popen management (avoids ProcessPoolExecutor + MPS conflicts)
    active = {}  # pid -> (process, experiment, start_time, log_file_handle)
    pending = list(experiments)

    mps_count = [0]  # track active MPS workers
    MAX_MPS_WORKERS = 3

    def needs_mps(method):
        """PGR methods use diffusion (accelerate) which must stay on MPS."""
        return method != 'sac'

    def can_launch(exp):
        """Check if we have a slot for this experiment."""
        if len(active) >= max_parallel:
            return False
        if device == 'hybrid' and needs_mps(exp['method']):
            return mps_count[0] < MAX_MPS_WORKERS
        return True

    def launch(exp):
        label = exp['label']
        log_path = os.path.join(log_dir, f'{label.replace("/", "_")}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        proc_env = os.environ.copy()
        proc_env['OMP_NUM_THREADS'] = str(threads)
        proc_env['MKL_NUM_THREADS'] = str(threads)
        proc_env['TORCH_NUM_THREADS'] = str(threads)
        # Force unbuffered stdout so diffusion training progress prints in real-time
        proc_env['PYTHONUNBUFFERED'] = '1'
        use_mps = False
        if device == 'hybrid':
            if needs_mps(exp['method']):
                # PGR methods MUST use MPS (accelerate auto-detects it)
                use_mps = True
                mps_count[0] += 1
                proc_env['PGR_DEVICE'] = 'mps'
            else:
                # SAC: force CPU and disable MPS detection
                proc_env['PGR_DEVICE'] = 'cpu'
                proc_env['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        else:
            proc_env['PGR_DEVICE'] = device
        lf = open(log_path, 'w')
        proc = subprocess.Popen(exp['cmd'], stdout=lf, stderr=subprocess.STDOUT, env=proc_env)
        active[proc.pid] = (proc, exp, time.time(), lf, use_mps)

    # Fill initial slots — respect MPS cap
    for exp in list(pending):
        if can_launch(exp):
            pending.remove(exp)
            launch(exp)

    # Dashboard cadence: print a per-worker status block every STATUS_INTERVAL
    # seconds while the poll loop is running.
    STATUS_INTERVAL = 60
    last_status_print = time.time()
    total_env_steps = args.epochs * 1000

    # Poll loop
    while active:
        time.sleep(2)

        # Periodic live status dashboard
        now_t = time.time()
        if now_t - last_status_print >= STATUS_INTERVAL:
            last_status_print = now_t
            _print_status_dashboard(active, log_dir, total_env_steps,
                                    len(pending), start_time)

        finished_pids = []
        for pid, (proc, exp, t0, lf, on_mps) in active.items():
            ret = proc.poll()
            if ret is not None:
                finished_pids.append(pid)
                lf.close()
                if on_mps:
                    mps_count[0] -= 1
                duration = time.time() - t0
                completed += 1
                success = (ret == 0)
                if not success:
                    failed += 1
                dev_tag = 'MPS' if on_mps else 'CPU'
                status = 'DONE' if success else 'FAIL'
                remaining = len(pending) + len(active) - len(finished_pids)
                if completed > 0:
                    eta = (time.time() - start_time) / completed * remaining
                    eta_str = f'{eta/60:.0f}min'
                else:
                    eta_str = '?'
                print(f'[{completed}/{len(experiments)}] {status} '
                      f'{exp["label"]} [{dev_tag}] ({duration/60:.1f}min) '
                      f'ETA: {eta_str}')

        for pid in finished_pids:
            del active[pid]

        # Fill freed slots — respect MPS cap
        for exp in list(pending):
            if can_launch(exp):
                pending.remove(exp)
                launch(exp)

    # Summary
    elapsed = time.time() - start_time
    print()
    print('=' * 70)
    print(f'COMPLETE: {completed - failed}/{len(experiments)} succeeded, '
          f'{failed} failed')
    print(f'Total time: {elapsed/3600:.1f} hours')
    print(f'Results in: {args.results_dir}')
    if failed > 0:
        print(f'\nRe-run to retry failed experiments (completed runs are skipped).')
    print('=' * 70)


if __name__ == '__main__':
    main()
