"""
Generate publication figures from experiment results.

Reframed narrative: PGR optimizes reward aggressively but is safety-blind,
incurring massive constraint violations. PGR+Memory (rare-event buffer +
Lagrangian) achieves ~98% cost reduction with minimal reward trade-off.

Produces:
  Figure 1: Reward and cost learning curves (SAC, PGR, PGR+Memory)
  Figure 2: DiffHz diagnostic — synthetic hazard rate over training
  Figure 3: Cost-reward Pareto summary (bar chart)
  Table 1: Summary statistics
"""

import json
import os
import glob
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available")


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_results(results_dir, method, seeds=(42, 123, 456)):
    """Load results JSON files for a method across seeds."""
    all_results = []
    for seed in seeds:
        # Try flat structure first (new)
        flat_path = os.path.join(results_dir, f'{method}_seed{seed}_results.json')
        if os.path.exists(flat_path):
            with open(flat_path) as f:
                all_results.append(json.load(f))
            continue
        # Fallback to nested structure (old)
        pattern = os.path.join(results_dir, f'{method}_seed{seed}',
                               f'{method}_seed{seed}_results.json')
        files = glob.glob(pattern)
        if not files:
            pattern = os.path.join(results_dir, f'{method}_seed{seed}', '*_results.json')
            files = glob.glob(pattern)
        if files:
            with open(files[0]) as f:
                all_results.append(json.load(f))
    return all_results


def smooth(x, window=10):
    """Moving average smoothing."""
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode='valid')


def mean_std_across_seeds(list_of_arrays):
    """Compute mean and std across seeds, using max length with NaN padding."""
    max_len = max(len(a) for a in list_of_arrays)
    padded = np.full((len(list_of_arrays), max_len), np.nan)
    for i, a in enumerate(list_of_arrays):
        padded[i, :len(a)] = a
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    return mean, std


METHODS = [
    ('sac', 'SAC (no diffusion)', '#ff7f0e'),
    ('pgr', 'PGR', '#1f77b4'),
    ('pgr_lagrangian', 'PGR+Lagrangian', '#d62728'),
    ('pgr_buffer', 'PGR+Buffer', '#9467bd'),
    ('pgr_lb', 'PGR+L+Buffer (ours)', '#2ca02c'),
]


# ── Figure 1: Learning Curves (main result) ─────────────────────────────────

def plot_learning_curves(results_dir, seeds=(42, 123, 456),
                         save_path='figure1_curves.pdf'):
    """Reward and cumulative cost curves — the main result figure."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for method, label, color in METHODS:
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            continue

        # Rewards (smoothed)
        reward_curves = [smooth(np.array(r['episode_rewards'])) for r in results_list]
        if reward_curves:
            mean_r, std_r = mean_std_across_seeds(reward_curves)
            x = np.arange(len(mean_r))
            axes[0].plot(x, mean_r, color=color, linewidth=2, label=label)
            axes[0].fill_between(x, mean_r - std_r, mean_r + std_r,
                                color=color, alpha=0.15)

        # Cumulative costs
        cost_curves = [np.cumsum(r['episode_costs']) for r in results_list]
        if cost_curves:
            mean_c, std_c = mean_std_across_seeds(cost_curves)
            x = np.arange(len(mean_c))
            axes[1].plot(x, mean_c, color=color, linewidth=2, label=label)
            axes[1].fill_between(x, mean_c - std_c, mean_c + std_c,
                                color=color, alpha=0.15)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('(a) Reward')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cumulative Cost')
    axes[1].set_title('(b) Safety Violations (lower = safer)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure 1 saved to {save_path}')
    plt.close()


# ── Figure 2: DiffHz Diagnostic ─────────────────────────────────────────────

def plot_diffhz(results_dir, seeds=(42, 123, 456),
                save_path='figure2_diffhz.pdf'):
    """DiffHz over training — diagnostic showing what the diffusion model captures."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for method, label, color in METHODS[1:]:  # All diffusion methods (skip SAC)
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            continue

        all_steps = []
        all_rates = []
        for r in results_list:
            if 'diffhz_log' in r and r['diffhz_log']:
                steps = [x[0] for x in r['diffhz_log']]
                rates = [x[1] for x in r['diffhz_log']]
                all_steps.append(steps)
                all_rates.append(rates)

        if not all_rates:
            continue

        # Individual seeds as thin lines
        for steps, rates in zip(all_steps, all_rates):
            ax.plot(steps, rates, color=color, alpha=0.2, linewidth=0.8)

        # Mean ± std (NaN-padded to max length)
        if len(all_rates) > 1:
            max_len = max(len(r) for r in all_rates)
            padded = np.full((len(all_rates), max_len), np.nan)
            for i, r in enumerate(all_rates):
                padded[i, :len(r)] = r
            mean_rates = np.nanmean(padded, axis=0)
            std_rates = np.nanstd(padded, axis=0)
            # Use longest step grid
            longest_idx = np.argmax([len(s) for s in all_steps])
            steps_common = all_steps[longest_idx][:max_len]
            ax.plot(steps_common, mean_rates, color=color, linewidth=2.5, label=label)
            ax.fill_between(steps_common,
                           np.maximum(mean_rates - std_rates, 0),
                           mean_rates + std_rates,
                           color=color, alpha=0.15)
        else:
            ax.plot(all_steps[0], all_rates[0], color=color, linewidth=2.5, label=label)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('DiffHz (fraction cost > 0.5)')
    ax.set_title('Diffusion Model Hazard Rate Over Training')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure 2 saved to {save_path}')
    plt.close()


# ── Figure 3: Cost-Reward Summary Bar Chart ─────────────────────────────────

def plot_summary_bars(results_dir, seeds=(42, 123, 456),
                      save_path='figure3_summary.pdf'):
    """Bar chart: reward vs cost trade-off across methods."""
    if not HAS_MPL:
        return

    labels = []
    mean_rewards, std_rewards = [], []
    mean_costs, std_costs = [], []

    for method, label, color in METHODS:
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            continue
        labels.append(label)
        rews = [np.mean(r['episode_rewards'][-50:]) for r in results_list]
        costs = [np.mean(r['episode_costs'][-50:]) for r in results_list]
        mean_rewards.append(np.mean(rews))
        std_rewards.append(np.std(rews))
        mean_costs.append(np.mean(costs))
        std_costs.append(np.std(costs))

    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c'][:len(labels)]
    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1, 1]})

    # Left: Reward bars
    bars1 = ax1.bar(x, mean_rewards, 0.6, yerr=std_rewards,
                    color=colors, alpha=0.85, edgecolor='black',
                    linewidth=0.5, capsize=5)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('(a) Reward (last 50 eps)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim(bottom=0)
    # Annotate values
    for bar, mr, sr in zip(bars1, mean_rewards, std_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sr + 10,
                f'{mr:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right: Cost bars
    bars2 = ax2.bar(x, mean_costs, 0.6, yerr=std_costs,
                    color=colors, alpha=0.85, edgecolor='black',
                    linewidth=0.5, capsize=5, hatch='//')
    ax2.set_ylabel('Episode Cost')
    ax2.set_title('(b) Cost (last 50 eps, lower = safer)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_ylim(bottom=0)
    # Annotate values
    for bar, mc, sc in zip(bars2, mean_costs, std_costs):
        y_pos = bar.get_height() + sc + max(mean_costs) * 0.03
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{mc:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure 3 saved to {save_path}')
    plt.close()


# ── Table 1: Summary Statistics ──────────────────────────────────────────────

def print_summary_table(results_dir, seeds=(42, 123, 456)):
    """Print summary table with the reframed metrics."""

    print('\n' + '=' * 90)
    print('Table 1: Summary Statistics (mean +/- std across seeds, last 50 episodes)')
    print('=' * 90)
    print(f'{"Method":<22} {"Reward":>16} {"Ep Cost":>16} {"Total Cost":>14} {"DiffHz":>14}')
    print('-' * 90)

    for method, label, _ in METHODS:
        results_list = load_results(results_dir, method, seeds)
        if not results_list:
            print(f'{label:<22} {"N/A":>16} {"N/A":>16} {"N/A":>14} {"N/A":>14}')
            continue

        # Use last 50 episodes for final stats
        rewards = [np.mean(r['episode_rewards'][-50:]) for r in results_list]
        ep_costs = [np.mean(r['episode_costs'][-50:]) for r in results_list]
        total_costs = [sum(r['episode_costs']) for r in results_list]

        diffhz_vals = []
        for r in results_list:
            if 'diffhz_log' in r and r['diffhz_log']:
                diffhz_vals.append(r['diffhz_log'][-1][1])

        mean_r, std_r = np.mean(rewards), np.std(rewards)
        mean_ec, std_ec = np.mean(ep_costs), np.std(ep_costs)
        mean_tc, std_tc = np.mean(total_costs), np.std(total_costs)

        if diffhz_vals:
            mean_hz = np.mean(diffhz_vals)
            std_hz = np.std(diffhz_vals)
            hz_str = f'{mean_hz:.1%}+/-{std_hz:.1%}'
        else:
            hz_str = 'N/A'

        print(f'{label:<22} {mean_r:>7.1f}+/-{std_r:>5.1f} '
              f'{mean_ec:>7.1f}+/-{std_ec:>5.1f} '
              f'{mean_tc:>6.0f}+/-{std_tc:>5.0f} '
              f'{hz_str:>14}')

    print('=' * 90)

    # Cost reduction callout
    pgr = load_results(results_dir, 'pgr', seeds)
    mem = load_results(results_dir, 'pgr_lb', seeds)
    if not mem:
        mem = load_results(results_dir, 'pgr_memory', seeds)  # backwards compat
    if pgr and mem:
        pgr_cost = np.mean([np.mean(r['episode_costs'][-50:]) for r in pgr])
        mem_cost = np.mean([np.mean(r['episode_costs'][-50:]) for r in mem])
        pgr_rew = np.mean([np.mean(r['episode_rewards'][-50:]) for r in pgr])
        mem_rew = np.mean([np.mean(r['episode_rewards'][-50:]) for r in mem])
        if pgr_cost > 0:
            reduction = (1 - mem_cost / pgr_cost) * 100
            rew_diff = (mem_rew - pgr_rew) / pgr_rew * 100
            print(f'\nPGR+Memory vs PGR: {reduction:.1f}% cost reduction, '
                  f'{rew_diff:+.1f}% reward change')
    print()


# ── Paper figures (ICML 2026 workshop submission) ──────────────────────────

_PAPER_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.6,
}


def plot_paper_fig_diffhz_cheetah(results_dir, seeds=(42, 123, 456),
                                   save_path='figure_diffhz_cheetah.pdf'):
    """
    Paper Fig. 1: DiffHz trajectory on Cheetah-run.

    Three lines (PGR baseline, PGR+L, PGR+L+Buffer), mean +/- std across 3 seeds,
    ten probe points at 10K-step intervals. Visually anchors the headline
    13.3% -> 0.6% -> 8.6% collapse-and-recovery story.
    """
    if not HAS_MPL:
        return

    paper_methods = [
        ('pgr',            r'PGR (unconstrained)',      '#1f77b4', '-',  'o'),
        ('pgr_lagrangian', r'PGR+Lagrangian',           '#d62728', '--', 's'),
        ('pgr_lb',         r'PGR+L+Buffer (ours)',      '#2ca02c', '-',  'D'),
    ]

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.35, 2.35))

        for method, label, color, ls, marker in paper_methods:
            results_list = load_results(results_dir, method, seeds)
            if not results_list:
                continue

            # Stack per-seed DiffHz arrays onto the common 10-step grid
            per_seed = []
            step_grid = None
            for r in results_list:
                log = r.get('diffhz_log') or []
                if not log:
                    continue
                steps = np.array([x[0] for x in log], dtype=float)
                rates = np.array([x[1] for x in log], dtype=float) * 100  # %
                per_seed.append(rates)
                step_grid = steps
            if not per_seed or step_grid is None:
                continue

            arr = np.stack(per_seed, axis=0)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=0)

            ax.plot(step_grid / 1000, mean, color=color, linestyle=ls,
                    marker=marker, markersize=3.5, label=label,
                    markeredgewidth=0, zorder=3)
            ax.fill_between(step_grid / 1000,
                            np.maximum(mean - std, 0),
                            mean + std,
                            color=color, alpha=0.18, linewidth=0, zorder=2)

        ax.set_xlabel(r'Environment steps ($\times 10^{3}$)')
        ax.set_ylabel(r'DiffHz (%)')
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=-0.5)
        ax.grid(True, which='major', alpha=0.25, linewidth=0.5)
        ax.legend(loc='upper left', frameon=False, handlelength=2.2,
                  borderaxespad=0.3)

        # Light de-emphasis of top/right spines
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

        plt.tight_layout(pad=0.3)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
        print(f'Paper Fig. DiffHz-Cheetah saved to {save_path}')
        plt.close()


def plot_paper_fig_feedback_loop(save_path='figure_feedback_loop.pdf'):
    """
    Paper Fig. 2: Hazard-compression feedback-loop diagram.

    Vertical flow with a loopback arrow, covering the five-step mechanism from
    Section 3.4 (i -> ii -> iii -> iv -> v -> i). The (ii)->(iii) edge is
    highlighted as the intervention point for the rare-event buffer.
    """
    if not HAS_MPL:
        return

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    steps = [
        ('(i)',   'Lagrangian penalty\n' r'suppresses $c$ in reward'),
        ('(ii)',  'Hazardous transitions\n' r'vanish from replay buffer'),
        ('(iii)', 'Diffusion retrains;\n' r'DiffHz collapses'),
        ('(iv)',  r'$Q$-networks miscalibrate''\n' 'near constraint boundary'),
        ('(v)',   r'$\lambda$ surges''\n' 'to compensate'),
    ]

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.35, 3.85))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        box_w, box_h = 5.4, 1.35
        x_center = 4.0
        y_positions = [10.5, 8.5, 6.5, 4.5, 2.5]

        for (tag, text), y in zip(steps, y_positions):
            xb = x_center - box_w / 2
            yb = y - box_h / 2
            box = FancyBboxPatch(
                (xb, yb), box_w, box_h,
                boxstyle='round,pad=0.08,rounding_size=0.18',
                linewidth=0.9, edgecolor='#222222', facecolor='#f4f4f6',
                zorder=2,
            )
            ax.add_patch(box)
            ax.text(xb + 0.22, y, tag, ha='left', va='center',
                    fontsize=9, fontweight='bold', color='#222222', zorder=3)
            ax.text(x_center + 0.35, y, text, ha='center', va='center',
                    fontsize=8.2, color='#222222', zorder=3)

        # Downward arrows (i -> ii -> iii -> iv -> v)
        for y_top, y_bot in zip(y_positions[:-1], y_positions[1:]):
            arrow = FancyArrowPatch(
                (x_center, y_top - box_h / 2 - 0.05),
                (x_center, y_bot + box_h / 2 + 0.05),
                arrowstyle='-|>', mutation_scale=10,
                color='#222222', linewidth=0.9, zorder=1,
            )
            ax.add_patch(arrow)

        # Intervention callout on (ii) -> (iii) edge.
        # Text is placed strictly to the RIGHT of the box column (right edge at
        # x_center + box_w/2) with ha='left' so the text block anchors past the
        # box edge and cannot overlap it.
        y_top = y_positions[1]
        y_bot = y_positions[2]
        edge_mid_y = (y_top - box_h / 2 + y_bot + box_h / 2) / 2
        ax.annotate(
            'rare-event\nbuffer\nintervenes',
            xy=(x_center + 0.06, edge_mid_y),              # arrow tip: inner edge
            xytext=(x_center + box_w / 2 + 0.45,           # text left-edge: past boxes
                    edge_mid_y),
            fontsize=7.5, color='#b71c1c', ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#b71c1c',
                            linewidth=0.9, shrinkA=2, shrinkB=2),
            zorder=4,
        )

        # Loopback: right-angle path from (v) back up to (i) along the right rail.
        # Drawn as three segments + an arrowhead on the final leg back into (i).
        x_right_edge = x_center + box_w / 2
        # Rail is offset enough to clear the rare-event callout text to its left.
        x_rail = x_right_edge + 1.8
        y_v = y_positions[-1]
        y_i = y_positions[0]
        loop_color = '#1565c0'

        # Segment 1: out of (v) to the right rail
        ax.plot([x_right_edge, x_rail], [y_v, y_v],
                color=loop_color, linewidth=1.1, zorder=1)
        # Segment 2: up the rail
        ax.plot([x_rail, x_rail], [y_v, y_i],
                color=loop_color, linewidth=1.1, zorder=1)
        # Segment 3: rail back into (i), with arrowhead
        loop_head = FancyArrowPatch(
            (x_rail, y_i), (x_right_edge, y_i),
            arrowstyle='-|>', mutation_scale=11,
            color=loop_color, linewidth=1.1, zorder=1,
        )
        ax.add_patch(loop_head)
        # Label near the top of the rail so it doesn't collide with the (ii)->(iii) callout
        ax.text(x_rail + 0.08, (y_i + y_v) / 2,
                'closes\nloop', rotation=90, fontsize=7.5,
                color=loop_color, ha='left', va='center')
        # Nudge xlim so the rail + label fit
        ax.set_xlim(0, x_rail + 1.0)

        plt.tight_layout(pad=0.1)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
        print(f'Paper Fig. feedback-loop saved to {save_path}')
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./safety_results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--paper', action='store_true',
                        help='Generate paper figures only (Fig. DiffHz-Cheetah, Fig. feedback-loop)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = tuple(args.seeds)

    if args.paper:
        if HAS_MPL:
            plot_paper_fig_diffhz_cheetah(
                args.results_dir, seeds,
                os.path.join(args.output_dir, 'fig_diffhz_cheetah.pdf'))
            plot_paper_fig_feedback_loop(
                os.path.join(args.output_dir, 'fig_feedback_loop.pdf'))
        else:
            print('Install matplotlib for figures: pip install matplotlib')
        raise SystemExit

    print_summary_table(args.results_dir, seeds)

    if HAS_MPL:
        plot_learning_curves(args.results_dir, seeds,
                            os.path.join(args.output_dir, 'figure1_curves.pdf'))
        plot_diffhz(args.results_dir, seeds,
                    os.path.join(args.output_dir, 'figure2_diffhz.pdf'))
        plot_summary_bars(args.results_dir, seeds,
                         os.path.join(args.output_dir, 'figure3_summary.pdf'))
    else:
        print("Install matplotlib for figures: pip install matplotlib")
