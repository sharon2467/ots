#!/usr/bin/env python3
"""Plot a frozen analyze_depth.py summary; never open predictions or checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ('D2', 'D3', 'W3', 'D4', 'W4')
PARAMETERS = dict(D2=3251680, D3=4303328, W3=4304576, D4=5354976, W4=5354941)
DEPTH = dict(D2=2, D3=3, W3=2, D4=4, W4=2)
SEEDS = (20261030, 20261032, 20261033)
COLORS = dict(D2='#586675', D3='#1976a3', W3='#83b8ce', D4='#bd602d', W4='#e3b18c')
FOOTNOTE = ('Existing reused test, n=1024; truth: inverse-Fourier labels. '
            'Errors are percentages; lower is better.')


def rows_by_identity(summary):
    if summary.get('stage') != 'test':
        raise ValueError('Expected the frozen test summary from analyze_depth.py.')
    rows = {}
    for row in summary['runs']:
        identity = (row['label'], row['training_seed'])
        if identity in rows:
            raise ValueError('Duplicate label/training_seed in summary.')
        if identity[0] not in ORDER:
            raise ValueError('Unknown architecture: ' + identity[0])
        rows[identity] = row
    return rows


def error_percent(row, species=None):
    if row is None:
        return np.nan
    value = (row['macro_relative_l2']['mean'] if species is None else
             row['metrics'][species]['relative_l2']['mean'])
    value = float(value) * 100
    if not np.isfinite(value) or value < 0:
        raise ValueError('Expected a finite nonnegative relative-L2 mean.')
    return value


def finish(fig, out, name, caption=''):
    fig.text(.02, .025, FOOTNOTE + ('\n' + caption if caption else ''),
             ha='left', va='bottom', fontsize=8, color='#424242')
    fig.tight_layout(rect=(0, .13, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(out / (name + '.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


def mark_missing(ax, x, text='Missing'):
    ax.text(x, .04, text, transform=ax.get_xaxis_transform(), rotation=90,
            ha='center', va='bottom', fontsize=9, color='#777777')


def primary_plot(rows, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ORDER))
    values = [error_percent(rows.get((label, SEEDS[0]))) for label in ORDER]
    ax.bar(x, values, color=[COLORS[label] for label in ORDER], width=.65)
    for position, value in zip(x, values):
        if np.isnan(value):
            mark_missing(ax, position)
        else:
            ax.annotate(f'{value:.3f}%', (position, value), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(x, [f'{label}\n{DEPTH[label]} hidden layers\n{PARAMETERS[label]/1e6:.4f}M params'
                      for label in ORDER])
    ax.set_ylabel('Mean macro VDF relative L2 (%)')
    ax.set_title('Primary training seed 20261030: depth and capacity controls')
    ax.set_ylim(bottom=0)
    ax.margins(y=.2)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    finish(fig, out, 'primary_macro', 'Macro: equal mean of electron, H and C errors per sample; no error bars.')


def infer_selected(rows, requested):
    inferred = {label[0] for label in rows if label[1] in SEEDS[1:] and label[0] in ('D3', 'D4')}
    inferred.update('D' + label[0][1:] for label in rows
                    if label[1] in SEEDS[1:] and label[0] in ('W3', 'W4'))
    if len(inferred) > 1:
        raise ValueError('Repeated runs contain both depth families; selection is ambiguous.')
    if requested and inferred and requested not in inferred:
        raise ValueError('Requested depth disagrees with repeated run labels.')
    return requested or next(iter(inferred), None)


def seeds_plot(rows, out, selected):
    labels = ['D2', selected, 'W' + selected[1:] if selected else None]
    tick_labels = [label or placeholder for label, placeholder in
                   zip(labels, ('D2', 'Selected depth', 'Matched width'))]
    fig, ax = plt.subplots(figsize=(8, 5))
    seed_colors = ('#333333', '#3159a2', '#ac553e')
    for index, (seed, color) in enumerate(zip(SEEDS, seed_colors)):
        values = [error_percent(rows.get((label, seed))) if label else np.nan for label in labels]
        # NaN breaks lines, so unavailable groups are never bridged or assigned zero.
        x = np.arange(3) + (index - 1) * .045
        ax.plot(x, values, 'o-', color=color, linewidth=1.2, markersize=6,
                label=f'Training seed {seed}')
    for position, label in enumerate(labels):
        missing = [str(seed) for seed in SEEDS if label is None or (label, seed) not in rows]
        if missing:
            ax.text(position, .02, 'Missing: ' + ', '.join(missing),
                    transform=ax.get_xaxis_transform(), rotation=90,
                    va='bottom', ha='center', fontsize=7, color='#888888')
    ax.set_xticks(range(3), tick_labels)
    ax.set_xlim(-.45, 2.45)
    ax.set_ylabel('Mean macro VDF relative L2 (%)')
    ax.set_title('Validation-selected depth: paired training-seed results')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=.2)
    finish(fig, out, 'paired_training_seeds',
           'Lines pair models trained with the same seed. Dots are individual runs; no confidence interval.'
           + (' Selected depth cannot be inferred from available repeats.' if selected is None else ''))


def species_plot(rows, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    species = ('e', 'i1', 'i2')
    centers = np.arange(3)
    width = .145
    for index, label in enumerate(ORDER):
        row = rows.get((label, SEEDS[0]))
        values = [error_percent(row, s) for s in species]
        ax.bar(centers + (index - 2) * width, values, width=width,
               label=label, color=COLORS[label])
    missing = [label for label in ORDER if (label, SEEDS[0]) not in rows]
    ax.set_xticks(centers, ('Electron', 'Hydrogen', 'Carbon'))
    ax.set_ylabel('Mean VDF relative L2 (%)')
    ax.set_title('Primary training seed 20261030: species errors')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, fontsize=9)
    finish(fig, out, 'primary_species',
           'Species means; no error bars.' + (' Missing groups: ' + ', '.join(missing) + '.' if missing else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--selected-depth', choices=('D3', 'D4'),
                        help='Optional selection label when all repeated runs are missing.')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = rows_by_identity(summary)
    selected = infer_selected(rows, args.selected_depth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ('primary_macro', 'paired_training_seeds', 'primary_species')
    if any((args.output_dir / (name + '.' + suffix)).exists()
           for name in names for suffix in ('png', 'pdf')):
        raise FileExistsError('Refusing to replace existing plots.')
    primary_plot(rows, args.output_dir)
    seeds_plot(rows, args.output_dir, selected)
    species_plot(rows, args.output_dir)
    print(json.dumps({'output_dir': str(args.output_dir), 'figures': list(names),
                      'selected_depth': selected, 'source_summary': str(args.summary)}))


if __name__ == '__main__':
    main()
