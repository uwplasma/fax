"""Plot the saved QA endpoint-convention audit; no JAX, downloads or solves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REFERENCE = Path(__file__).with_name('references')/'qa_vorticity_radial_audit_20261021.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path, default=Path('docs/_static/qa_ae3d_radial_audit'))
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as data:
        s, labels, reference = (data[key] for key in ('s', 'harmonics', 'reference'))
        periodic, endpoint = data['public_periodic'], data['public_endpoint']
        metadata = json.loads(str(data['metadata_json']))
    strongest = np.argsort(np.trapezoid(reference**2, s, axis=0))[-5:][::-1]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=False)
    for ax, column in zip(axes.ravel()[:5], strongest):
        ax.plot(s, reference[:, column], color='black', lw=2, label='Archived AE3D profile')
        ax.plot(s, periodic[:, column], color='#d97706', ls='--', lw=1.5,
                label='Public AE3D formulas, periodic geometry grid')
        ax.plot(s, endpoint[:, column], color='#15803d', ls='-.', lw=1.5,
                label='Same formulas, public-source endpoint convention')
        m, n = labels[column]
        ax.set(title=f'm={m}, n={n}', xlabel='Normalized toroidal flux s', ylabel='Signed potential [V]')
        ax.grid(alpha=.2)
    ax = axes.ravel()[-1]
    cases = [metadata['models'][key] for key in ('public_periodic', 'public_endpoint')]
    errors = [100*case['retained_profile_relative_l2'] for case in cases]
    bars = ax.bar(['Periodic\ngeometry grid', 'Public-source\nendpoint convention'], errors,
                  color=['#d97706', '#15803d'])
    ax.bar_label(bars, labels=[f'{error:.2f}%' for error in errors], padding=4)
    ax.set(title='Error across all 35 archived profiles', ylabel='Aligned relative L2 error [%]',
           ylim=(0, max(errors)*1.25))
    ax.grid(axis='y', alpha=.2)
    for index, case in enumerate(cases):
        ax.text(index, errors[index]/2, f"{case['frequency_hz']/1000:.5f} kHz",
                ha='center', va='center', color='white', fontsize=9)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=1, frameon=False, fontsize=9)
    fig.suptitle('QA radial-profile audit: angular endpoint convention\n'
                 '162 archived harmonics; reconstructed public AE3D formulas, not imported matrices', fontsize=12)
    fig.tight_layout(rect=(0, .12, 1, .91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        destination = args.output.with_suffix('.'+extension)
        fig.savefig(destination, dpi=170)
        print(destination)
    plt.close(fig)


if __name__ == '__main__':
    main()
