"""Plot saved QA profiles; no downloads, matrix construction or eigenmode solve."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REFERENCE = Path(__file__).with_name('references')/'qa_ae3d_comparison.npz'
MODELS = ('native_physical', 'native_interchanged', 'public_endpoint')
LABELS = ('Native FAX', 'FAX, both shear index swaps', 'Public AE3D formulas + endpoint convention')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/_static/qa_ae3d_comparison.png'))
    args = parser.parse_args()
    with np.load(REFERENCE, allow_pickle=False) as data:
        s, harmonics, reference = (data[k] for k in ('s', 'harmonics', 'reference'))
        metadata = json.loads(str(data['metadata_json']))
        strongest = np.argsort(np.trapezoid(reference**2, s, axis=0))[-5:][::-1]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, column in zip(axes.ravel(), strongest):
            ax.plot(s, reference[:, column], 'k-', lw=2, label='Archived AE3D')
            for model, label, style in zip(MODELS, LABELS, ('-', '--', ':')):
                ax.plot(s, data[model][:, column], style, label=label)
            m, n = harmonics[column]
            ax.set(title=f'm={m}, n={n}', xlabel='Toroidal flux s', ylabel='Aligned potential [V]')
            ax.grid(alpha=.2)
        cases = [metadata['models'][k] for k in MODELS]
        errors = [100*c['retained_profile_relative_l2'] for c in cases]
        ax = axes.ravel()[-1]
        bars = ax.bar(['Native FAX', 'Both swaps', 'Public AE3D'], errors)
        ax.bar_label(bars, labels=[f'{e:.2f}%' for e in errors], padding=3)
        ax.set(title='All 35 retained profiles', ylabel='Aligned relative L2 error [%]', ylim=(0, max(errors)*1.2))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
        fig.suptitle('QA comparison: archived AE3D vs reconstructed operators (unresolved)\n'
                     + '; '.join(f'{label}: {c["frequency_hz"]/1000:.3f} kHz'
                                 for label, c in zip(('Native', 'Swapped', 'Public AE3D'), cases)))
        fig.tight_layout(rect=(0, .09, 1, .91))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
