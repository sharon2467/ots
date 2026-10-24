#!/usr/bin/env python3
"""Render scientific captions from saved metric summaries; no model execution.

Example from repository root:
python reproduction/publication/render_report_figures.py --output-dir /tmp/ots-report-figures
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'restoration_20261023/work/ots')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    depth = args.work_root / 'outputs/depth_study_20261021'
    renderer = module(depth / 'plot_depth.py', 'depth_report_plot')
    summary = json.loads((depth / 'analysis/final/summary.json').read_text())
    rows = renderer.rows_by_identity(summary)
    if len(rows) != 8:
        raise ValueError('Expected the eight frozen checkpoints in this comparison.')
    renderer.FOOTNOTE = (
        'Eight frozen checkpoints; three intermediate models excluded. '
        'Four included runs have incomplete training tails.\n'
        'Existing test n=1024; inverse-Fourier label VDF errors; lower is better.')
    finish = renderer.finish

    def readable_finish(fig, out, name, caption=''):
        for ax in fig.axes:
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + 0.12 * (high - low))
            for text in ax.texts:
                text.set_text(text.get_text().replace('Missing', 'Not included'))
        return finish(fig, out, name, caption.replace('Missing groups', 'Groups not included'))

    renderer.finish = readable_finish
    out = args.output_dir / 'depth'
    out.mkdir()
    renderer.primary_plot(rows, out)
    renderer.seeds_plot(rows, out, 'D4')
    renderer.species_plot(rows, out)

    unified = args.work_root / 'outputs/fig_unified_test_20261021'
    renderer = module(unified / 'plot_models.py', 'unified_report_plot')
    original_payload = json.loads((unified / 'results.json').read_text())
    payload = copy.deepcopy(original_payload)
    for model in payload['models']:
        model['display_name'] = model['display_name'].replace(
            '[checkpoint restored]', '[frozen]').replace('[partial]', '[intermediate]')
    # Display names are the only changed input fields, including nested metrics.
    for old, new in zip(original_payload['models'], payload['models']):
        assert {k:v for k,v in old.items() if k != 'display_name'} == {
            k:v for k,v in new.items() if k != 'display_name'}
    draw = renderer.draw_chart

    def scientific_caption(*a, **kw):
        kw['title'] = kw['title'].replace('已完成 Fig 模型', '已保存 Fig 模型')
        kw['scale_note'] = kw.get('scale_note', '') + (
            '\n状态：intermediate = 中间权重，仅作探索比较；'
            'frozen = 已选定权重，但训练尾段未完成。')
        return draw(*a, **kw)

    renderer.draw_chart = scientific_caption
    renderer.plot(payload, args.output_dir / 'unified', 240, 10)
    for path in args.output_dir.rglob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    print(json.dumps({'output_dir':str(args.output_dir), 'numerical_inputs_unchanged':True}))


if __name__ == '__main__':
    main()
