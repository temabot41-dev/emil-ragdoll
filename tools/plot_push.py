#!/usr/bin/env python3
"""Draw push-run frames every 0.1 s: side view (x-z) and top view (x-y).

Usage: python3 -I tools/plot_push.py RUN.csv [DBG.csv] OUT.png [t0 t1]
RUN.csv: bench push CSV (docs/BENCH_FORMAT.md). DBG.csv (optional, from `emil_bench push1`): CoM, capture point,
step target, balancer mode. Uses PIL only.
"""
import csv
import sys

from PIL import Image, ImageDraw


def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in r]


def main():
    args = sys.argv[1:]
    run = load(args[0])
    dbg = load(args[1]) if len(args) >= 3 and args[1].endswith('.csv') else None
    out = args[2] if dbg else args[1]
    rest = args[3:] if dbg else args[2:]
    t0, t1 = (float(rest[0]), float(rest[1])) if len(rest) >= 2 else (0.0, 2.0)
    frames = [r for r in run if t0 - 1e-6 <= r['t'] <= t1 + 1e-6 and abs(r['t'] * 10 - round(r['t'] * 10)) < 0.09]
    dmap = {round(d['t'], 2): d for d in dbg} if dbg else {}
    n = len(frames)
    cw, ch = 220, 300  # cell: side view 180 px tall, top view below
    img = Image.new('RGB', (cw * min(n, 10), ch * ((n + 9) // 10)), 'white')
    dr = ImageDraw.Draw(img)
    # Common x range from pelvis track.
    xs = [r['pelvis_x'] for r in frames] + [r['l_toe_x'] for r in frames] + [r['r_toe_x'] for r in frames]
    xmid = 0.5 * (min(xs) + max(xs))
    scale = 100.0  # px per m
    for i, r in enumerate(frames):
        ox, oy = (i % 10) * cw, (i // 10) * ch
        def side(x, z):
            return ox + cw / 2 + (x - xmid) * scale, oy + 190 - z * scale
        def top(x, y):
            return ox + cw / 2 + (x - xmid) * scale, oy + 245 - y * scale
        dr.rectangle([ox, oy, ox + cw - 1, oy + ch - 1], outline='gray')
        dr.line([side(xmid - 1.1, 0), side(xmid + 1.1, 0)], fill='gray')
        dr.text((ox + 4, oy + 2), 't=%.1f' % r['t'], fill='black')
        dr.line([side(r['pelvis_x'], r['pelvis_z']), side(r['head_x'], r['head_z'])], fill='black', width=3)
        if 'chest_x' in r:  # reaction CSV: arms as chest -> hand lines (left blue, right red)
            for s_, col_ in (('l', 'blue'), ('r', 'red')):
                dr.line([side(r['chest_x'], r['chest_z']), side(r[s_ + '_hand_x'], r[s_ + '_hand_z'])], fill=col_, width=2)
        for s, col in (('l', 'blue'), ('r', 'red')):
            dr.line([side(r['pelvis_x'], r['pelvis_z']), side(r[s + '_ankle_x'], r[s + '_ankle_z'])], fill=col, width=2)
            dr.line([side(r[s + '_heel_x'], r[s + '_heel_z']), side(r[s + '_toe_x'], r[s + '_toe_z'])], fill=col, width=3)
            w = 3 if r[s + '_contact'] > 0.5 else 1
            dr.line([top(r[s + '_heel_x'], r[s + '_heel_y']), top(r[s + '_toe_x'], r[s + '_toe_y'])], fill=col, width=w)
        px, py = top(r['pelvis_x'], r['pelvis_y'])
        dr.ellipse([px - 3, py - 3, px + 3, py + 3], outline='black')
        d = dmap.get(round(r['t'], 2))
        if d:
            cx, cy = top(d['com_x'], d['com_y'])
            dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill='green')
            xx, xy = top(d['xi_x'], d['xi_y'])
            dr.line([cx, cy, xx, xy], fill='green')
            dr.rectangle([xx - 3, xy - 3, xx + 3, xy + 3], outline='green')
            if d['mode'] == 1:
                gx, gy = top(d['tgt_x'], d['tgt_y'])
                dr.line([gx - 4, gy - 4, gx + 4, gy + 4], fill='magenta', width=2)
                dr.line([gx - 4, gy + 4, gx + 4, gy - 4], fill='magenta', width=2)
            dr.text((ox + 4, oy + 12), 'm%d sw%d ph%d' % (d['mode'], d['swing'], d['phase']), fill='black')
    img.save(out)


if __name__ == '__main__':
    main()
