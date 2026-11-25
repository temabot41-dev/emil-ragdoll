import os
TMP = os.environ.get('EMIL_TMP') or '/tmp/emil'  # scratch dir of the bench output
import csv, sys
# Per touchdown in $EMIL_TMP/pk/one/dbg.csv: time, leg, target and capture point along axis (x or y), landing peak.
ax = sys.argv[2] if len(sys.argv) > 2 else 'x'
rows = list(csv.DictReader(open((TMP + '/pk/one/dbg.csv'))))
fn = lambda r, s: float(r['fnL' if s == 0 else 'fnR'])
lift = [float(b['t']) for a, b in zip(rows, rows[1:]) if a['mode'] == '0' and b['mode'] == '1']
out = []
for i in range(1, len(rows)):
  a, b = rows[i - 1], rows[i]
  if a['mode'] == '1' and b['mode'] == '0':
    s, t0 = int(a['swing']), float(b['t'])
    w = [fn(r, s) for r in rows[max(0, i - 8):i + 12] if t0 - 0.06 <= float(r['t']) <= t0 + 0.1]
    out.append('%.2f %s foot=%.2f xi=%.2f pk=%.0f' % (t0, 'LR'[s], float(b['a' + ax + 'LR'[s]]), float(a['xi_' + ax]),
                                                    max(w) if w else 0))
print('%s first_lift=%.2f | %s' % (sys.argv[1], lift[0] if lift else -1, ' ; '.join(out[:5])))
