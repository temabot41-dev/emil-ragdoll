# Print the push1 debug trace (every k-th frame between t0 and t1) with key columns.
import os
TMP = os.environ.get('EMIL_TMP') or '/tmp/emil'  # scratch dir of the bench output
import sys, csv
rows = list(csv.DictReader(open((TMP + '/pk/one/dbg.csv'))))
t0, t1, k = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
cols = sys.argv[4].split(',')
print('t     ' + ' '.join('%7s' % c[:7] for c in cols))
for i, r in enumerate(rows):
    t = float(r['t'])
    if t0 <= t <= t1 and i % k == 0:
        print('%5.2f ' % t + ' '.join('%7s' % r[c] for c in cols))
