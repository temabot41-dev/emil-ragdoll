import os
TMP = os.environ.get('EMIL_TMP') or '/tmp/emil'  # scratch dir of the bench output
import sys, re, collections
for r in sys.argv[1:]:
    st = collections.Counter(); sw = collections.Counter(); n = collections.Counter()
    for line in open((TMP + '/pk/%s/summary.txt') % r):
        m = re.match(r'([a-z]+)\d+@\S+.*fell=(\d).*swings=(\d+)', line)
        if not m: continue
        g = m.group(1); n[g] += 1; st[g] += m.group(2) == '0'; sw[g] += int(m.group(3))
    print(r, '  '.join('%s %d/%d sw%.1f' % (g, st[g], n[g], sw[g] / n[g]) for g in sorted(n)))
