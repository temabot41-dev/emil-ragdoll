# Make a custom Android debug template with minSdkVersion patched in the binary manifest (non-Gradle export
# cannot override Min SDK). In-place int patch of the uses-sdk/minSdkVersion typed value; other entries copied as is.
import sys, struct, zipfile
src, dst, new_min = sys.argv[1], sys.argv[2], int(sys.argv[3])
zin = zipfile.ZipFile(src)
man = bytearray(zin.read('AndroidManifest.xml'))
pos = 8
_, sp_hdr, sp_size = struct.unpack_from('<HHI', man, pos)
n_str, _, flags, str_start, _ = struct.unpack_from('<IIIII', man, pos + 8)
offs = struct.unpack_from('<%dI' % n_str, man, pos + sp_hdr)
def s_at(o):
    p = pos + str_start + o
    if flags & 0x100:
        p += 2 if man[p] & 0x80 == 0 else 3
        return bytes(man[p:man.index(0, p)]).decode('utf8', 'replace')
    ln = struct.unpack_from('<H', man, p)[0]
    return bytes(man[p + 2:p + 2 + 2 * ln]).decode('utf16')
strs = [s_at(o) for o in offs]
p = pos + sp_size
done = 0
while p < len(man):
    t, h, sz = struct.unpack_from('<HHI', man, p)
    if t == 0x0102 and strs[struct.unpack_from('<I', man, p + 20)[0]] == 'uses-sdk':
        a_start, a_size, a_count = struct.unpack_from('<HHH', man, p + 24)
        for i in range(a_count):
            ap = p + 16 + a_start + i * a_size
            if strs[struct.unpack_from('<I', man, ap + 4)[0]] == 'minSdkVersion':
                struct.pack_into('<I', man, ap + 16, new_min)
                done += 1
    p += sz
assert done == 1, done
zout = zipfile.ZipFile(dst, 'w')
for info in zin.infolist():
    data = bytes(man) if info.filename == 'AndroidManifest.xml' else zin.read(info.filename)
    zout.writestr(info, data, compress_type=info.compress_type)
zout.close()
print('patched minSdkVersion ->', new_min)
