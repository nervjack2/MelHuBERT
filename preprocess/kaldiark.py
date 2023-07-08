#!/usr/bin/env python3
#
# Copyright (C) 2022  Hao Tang
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import sys
import struct
import array
import numpy


def parse_ark_entry(f):
    key = parse_key(f)

    if key == None:
        return None, None

    mat = parse_feat_matrix(f)
    return key, mat


def write_ark_entry(k, feat, f):
    write_key(k, f)
    write_feat_matrix(feat, f)


def parse_key(f):
    key = bytearray()

    c = f.read(1)
    if not c:
        return None

    while c and c != b' ':
        key.extend(c)
        c = f.read(1)

    return key.decode()


def write_key(k, f):
    b = bytes(k.encode('ascii'))
    f.write(b)
    f.write(b' ')


def parse_binary_sig(f):
    sig = f.read(2)

    # '\x00\x42' is NULL followed by a B
    if sig != b'\x00\x42':
        print('unsupported binary format')
        print('expect 0042 but {}'.format(sig.hex()))
        exit(1)


def write_binary_sig(f):
    f.write(b'\x00\x42')


def parse_feat_matrix(f):
    parse_binary_sig(f)

    sig = f.read(3)

    if sig == b'CM ':
        return parse_compressed_matrix(f)
    elif sig == b'FM ':
        return parse_float_matrix(f)
    elif sig == b'DM ':
        return parse_double_matrix(f)
    else:
        print('unsupported matrix format')
        exit(1)


def write_feat_matrix(feat, f):
    write_binary_sig(f)
    f.write(b'FM ')
    write_float_matrix(feat, f)


def short_to_float(min_value, value_range, value):
    return min_value + value_range * value / 65535.0


def char_to_float(p0, p25, p75, p100, value):
    if value <= 64:
        return p0 + (p25 - p0) * value * (1 / 64)
    elif value <= 192:
        return p25 + (p75 - p25) * (value - 64) * (1 / 128)
    else:
        return p75 + (p100 - p75) * (value - 192) * (1 / 63)


def parse_compressed_matrix(f):
    header = f.read(16)
    min_value, value_range, rows, cols = struct.unpack('<ffii', header)
    result = []

    per_col_header = []
    for i in range(cols):
        s = f.read(8)
        p0, p25, p75, p100 = struct.unpack('<HHHH', s)
        p0 = short_to_float(min_value, value_range, p0)
        p25 = short_to_float(min_value, value_range, p25)
        p75 = short_to_float(min_value, value_range, p75)
        p100 = short_to_float(min_value, value_range, p100)
        per_col_header.append((p0, p25, p75, p100))

    for i in range(cols):
        data = array.array('B')
        data.fromfile(f, rows)
        p0, p25, p75, p100 = per_col_header[i]
        a = [char_to_float(p0, p25, p75, p100, e) for e in data]
        result.extend(a)

    return numpy.array(result).reshape(cols, rows).T


def parse_float_matrix(f):
    header = f.read(10)
    rw, rows, cw, cols = struct.unpack('<bibi', header)
    if rw != 4 or cw != 4:
        print('broken header')
        exit(1)
    data = array.array('f')
    data.fromfile(f, rows * cols)
    return numpy.array(data).reshape(rows, cols)


def parse_double_matrix(f):
    header = f.read(10)
    rw, rows, cw, cols = struct.unpack('<bibi', header)
    if rw != 4 or cw != 4:
        print('broken header')
        exit(1)
    data = array.array('d')
    data.fromfile(f, rows * cols)
    return numpy.array(data).reshape(rows, cols)


def write_float_matrix(mat, f):
    rows = len(mat)
    cols = len(mat[0])
    f.write(struct.pack('<bibi', 4, rows, 4, cols))
    data = array.array('f')
    for v in mat:
        data.extend(v)
    data.tofile(f)


if __name__ == '__main__':
    f = open(sys.argv[1], 'rb')
    while True:
        key, mat = parse_ark_entry(f)
        if key:
            print('{}, rows: {}, cols: {}'.format(key, mat.shape[0], mat.shape[1]))
            print('{}'.format(mat))
        else:
            break
    f.close()

