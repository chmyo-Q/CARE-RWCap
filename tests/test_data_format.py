"""Check the GGFT file boundary without invoking the numerical solver."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generate_data import RECORD_VALUES, check_format


class DataFormatTest(unittest.TestCase):
    def test_complete_record_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poisson.bin'
            with path.open('wb') as stream:
                stream.write(struct.pack('<dd', 23.0, 1.0))
                stream.truncate(16 + RECORD_VALUES * 8)
            self.assertEqual(check_format(path, 1)['samples'], 1)
            with self.assertRaisesRegex(ValueError, 'complete records'):
                check_format(path, 2)

    def test_wrong_header_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gradient.bin'
            path.write_bytes(struct.pack('<dd', 21.0, 1.0))
            with self.assertRaisesRegex(ValueError, 'header'):
                check_format(path, 1)


if __name__ == '__main__':
    unittest.main()
