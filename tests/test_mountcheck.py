# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlperf_common import mountcheck  # noqa: E402


class VerifyMountsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.data = self.tmp / "data"
        (self.data / "metadata").mkdir(parents=True)
        (self.data / "images").mkdir()
        # 12011 bytes, between one slice (4096) and the sampling budget (40960)
        (self.data / "metadata" / "classes.csv").write_bytes(os.urandom(12011))
        (self.data / "small.bin").write_bytes(os.urandom(1000))
        (self.data / "large.bin").write_bytes(os.urandom(100000))
        for i in range(3):
            (self.data / "images" / f"{i}.jpg").write_bytes(os.urandom(20000))
        self.csv = self.tmp / "expected.csv"
        self.mounts = [f"DATA:{self.data}"]
        self.filtered = [".jpg"]
        with contextlib.redirect_stdout(io.StringIO()):
            mountcheck.initialize_expected_mounts(self.csv, self.mounts, self.filtered)

    def tearDown(self):
        self._tmp.cleanup()

    def verify(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mountcheck.verify_actual_mounts(self.csv, self.mounts, self.filtered, verbosity=1)
        return [line for line in out.getvalue().splitlines() if "WARNING" in line]

    def truncate(self, path, size):
        with open(path, "r+b") as f:
            f.truncate(size)

    def test_unchanged_tree_has_no_warnings(self):
        self.assertEqual(self.verify(), [])

    def test_small_file_truncation_is_reported(self):
        self.truncate(self.data / "small.bin", 500)
        self.assertTrue(any("small.bin" in w for w in self.verify()))

    def test_large_file_truncation_is_reported(self):
        self.truncate(self.data / "large.bin", 60000)
        self.assertTrue(any("large.bin" in w for w in self.verify()))

    def test_mid_size_file_truncation_is_reported(self):
        path = self.data / "metadata" / "classes.csv"
        self.truncate(path, 6000)
        warnings = self.verify()
        self.assertTrue(
            any(str(path) in w and "num bytes mismatch" in w for w in warnings),
            warnings,
        )

    def test_mid_size_file_append_is_reported(self):
        path = self.data / "metadata" / "classes.csv"
        with open(path, "ab") as f:
            f.write(b"\0" * 500)
        self.assertTrue(any(str(path) in w for w in self.verify()))

    def test_filtered_out_file_truncation_is_reported_on_parent_dir(self):
        self.truncate(self.data / "images" / "1.jpg", 10000)
        warnings = self.verify()
        self.assertTrue(
            any(f"{self.data / 'images'} num bytes mismatch" in w for w in warnings),
            warnings,
        )


if __name__ == "__main__":
    unittest.main()
