#!/usr/bin/env python3
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import argparse
from pathlib import Path


def load_csv(expected_mounts_csv: Path) -> list[dict]:
    lines = expected_mounts_csv.read_text().strip().split("\n")
    rows = []
    keys = lines[0].split(",")
    for line in lines[1:]:
        values = line.split(",")
        row = {key: value for key, value in zip(keys, values)}
        row["num_files"] = int(row["num_files"])
        row["num_bytes"] = int(row["num_bytes"])
        rows.append(row)
    return rows


def save_csv(rows: list[dict], expected_mounts_csv: Path) -> None:
    lines = []
    columns = rows[0].keys()
    header = ",".join(columns)
    lines.append(header)
    for row in rows:
        values = ",".join(map(str, row.values()))
        lines.append(values)
    lines = "\n".join(lines)
    expected_mounts_csv.write_text(lines + "\n")
    print(f"{expected_mounts_csv} saved!")


def split(mounts_to_verify: list[str]) -> dict[str, Path]:
    mappings = {}
    for key_path_mapping in mounts_to_verify:
        key, path = key_path_mapping.split(":")
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"{repr(path)} for key={repr(key)} does not exists!")
        if key in mappings:
            raise RuntimeError(f"key={repr(key)} already used!")
        mappings[key] = path
    return mappings


def is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def scan(path: Path, key: str, root: Path) -> list[dict]:
    if is_hidden(path):
        return []
    if path.is_file():
        row = {}
        row["key"] = key
        row["type"] = "file"
        row["relative_path"] = str(path.relative_to(root))
        row["full_path"] = str(path)
        row["num_files"] = 1
        row["num_bytes"] = path.stat().st_size
        return [row]
    elif path.is_dir():
        rows = []
        for sub_path in path.glob("*"):
            rows += scan(sub_path, key, root)
        row = {}
        row["key"] = key
        row["type"] = "dir"
        row["relative_path"] = str(path.relative_to(root))
        row["full_path"] = str(path)
        row["num_files"] = sum([row["num_files"] for row in rows if row["type"] == "file"])
        row["num_bytes"] = sum([row["num_bytes"] for row in rows if row["type"] == "file"])
        rows.append(row)
        return rows
    else:
        raise RuntimeError(f"{repr(path)} is not a file nor a dir!")


def inspect(mounts_to_verify: list[str]) -> list[dict]:
    rows = []
    for key, path in split(mounts_to_verify).items():
        rows += scan(path, key, path)
    return rows


def print_check_info(message: str, verbosity: int, is_root_path: bool) -> None:
    if verbosity == 2 or (verbosity == 1 and is_root_path):
        print(message + "\n", end="")


def initialize_expected_mounts(expected_mounts_csv: Path, mounts_to_verify: list[str]) -> None:
    rows = inspect(mounts_to_verify)
    for row in rows:
        del row["full_path"]
    save_csv(rows, expected_mounts_csv)


def verify_actual_mounts(expected_mounts_csv: Path, mounts_to_verify: list[str], verbosity: int) -> None:
    expected_rows = load_csv(expected_mounts_csv)
    actual_rows = inspect(mounts_to_verify)
    mappings = split(mounts_to_verify)

    actual_rows_grouped = {}
    for actual in actual_rows:
        row_id = (actual["key"], actual["type"], actual["relative_path"])
        assert row_id not in actual_rows_grouped
        actual_rows_grouped[row_id] = actual

    for expected in expected_rows:
        row_id = (expected["key"], expected["type"], expected["relative_path"])
        is_root_path = expected["relative_path"] == "."

        if row_id not in actual_rows_grouped:
            mount_key = expected["key"]
            mount_path = mappings.get(mount_key, None)
            if mount_path is None:
                print_check_info(
                    f"mountcheck WARNING missing key:path mapping in --mounts_to_verify for key={repr(mount_key)}",
                    verbosity,
                    is_root_path,
                )
            else:
                missing_path = Path(mount_path) / Path(expected["relative_path"])
                print_check_info(
                    f"mountcheck WARNING {expected['type']} {missing_path} does not exist!",
                    verbosity,
                    is_root_path,
                )
            continue

        actual = actual_rows_grouped[row_id]

        if expected["num_bytes"] == actual["num_bytes"]:
            print_check_info(
                f"mountcheck OK {actual['full_path']} {actual['num_bytes']} bytes",
                verbosity,
                is_root_path,
            )
        else:
            print_check_info(
                f"mountcheck WARNING {actual['full_path']} num bytes mismatch! expected={expected['num_bytes']} actual={actual['num_bytes']}",
                verbosity,
                is_root_path,
            )

        if expected["type"] == "dir":
            if expected["num_files"] == actual["num_files"]:
                print_check_info(
                    f"mountcheck OK {actual['full_path']} {actual['num_files']} files",
                    verbosity,
                    is_root_path,
                )
            else:
                print_check_info(
                    f"mountcheck WARNING {actual['full_path']} num files mismatch! expected={expected['num_files']} actual={actual['num_files']}",
                    verbosity,
                    is_root_path,
                )


def main(args: argparse.Namespace) -> None:
    if args.initialize:
        initialize_expected_mounts(args.expected_mounts_csv, args.mounts_to_verify)
    else:
        verify_actual_mounts(args.expected_mounts_csv, args.mounts_to_verify, args.verbosity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected_mounts_csv",
        type=Path,
        default="expected-mounts.csv",
        help="""
        CSV file with expected mounts.
        When --initialize is passed this file is created.
        """,
    )
    parser.add_argument(
        "--mounts_to_verify",
        type=str,
        nargs="*",
        default=[],
        help="""
        Sequence of key:path mappings.
        For example '--mounts_to_verify DATADIR:/path/to/data MODELDIR:/path/to/model FILE:/path/to/file'.
        Keys must be unique.
        Keys specified for initialization must later be passed during verification.
        """,
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="""
        Whether to initialize expected_mounts_csv based on mounts_to_verify.
        """,
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="""
        Verbosity level.
        If 0, prints nothing.
        If 1, prints root paths check info only.
        If 2, prints everything.
        """,
    )
    main(args=parser.parse_args())
