import sqlite3

import pytest

from scrydb.core import _bundled_hamming_path


def test_bundled_extension_is_discoverable():
    path = _bundled_hamming_path()
    assert path is not None, (
        "No compiled hamming extension found for this platform -- did the "
        "package build step run? (pip install -e . should compile it.)"
    )
    assert path.is_file()


def test_hamming_distance_via_raw_sqlite():
    path = _bundled_hamming_path()
    if path is None:
        pytest.skip("hamming extension not built for this platform")

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(str(path))
    conn.enable_load_extension(False)

    # 0xFF vs 0x0F differ in the top nibble -> 4 bits differ
    (distance,) = conn.execute("SELECT hamming_distance(x'FF00', x'0F00')").fetchone()
    assert distance == 4

    (distance,) = conn.execute("SELECT hamming_distance(x'0000', x'0000')").fetchone()
    assert distance == 0

    conn.close()


def test_index_open_loads_extension_by_default(tmp_path):
    from scrydb import Index

    db_path = tmp_path / "idx.db"
    with Index.open(db_path) as index:
        (distance,) = index.conn.execute(
            "SELECT hamming_distance(x'FF00', x'0F00')"
        ).fetchone()
        assert distance == 4
