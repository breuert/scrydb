import sqlite3

import sqlite_vec


def test_sqlite_vec_is_loadable():
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    (version,) = conn.execute("SELECT vec_version()").fetchone()
    assert version

    conn.execute("CREATE VIRTUAL TABLE t USING vec0(embedding float[4])")
    conn.close()


def test_index_open_loads_extension_by_default(tmp_path):
    from scrydb import Index

    db_path = tmp_path / "idx.db"
    with Index.open(db_path) as index:
        (version,) = index.conn.execute("SELECT vec_version()").fetchone()
        assert version


def test_index_open_can_skip_extension(tmp_path):
    from scrydb import Index

    db_path = tmp_path / "idx.db"
    with Index.open(db_path, vec_ext_path=None) as index:
        import sqlite3 as sqlite3_module

        try:
            index.conn.execute("SELECT vec_version()")
        except sqlite3_module.OperationalError:
            pass
        else:
            raise AssertionError("vec_version() should be unavailable when vec_ext_path=None")

        # Lexical search still works without the extension loaded.
        index.index_documents([{"id": "1", "text": "hello world"}])
        results = index.search("hello", mode="lexical")
        assert [r.id for r in results] == ["1"]
