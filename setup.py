"""
Build step for scrydb.

Project metadata lives in pyproject.toml (PEP 621). This file exists only
to hook a custom build step: compiling ext/hamming.c into a
platform-native, loadable SQLite extension --

    hamming.so     on Linux
    hamming.dylib  on macOS

-- so the compiled extension ships inside the installed package (as
scrydb/ext/hamming.{so,dylib}) and scrydb.Index.open() can find it
automatically at runtime (see scrydb/core.py:_bundled_hamming_path()).

hamming.c is NOT a Python C-extension module (it has no Python.h /
PyInit_* entrypoint) -- it's a plain SQLite loadable extension, so we
can't use setuptools' normal `Extension`/`build_ext` machinery, which
assumes it's building something importable by Python. Instead this
compiles/links it directly with the platform's C compiler.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

HERE = Path(__file__).parent.resolve()
EXT_DIR = HERE / "src" / "scrydb" / "ext"
EXT_SRC = EXT_DIR / "hamming.c"

# Output filename per platform, matching what scrydb.core._bundled_extension_filename()
# looks for at runtime.
OUTPUT_NAME = {
    "Linux": "hamming.so",
    "Darwin": "hamming.dylib",
}


def _brew_sqlite_prefix():
    try:
        return subprocess.check_output(
            ["brew", "--prefix", "sqlite"],
            text=True,
        ).strip()
    except Exception:
        return None


def _find_compiler() -> "str | None":
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _compile_command(compiler: str, system: str, output: Path) -> list:
    if system == "Darwin":
        # Build a standalone dynamic library. No need to link libsqlite3:
        # sqlite3ext.h resolves the SQLite API through a function-pointer
        # table (sqlite3_api) passed in at sqlite3_hamming_init() time, not
        # through symbols this .dylib needs resolved at link time.
        # return [compiler, "-O2", "-fPIC", "-dynamiclib", "-o", str(output), str(EXT_SRC)]
        prefix = _brew_sqlite_prefix()
        include_dir = f"{prefix}/include"
        lib_dir = f"{prefix}/lib"
        return [compiler, 
                "-O2", 
                "-fPIC", 
                "-dynamiclib", 
                "-Wall", 
                "-Wextra", 
                "-I", include_dir,
                "-L", lib_dir,
                "-o", str(output), str(EXT_SRC)]
    # Linux (and other ELF/gcc-compatible platforms, best-effort)
    return [compiler, "-O2", "-fPIC", "-shared", "-o", str(output), str(EXT_SRC)]


def compile_hamming_extension() -> None:
    system = platform.system()

    if system not in OUTPUT_NAME:
        print(
            f"scrydb: WARNING - skipping hamming_distance extension build: "
            f"unsupported platform {system!r} (only Linux and macOS are "
            "supported). The package will still install, but Hamming-"
            "distance (binary/hex) search will be unavailable; pass "
            "hamming_ext_path=None to scrydb.Index.open() to use lexical/"
            "cosine search only.",
            file=sys.stderr,
        )
        return

    compiler = _find_compiler()
    if compiler is None:
        raise RuntimeError(
            "scrydb: no C compiler found (checked $CC, then cc/gcc/clang on PATH). "
            "A C compiler is required to build scrydb's hamming_distance SQLite "
            "extension:\n"
            "  - macOS: install the Xcode Command Line Tools "
            "(`xcode-select --install`).\n"
            "  - Debian/Ubuntu: `sudo apt-get install build-essential libsqlite3-dev`\n"
            "  - Fedora/RHEL:   `sudo dnf install gcc sqlite-devel`\n"
            "  - Arch:          `sudo pacman -S base-devel sqlite`"
        )

    output = EXT_DIR / OUTPUT_NAME[system]
    cmd = _compile_command(compiler, system, output)
    print(f"scrydb: compiling hamming_distance extension for {system}: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "scrydb: failed to compile the hamming_distance SQLite extension.\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- stdout ---\n{exc.stdout}\n"
            f"--- stderr ---\n{exc.stderr}\n"
            "This is usually a missing 'sqlite3ext.h' header -- install your "
            "platform's SQLite development package (e.g. `libsqlite3-dev` on "
            "Debian/Ubuntu, `sqlite-devel` on Fedora/RHEL, or `brew install "
            "sqlite` on macOS) and retry."
        ) from exc

    print(f"scrydb: built {output.relative_to(HERE)}")


class build_py(_build_py):
    """Compile the native extension before the usual build_py copies package
    files (including whatever compile_hamming_extension() just produced) into
    the build directory."""

    def run(self):
        compile_hamming_extension()
        super().run()


cmdclass = {"build_py": build_py}

# This distribution contains a compiled, platform-specific shared library
# (hamming.so/hamming.dylib) even though there's no Python C-extension
# module (no Extension() objects, so setuptools has no other signal that
# this isn't a pure-Python package). Left alone, setuptools/wheel would tag
# the built wheel "py3-none-any", which is wrong: pip would then happily
# install a wheel built (and compiled) on Linux onto macOS, or vice versa,
# silently shipping the wrong/missing native library. Overriding
# bdist_wheel to mark the distribution as "not pure" forces a
# platform-specific tag (e.g. "py3-none-manylinux_2_35_x86_64" or
# "py3-none-macosx_11_0_arm64") instead, so pip only ever installs a wheel
# built on/for a matching platform.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

    cmdclass["bdist_wheel"] = bdist_wheel
except ImportError:
    # `wheel` isn't installed -- fine for `pip install .` (which builds via
    # build_meta and doesn't need the wheel package directly available
    # here), just means `python setup.py bdist_wheel` isn't usable until
    # `pip install wheel` is run.
    pass


setup(cmdclass=cmdclass)
