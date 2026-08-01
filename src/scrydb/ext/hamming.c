#include <stdint.h>
#include <string.h>
#include <sqlite3ext.h>

SQLITE_EXTENSION_INIT1

static void hamming_distance(sqlite3_context *context, int argc, sqlite3_value **argv);

// Extension entrypoint (called by SQLite when you .load the dylib).
// Named sqlite3_hamming_init so that SQLite finds it automatically from
// the extension filename "hamming.dylib" -- no need to pass an explicit
// entry point when loading (e.g. `.load ./hamming.dylib` in the sqlite3
// CLI, or conn.load_extension("./hamming.dylib") in Python).
int sqlite3_hamming_init(
  sqlite3 *db,
  char **pzErrMsg,
  const sqlite3_api_routines *pApi
){
  (void)pzErrMsg;
  SQLITE_EXTENSION_INIT2(pApi);

  return sqlite3_create_function(
    db,
    "hamming_distance",   // SQL function name
    2,                    // argc
    SQLITE_UTF8 | SQLITE_DETERMINISTIC,
    0,                    // user data
    hamming_distance,     // xFunc
    0, 0
  );
}

static void hamming_distance(sqlite3_context *context, int argc, sqlite3_value **argv) {
  if (argc != 2) {
    sqlite3_result_error(context, "hamming_distance() requires 2 arguments", -1);
    return;
  }

  if (sqlite3_value_type(argv[0]) == SQLITE_NULL || sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    sqlite3_result_null(context);
    return;
  }

  const unsigned char *blob1 = sqlite3_value_blob(argv[0]);
  const unsigned char *blob2 = sqlite3_value_blob(argv[1]);
  int len1 = sqlite3_value_bytes(argv[0]);
  int len2 = sqlite3_value_bytes(argv[1]);

  if (!blob1 || !blob2) {
    sqlite3_result_null(context);
    return;
  }

  if (len1 != len2) {
    sqlite3_result_error(context, "vectors must be same length", -1);
    return;
  }

  // SQLite does not guarantee 8-byte alignment for blob pointers, so we
  // memcpy into aligned locals rather than casting to (uint64_t *) and
  // dereferencing directly (which is undefined behavior on a misaligned
  // pointer and can crash on some platforms/optimization levels).
  int chunks = len1 / 8;
  uint64_t distance = 0;

  for (int i = 0; i < chunks; i++) {
    uint64_t a, b;
    memcpy(&a, blob1 + i * 8, sizeof(a));
    memcpy(&b, blob2 + i * 8, sizeof(b));
    distance += __builtin_popcountll(a ^ b);
  }

  for (int i = chunks * 8; i < len1; i++) {
    distance += __builtin_popcount((unsigned)(blob1[i] ^ blob2[i]));
  }

  sqlite3_result_int64(context, (sqlite3_int64)distance);
}
