// Copyright (c) 2026 SQLite Cloud, Inc.
// Licensed under the Elastic License 2.0 (see LICENSE).

import 'dart:ffi';

import 'package:sqlite3/sqlite3.dart';

// @Native resolves from the code asset declared in hook/build.dart.
// The asset ID is 'package:sqlite_adam/src/native/sqlite_adam_extension.dart'.
@Native<Int Function(Pointer<Void>, Pointer<Void>, Pointer<Void>)>(
  assetId: 'package:sqlite_adam/src/native/sqlite_adam_extension.dart',
)
external int sqlite3_adam_init(
  Pointer<Void> db,
  Pointer<Void> pzErrMsg,
  Pointer<Void> pApi,
);

extension SqliteAdamExtension on Sqlite3 {
  /// Loads the adam SQLite extension.
  ///
  /// Call once at app startup. All subsequently opened databases
  /// will have the `adam_*` SQL functions available.
  ///
  /// Works with both `sqlite3` package and `drift` ORM.
  void loadSqliteAdamExtension() {
    ensureExtensionLoaded(
      SqliteExtension(
        Native.addressOf<
            NativeFunction<
                Int Function(Pointer<Void>, Pointer<Void>, Pointer<Void>)>>(
          sqlite3_adam_init,
        ).cast(),
      ),
    );
  }
}
