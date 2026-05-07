# sqlite_adam

Adam is an embeddable AI agent library for SQLite, providing on-device chat, voice, vision, memory, and tool-calling. It works seamlessly on iOS, Android, Windows, Linux, and macOS, integrating directly with your embedded SQLite database.

## Installation

```
dart pub add sqlite_adam
```

Requires Dart 3.10+ / Flutter 3.38+.

## Usage

### With `sqlite3`

```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_adam/sqlite_adam.dart';

void main() {
  // Load once at startup.
  sqlite3.loadSqliteAdamExtension();

  final db = sqlite3.openInMemory();

  // Configure the agent (see API.md for the full list of options).
  db.execute("SELECT adam_config('provider', 'anthropic')");
  db.execute("SELECT adam_config('api_key', 'sk-...')");

  // Run the agent.
  final result = db.select("SELECT adam('Say hello in 3 words')");
  print(result.first[0]);

  db.dispose();
}
```

### With `drift`

```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_adam/sqlite_adam.dart';
import 'package:drift/native.dart';

Sqlite3 loadExtensions() {
  sqlite3.loadSqliteAdamExtension();
  return sqlite3;
}

// Use when creating the database:
NativeDatabase.createInBackground(
  File(path),
  sqlite3: loadExtensions,
);
```

## Supported platforms

| Platform | Architectures |
|----------|---------------|
| Android  | arm64, x64 |
| iOS      | arm64 (device + simulator) |
| macOS    | arm64, x64 |
| Linux    | arm64, x64 |
| Windows  | x64 |

## API

See the full [Adam API documentation](https://github.com/sqliteai/adam/blob/main/API.md).

## License

See [LICENSE](LICENSE).
