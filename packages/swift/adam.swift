// adam.swift
// Provides the path to the adam SQLite extension for use with sqlite3_load_extension.

import Foundation

public struct adam {
    /// Returns the absolute path to the adam dylib for use with sqlite3_load_extension.
    public static var path: String {
        #if os(macOS)
        return Bundle.main.bundlePath + "/Contents/Frameworks/adam.framework/adam"
        #else
        return Bundle.main.bundlePath + "/Frameworks/adam.framework/adam"
        #endif
    }
}
