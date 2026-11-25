// Scratch directory of the bench output (CSV, traces): $EMIL_TMP, default /tmp/emil. Created with parents on first use.
#pragma once
#include <sys/stat.h>

#include <cstdlib>
#include <string>

inline const std::string& TmpDir() {
  static const std::string dir = [] {
    const char* e = std::getenv("EMIL_TMP");
    std::string d = (e && *e) ? e : "/tmp/emil";
    for (size_t i = 1; i <= d.size(); ++i)
      if (i == d.size() || d[i] == '/') mkdir(d.substr(0, i).c_str(), 0755);
    return d;
  }();
  return dir;
}

// TmpPath("pk/one") gives "<TmpDir>/pk/one".
inline std::string TmpPath(const char* rel) { return TmpDir() + "/" + rel; }
