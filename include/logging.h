#pragma once

#include <glog/logging.h>

// Simple logging macros wrapping glog
// Usage:
//   LOG_INFO << "message";
//   LOG_ERROR << "error message";
//   LOG_FATAL << "fatal error";  // will abort

#define LOG_INFO LOG(INFO)
#define LOG_WARNING LOG(WARNING)
#define LOG_ERROR LOG(ERROR)
#define LOG_FATAL LOG(FATAL)
