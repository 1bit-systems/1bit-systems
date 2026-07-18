// simple_tokenizer.cpp — single definition point for the shared g_tokenizer
// instance declared in include/simple_tokenizer.h. Lives in the
// backend_manager static lib so every target that links it (unified_server,
// backend_demo, ...) gets the symbol, regardless of whether that target
// actually loads/uses a real tokenizer.
#include "simple_tokenizer.h"

SimpleTokenizer g_tokenizer;
