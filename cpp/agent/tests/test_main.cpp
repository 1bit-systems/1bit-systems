// Single doctest entrypoint for the onebit_agent_test binary. Every other
// tests/*.cpp file in this folder includes <doctest/doctest.h> without the
// IMPLEMENT define so doctest registers their TEST_CASEs into this main.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
