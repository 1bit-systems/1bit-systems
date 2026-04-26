# 1bit-retrieval — keyword BM25 index over docs/wiki markdown.
# Port of crates/1bit-retrieval/. Pure-C++23, no external link deps beyond
# the doctest test framework.

add_library(onebit_retrieval_lib STATIC
    src/stopwords.cpp
    src/chunker.cpp
    src/wiki_index.cpp
)
target_include_directories(onebit_retrieval_lib PUBLIC include)
target_link_libraries(onebit_retrieval_lib PUBLIC onebit::core)
add_library(onebit::retrieval ALIAS onebit_retrieval_lib)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_retrieval_test
        tests/chunker_test.cpp
        tests/stopwords_test.cpp
        tests/wiki_index_test.cpp
    )
    target_link_libraries(onebit_retrieval_test PRIVATE
        onebit::retrieval
        doctest::doctest
    )
    add_test(NAME onebit_retrieval COMMAND onebit_retrieval_test)
endif()
