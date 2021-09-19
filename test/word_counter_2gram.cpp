#include <gtest/gtest.h>
#include <vector>
#include "word_counter_2gram.hpp"

TEST(WordCounter2Gram, SaveLoad) {
    WordCounter2Gram<int> counter, c2;

    for(int i=0;i<100;i++) {
        std::vector<int> sv;

        for(int j=0;j<100;j++)
            counter.eat((i << 5) + j);
        counter.sentenceEnd();
    }
    std::vector<int> sv = { 7, 10, 20, 50, 100};
    auto f1 = counter.feature(sv.begin(), sv.end());

    size_t n, m;
    EXPECT_TRUE(counter.save(nullptr, 0, n));

    char* buf = new char[n];
    EXPECT_TRUE(counter.save(buf, n, m));
    EXPECT_EQ(m, n);

    EXPECT_TRUE(c2.load(buf, n, m));
    EXPECT_EQ(m, n);

    auto f2 = c2.feature(sv.begin(), sv.end());
    EXPECT_EQ(f1, f2);

    delete[] buf;
}

