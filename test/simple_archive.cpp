#include <gtest/gtest.h>
#include <map>
#include "simple_archive.hpp"

using namespace std;

TEST(SimpleArchiveTest, Integer) {
    char buf[8];
    {
        size_t n, nx;
        char a = 0xff, ax;
        size_t s = sizeof(a);
        EXPECT_TRUE(writeToBuf(a, nullptr, 0, n));
        EXPECT_EQ(n, s);
        EXPECT_TRUE(writeToBuf(a, buf, s, n));
        EXPECT_EQ(n, s);

        EXPECT_TRUE(readFromBuf(ax, buf, s, n));
        EXPECT_EQ(ax, a);
        EXPECT_EQ(n, s);
    }
    {
        size_t n, nx;
        uint16_t a = 0xff11, ax;
        size_t s = sizeof(a);
        EXPECT_TRUE(writeToBuf(a, nullptr, 0, n));
        EXPECT_EQ(n, s);
        EXPECT_TRUE(writeToBuf(a, buf, s, n));
        EXPECT_EQ(n, s);

        EXPECT_TRUE(readFromBuf(ax, buf, s, n));
        EXPECT_EQ(ax, a);
        EXPECT_EQ(n, s);
    }
    {
        size_t n, nx;
        uint32_t a = 0xff11, ax;
        size_t s = sizeof(a);
        EXPECT_TRUE(writeToBuf(a, nullptr, 0, n));
        EXPECT_EQ(n, s);
        EXPECT_TRUE(writeToBuf(a, buf, s, n));
        EXPECT_EQ(n, s);

        EXPECT_TRUE(readFromBuf(ax, buf, s, n));
        EXPECT_EQ(ax, a);
        EXPECT_EQ(n, s);
    }
    {
        size_t n, nx;
        size_t a = 0xff11, ax;
        size_t s = sizeof(a);
        EXPECT_TRUE(writeToBuf(a, nullptr, 0, n));
        EXPECT_EQ(n, s);
        EXPECT_TRUE(writeToBuf(a, buf, s, n));
        EXPECT_EQ(n, s);

        EXPECT_TRUE(readFromBuf(ax, buf, s, n));
        EXPECT_EQ(ax, a);
        EXPECT_EQ(n, s);
    }
}

TEST(SimpleArchiveTest, Map) {
    map<int,int> f = { {2,3}, {3,4} };
    size_t n;
    EXPECT_TRUE(writeToBuf(f, nullptr, 0, n)) << "GET buffer length failed";

    char* buf = new char[n];
    size_t m;
    EXPECT_TRUE(writeToBuf(f, buf, n, m)) << "Write to result failed";
    EXPECT_EQ(n, m) << "bad length";

    map<int,int> u;
    size_t k;
    EXPECT_TRUE(readFromBuf(u, buf, m, k)) << "readFromBuf failed";
    EXPECT_EQ(k, m) << "buffer should be fully consumed";

    EXPECT_EQ(u.size(), 2);
    EXPECT_EQ(u[2], 3);
    EXPECT_EQ(u[3], 4);

    delete[] buf;
}

TEST(SimpleArchiveTest, MapOfMap) {
    map<int,map<int,int>> f = { {2, {{4, 5}, {6,3}}}, {3, {{7, 8}, {9,10}}} };
    size_t m, n;

    EXPECT_TRUE(writeToBuf(f, nullptr, 0, m));
    char* buf = new char[m];
    EXPECT_TRUE(writeToBuf(f, buf, m, n));
    EXPECT_EQ(m, n);

    map<int,map<int,int>> u;
    EXPECT_TRUE(readFromBuf(u, buf, m, n));
    EXPECT_EQ(m, n);

    EXPECT_EQ(u.size(), 2);
    EXPECT_EQ(u[2][4], 5);
    EXPECT_EQ(u[3][7], 8);

    delete[] buf;
}

