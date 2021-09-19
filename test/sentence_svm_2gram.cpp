#include <gtest/gtest.h>
#include "sentence_svm_2gram.hpp"
#include <vector>
#include <iostream>
using namespace std;

class SentenceSVM2GramTest: public ::testing::Test {
    protected:
        SentenceSVM<uint16_t> svm, svm2;
        vector<decltype(svm)::sample_type> samples;
        vector<double> labels;
        vector<std::vector<uint16_t>> uy;

        void SetUp() override {
            for(uint16_t i=0;i<100;i++) {
                vector<uint16_t> mn;

                for(uint16_t j=0;j<100;j++)
                    mn.push_back(i << 7 & j);

                svm.feed_sentence(mn.begin(), mn.end());
            }

            for(uint16_t i=0;i<8;i++) {
                vector<uint16_t> mn;

                for(uint16_t j=0;j<20;j++)
                    mn.push_back(i << 5 & j);

                uy.push_back(mn);
            }

            for(uint16_t i=0;i<30;i++) {
                vector<uint16_t> mn;

                for(uint16_t j=0;j<30;j++)
                    mn.push_back((j << (i << 7) & 0x0F) + i + j);

                samples.push_back(svm.feature2(mn.begin(), mn.end()));
                labels.push_back((i % 10 < 5) ? -1 : 1);
            }
            svm.train_with_features(samples, labels);
        }
};


TEST_F(SentenceSVM2GramTest, TrainPredict) {
    for(auto& un: uy) {
        auto v = svm.predict(un.begin(), un.end());
        EXPECT_TRUE(v >= -1 || v <= 1);
    }
}

TEST_F(SentenceSVM2GramTest, SaveLoadWithBuffer) {
    size_t bufsize, b2;
    ASSERT_TRUE(svm.save(nullptr, 0, bufsize));
    char* buf = new char[bufsize];
    ASSERT_TRUE(svm.save(buf, bufsize, b2));
    EXPECT_EQ(b2, bufsize);

    std::cout << bufsize << std::endl;
    ASSERT_TRUE(svm2.load(buf, bufsize, b2));
    EXPECT_EQ(b2, bufsize);
    delete[] buf;

    for(auto& un: uy) {
        auto v  = svm.predict(un.begin(), un.end());
        auto v2 = svm2.predict(un.begin(), un.end());
        EXPECT_EQ(v2, v);
    }
}

TEST_F(SentenceSVM2GramTest, SaveLoadWithIOS) {
    std::ostringstream oss;
    oss << svm;
    std::istringstream iss(oss.str());
    iss >> svm2;

    for(auto& un: uy) {
        auto v  = svm.predict(un.begin(), un.end());
        auto v2 = svm2.predict(un.begin(), un.end());
        EXPECT_EQ(v2, v);
    }
}

