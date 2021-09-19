#include <iostream>
#include <assert.h>
#include "word_counter_2gram.hpp"
#include "sentence_svm_2gram.hpp"
#include <dlib/svm.h>
using namespace std;

int main() {
    WordCounter2Gram<int> counter;
    SentenceSVM<int> svm;
    cout << svm;
    svm.train_with_features(std::vector<SentenceSVM<int>::sample_type>(), std::vector<double>());

    for(int i=0;i<0xf000;i++) {
        counter.eat(i);
    }

    char hello[] = "abcdefghijklmn";
    auto f = counter.feature(hello, hello + sizeof(hello));

    cout << get<0>(f) << " " << get<1>(f) << " " << get<2>(f) << endl;

    WordCounter2Gram<int> bx;
    char* buf = new char[counter.savesize()];
    size_t n = counter.savesize(), m = 0;
    assert(counter.save(buf, n, m));
    assert(n == m);
    cout << n << " " << m << endl;
    assert(bx.load(buf, n, m));
    assert(n == m);

    auto g = bx.feature(hello, hello + sizeof(hello));
    assert(get<0>(g) == get<0>(f));
    assert(get<1>(g) == get<1>(f));
    assert(get<2>(g) == get<2>(f));

    delete[] buf;
}

