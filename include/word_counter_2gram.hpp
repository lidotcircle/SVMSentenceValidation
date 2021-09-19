#ifndef _WORD_COUNTER_HPP_
#define _WORD_COUNTER_HPP_

#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <type_traits>
#include <iterator>
#include "simple_archive.hpp"


template<typename T>
using is_input_iterator_t = typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<T>::iterator_category,
        std::input_iterator_tag>::value, 
    void>::type;

template<typename T, typename VT>
constexpr bool is_iterator_value_type_same_with_v = 
    std::is_same<typename std::iterator_traits<T>::value_type, VT>::value;

template<typename TWord>
class WordCounter2Gram {
private:
    std::map<TWord, int> ngram1;
    std::map<TWord, std::map<TWord, int>> ngram2;
    std::vector<TWord> prev;

public:
    void eat(TWord word) {
        ngram1[word] += 1;

        if (!prev.empty()) {
            ngram2[prev[0]][word] += 1;
            prev.clear();
        }

        prev.push_back(word);
    }
    template<typename TIter, typename = is_input_iterator_t<TIter>>
    void eat(TIter begin, TIter end) {
        for(;begin != end;begin++) {
            this->eat(*begin);
        }
    }
    void sentenceEnd() { this->prev.clear(); }

    void clear() {
        this->ngram1.clear();
        this->ngram2.clear();
        this->prev.clear();
    }

    size_t savesize() const {
        size_t n1 = 0, n2 = 0;
        writeToBuf(this->ngram1, nullptr, 0, n1);
        writeToBuf(this->ngram2, nullptr, 0, n2);

        return n1 + n2;
    }
    bool save(char* buf, size_t bufsize, size_t& saved) const {
        if(buf == nullptr) {
            saved = this->savesize();
            return true;
        }

        size_t n1 = 0, n2 = 0;
        if(!writeToBuf(this->ngram1, buf, bufsize, n1)) {
            return false;
        }
        if(!writeToBuf(this->ngram2, buf + n1, bufsize - n1, n2)) {
            return false;
        }
        saved = n1 + n2;
        return true;
    }
    bool load(char* buf, size_t bufsize, size_t& read) {
        size_t n1 = 0, n2 = 0;
        if(!readFromBuf(this->ngram1, buf, bufsize, n1)) {
            return false;
        }

        if(!readFromBuf(this->ngram2, buf + n1, bufsize - n1, n2)) {
            return false;
        }

        read = n1 + n2;
        return true;
    }

    template<
        typename TWordIter,
        typename = is_input_iterator_t<TWordIter>>
    std::tuple<int,int,int> feature(TWordIter begin, TWordIter end) {
        this->prev.clear();
        int gram1 = 0, gram2 = 0, len = 0;

        for(;begin != end;begin++) {
            auto word = *begin;
            len++;
            gram1 += this->ngram1[word];

            if (!prev.empty()) {
                prev.clear();
                gram2 += this->ngram2[prev[0]][word];
            }

            this->prev.push_back(word);
        }

        return std::make_tuple(len, gram1, gram2);
    }
};

#endif // _WORD_COUNTER_HPP_

