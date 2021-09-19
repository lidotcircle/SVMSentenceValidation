#include <map>
#include <vector>
#include <dlib/svm.h>
#include <assert.h>
#include "word_counter_2gram.hpp"


template<typename TWord>
class SentenceSVM {
    public:
        typedef dlib::matrix<double,5,1> sample_type;

    private:
        typedef dlib::radial_basis_kernel<sample_type> kernel_type;
        typedef dlib::decision_function<kernel_type> dec_funct_type;
        typedef dlib::normalized_function<dec_funct_type> funct_type;
        WordCounter2Gram<TWord> counter;
        funct_type learned_function;

        template<typename WTIter, typename = is_input_iterator_t<WTIter>>
        auto feature1(WTIter begin, WTIter end) {
            return counter.feature(begin, end);
        }

    public:
        template<typename STIter, typename VType = typename std::iterator_traits<STIter>::value_type,
            typename = is_input_iterator_t<STIter>,
            typename = typename std::enable_if<
                std::is_same<VType, std::pair<std::vector<TWord>,bool>>::value, void>::type>
        void train(STIter begin, STIter end) 
        {
            this->counter.clear();
            std::vector<sample_type> samples;
            std::vector<double> labels;

            for(;begin != end;begin++) {
                auto nv = *begin;
                samples.push_back(this->feature2(nv.first.begin(), nv.first.end()));
                labels.push_back(nv.second ? 1 : -1);
            }

            this->train_with_features(samples, labels);
        }

        void train_with_features(std::vector<sample_type> samples, std::vector<double> labels) 
        {
            assert(samples.size() == labels.size());

            dlib::vector_normalizer<sample_type> normalizer;
            normalizer.train(samples);
            for(size_t i=0;i<samples.size();i++)
                samples[i] = normalizer(samples[i]);
            dlib::randomize_samples(samples, labels);

            dlib::svm_c_trainer<kernel_type> trainer;
            double m = 0;
            double ga = 0;
            double ca = 0;
            for(double gamma = 0.00001;gamma<=1;gamma *= 5) {
                for(double C = 1; C<100000; C *= 5) {
                    trainer.set_kernel(kernel_type(gamma));
                    trainer.set_c(C);
                    auto result = dlib::cross_validate_trainer(trainer, samples, labels, 3);
                    double mx = 2 * dlib::prod(result) / dlib::sum(result);
                    if(mx > m) {
                        ga = gamma;
                        ca = C;
                    }
                    m = mx > m ? mx : m;
                }
            }

            trainer.set_kernel(kernel_type(ga));
            trainer.set_c(ca);

            this->learned_function.normalizer = normalizer;
            this->learned_function.function = trainer.train(samples, labels);
        }

        template<typename STIter, typename VType = typename std::iterator_traits<STIter>::value_type,
            typename = is_input_iterator_t<STIter>,
            typename = typename std::enable_if<std::is_same<VType, std::vector<TWord>>::value, void>::type>
        void feed(STIter begin, STIter end)
        {
            for(;begin != end;begin++) {
                auto sentence = *begin;

                for(auto& word: sentence) {
                    this->counter.eat(word);
                }

                this->counter.sentenceEnd();
            }
        }

        template<typename STIter, typename VType = typename std::iterator_traits<STIter>::value_type,
            typename = is_input_iterator_t<STIter>,
            typename = typename std::enable_if<std::is_same<VType, TWord>::value, void>::type>
        void feed_sentence(STIter begin, STIter end)
        {
            for(;begin != end;begin++) {
                this->counter.eat(*begin);
            }
            this->counter.sentenceEnd();
        }

        template<typename WTIter, typename = is_input_iterator_t<WTIter>>
        auto feature2(WTIter begin, WTIter end) {
            auto ft = this->feature1(begin, end);
            int a,b,c;
            std::tie(a,b,c) = ft;

            sample_type sample;
            sample(0) = a; sample(1) = b; sample(2) = c;
            sample(3) = (double)b/a; sample(4) = (double)c/a;
            return sample;
        }

        template<typename WTIter, typename = is_input_iterator_t<WTIter>>
        int  predict(WTIter begin, WTIter end) {
            auto ft = counter.feature(begin, end);
            int a, b, c = a = b = 0;
            std::tie(a, b, c) = ft;
            sample_type x;
            x(0) = a; x(1) = b; x(2) = c;
            x(3) = (double)b/a; x(4) = (double)c/a;

            return this->learned_function(x);
        }

        bool save(char* buf, size_t bufsize, size_t& writed) const {
            std::ostringstream membuf;
            dlib::serialize(membuf) << this->learned_function;

            if (buf == nullptr) {
                this->counter.save(nullptr, 0, writed);
                writed += membuf.str().size();
                writed += 2 * sizeof(size_t);
                return true;
            }

            size_t s1 = membuf.str().size();
            size_t n = sizeof(s1);
            if (bufsize < n) 
                return false;
            *(size_t*)buf = s1;

            if (bufsize < n + s1)
                return false;
            memcpy(buf + n, membuf.str().c_str(), s1);
            n += s1;

            size_t s2;
            n += sizeof(s2);
            if (bufsize < n)
                return false;

            if (!this->counter.save(nullptr, 0, s2))
                return false;

            if (bufsize < n + s2)
                return false;

            bool result = this->counter.save(buf + n, bufsize - n, writed);
            assert(writed == s2);

            writed += n;
            return result;
        };

        bool load(char* buf, size_t bufsize, size_t& readed) {
            size_t s1 = 0;
            size_t n = sizeof(s1);
            if (bufsize < n)
                return false;

            s1 = *(size_t*)buf;
            if (bufsize < n + s1)
                return false;

            if (s1 != 0) {
                std::string data(buf + n, buf + n + s1);
                dlib::deserialize(data) >> this->learned_function;
            }
            n += s1;

            size_t s2 = 0;
            if (bufsize < n + sizeof(s2))
                return false;
            n += sizeof(s2);

            s2 = *(size_t*)(buf + n);
            if (bufsize < n + s2)
                return false;

            bool result = this->counter.load(buf + n, bufsize - n, readed);
            assert(readed == s2);

            readed += n;
            return result;
        }
};

template<typename TWord>
std::ostream& operator<<(std::ostream& o, const SentenceSVM<TWord>& s) {
    size_t n;
    if (!s.save(nullptr, 0, n))
        throw std::runtime_error("serialization failed");

    char* buf = new char[n];

    if (!s.save(buf, n, n)) {
        delete[] buf;
        throw std::runtime_error("serialization failed");
    }

    o.write(buf, n);
    delete[] buf;
    return o;
}

template<typename TWord>
std::istream& operator>>(std::istream& input, SentenceSVM<TWord>& s) {
    std::istream_iterator<char> eos;
    std::vector<char> buf(std::istream_iterator<char>(input), eos);
    size_t n = buf.size();

    char* bufchar = new char[n];
    for(size_t i=0;i<buf.size();i++) bufchar[i] = buf[i];

    if (!s.load(bufchar, n, n)) {
        delete[] bufchar;
        throw std::runtime_error("deserialization failed");
    }

    delete[] bufchar;
    return input;
}

