#ifndef _SIMPLE_ARCHIVE_HPP_
#define _SIMPLE_ARCHIVE_HPP_

#include <map>
#include <type_traits>

template<typename IT,
         typename = typename std::enable_if<std::is_integral<IT>::value, void>::type>
bool writeToBuf(const IT& i, char* buf, size_t bufsize, size_t& writed) {
    size_t n = sizeof(i);

    if (buf != nullptr) {
        if(n > bufsize) {
            return false;
        }
        *(IT*)buf = i;
    }

    writed = n;
    return true;
}

template<typename IT,
         typename = typename std::enable_if<std::is_integral<IT>::value, void>::type>
bool readFromBuf(IT& i, char* buf, size_t bufsize, size_t& read) {
    size_t n = sizeof(i);
    if(n > bufsize) {
        return false;
    }

    i = *(IT*)buf;
    read = n;
    return true;
}

template<typename KT, typename VT>
bool writeToBuf(const std::map<KT,VT>& map, char* buf, size_t bufsize, size_t& writed) {
    bool nowrite = buf == nullptr;
    size_t i = sizeof(size_t);
    if(!nowrite) {
        if (i > bufsize) {
            return false;
        }
        *((size_t*)buf) = map.size();
    }

    for(auto& kv: map) {
        size_t n = 0;
        if(!writeToBuf(kv.first, nowrite ? nullptr : buf + i, bufsize - i, n)) {
            return false;
        };
        i += n;
        if(!writeToBuf(kv.second, nowrite ? nullptr : buf + i, bufsize - i, n)) {
            return false;
        }
        i += n;
    }

    writed = i;
    return true;
}

template<typename KT, typename VT>
bool readFromBuf(std::map<KT,VT>& map, char* buf, size_t bufsize, size_t& read) {
    size_t i = sizeof(size_t);
    if(bufsize < i) {
        return false;
    }

    size_t n = *(size_t*)buf;
    for(;n>0;n--) {
        KT k;
        VT v;
        size_t m;
        if(!readFromBuf(k, buf + i, bufsize - i, m)) {
            return false;
        }
        i += m;
        if(!readFromBuf(v, buf + i, bufsize - i, m)) {
            return false;
        }
        i += m;

        if(map.find(k) != map.end()) {
            return false;
        }
        map.insert(std::make_pair(k, v));
    }

    read = i;
    return true;
}

#endif // _SIMPLE_ARCHIVE_HPP_
