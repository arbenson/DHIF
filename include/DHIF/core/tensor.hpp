#pragma once
#ifndef DHIF_TENSOR_HPP
#define DHIF_TENSOR_HPP 1

#include "assert.h"

namespace DHIF {

template <class F>
class Tensor {
public:
    Tensor(int m=0, int n=0, int p=0): m_(m), n_(n), p_(p), owndata_(true) {
#ifndef RELEASE
        CallStackEntry entry("Tensor::Tensor");
#endif
        Allocate();
    }

    Tensor(const Tensor& C): m_(C.m_), n_(C.n_), p_(C.p_)
    {
#ifndef RELEASE
        CallStackEntry entry("Tensor::Tensor");
#endif
        owndata_ = false;
        data_ = C.data_;
    }

    ~Tensor() {
#ifndef RELEASE
        CallStackEntry entry("Tensor::~Tensor");
#endif
        if (owndata_)
            Deallocate();
    }

    void Resize(int m, int n, int p)  {
#ifndef RELEASE
        CallStackEntry entry("Tensor::Resize");
#endif
        assert( owndata_ );
        if (m_ != m || n_ != n || p_ != p) {
            Deallocate();
            m_ = m;
            n_ = n;
            p_ = p;
            Allocate();
        }
    }

    const F& operator()(int i, int j, int k) const  {
#ifndef RELEASE
        CallStackEntry entry("Tensor::operator()");

#endif
        if (!( i >= 0 && i < m_ && j >= 0 && j < n_ && k >= 0 && k < p_)) {
	    std::cout << i << " " << j << " " << k << std::endl;
	}

        assert( i >= 0 && i < m_ && j >= 0 && j < n_ && k >= 0 && k < p_);
        return data_[i + j * m_ + k * m_ * n_];
    }
    F& operator()(int i, int j, int k)  {
#ifndef RELEASE
        CallStackEntry entry("Tensor::operator()");
#endif
        if (!( i >= 0 && i < m_ && j >= 0 && j < n_ && k >= 0 && k < p_)) {
	    std::cout << i << " " << j << " " << k << std::endl;
	    }
        assert( i >= 0 && i < m_ && j >= 0 && j < n_ && k >= 0 && k < p_);
        return data_[i + j * m_ + k * m_ * n_];
    }

    const F& operator()(Index3 ind) const {
#ifndef RELEASE
        CallStackEntry entry("Tensor::operator()");
#endif
	return this->operator()(ind(0), ind(1), ind(2));
    }

    F& operator()(Index3 ind) {
#ifndef RELEASE
        CallStackEntry entry("Tensor::operator()");
#endif
	return this->operator()(ind(0), ind(1), ind(2));
    }

    int M() const { return m_; }
    int N() const { return n_; }
    int P() const { return p_; }

private:
    int m_, n_, p_;
    bool owndata_;
    std::vector<F> data_;

    inline bool ValidDimensions() const { return m_ > 0 && n_ > 0 && p_ > 0; }

    void Allocate() {
#ifndef RELEASE
        CallStackEntry entry("Tensor::Allocate");
#endif
        if (ValidDimensions()) {
            data_.resize(m_ * n_ * p_);
        }
    }

    void Deallocate() {
#ifndef RELEASE
        CallStackEntry entry("Tensor::Deallocate");
#endif
        if (ValidDimensions()) {
            data_.resize(0);
        }
    }

};

}
#endif  // _TENSOR_HPP_
