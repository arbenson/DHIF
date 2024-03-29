#pragma once
#ifndef DHIF_NUMTNS_HPP
#define DHIF_NUMTNS_HPP 1

#include "assert.h"

namespace DHIF {

template <class F>
class NumTns {
public:
    NumTns(int m=0, int n=0, int p=0): _m(m), _n(n), _p(p), _owndata(true) {
#ifndef RELEASE
        CallStackEntry entry("NumTns::NumTns");
#endif
        Allocate();
    }

    NumTns(int m, int n, int p, bool owndata, F* data): _m(m), _n(n), _p(p),
                                                        _owndata(owndata) {
#ifndef RELEASE
        CallStackEntry entry("NumTns::NumTns");
#endif
        if (_owndata) {
            Allocate();
            if (ValidDimensions()) {
                for (int i = 0; i < _m * _n * _p; ++i) {
                    _data[i] = data[i];
                }
            }
        } else {
            _data = data;
        }
    }

    NumTns(const NumTns& C): _m(C._m), _n(C._n), _p(C._p),
                             _owndata(C._owndata) {
#ifndef RELEASE
        CallStackEntry entry("NumTns::NumTns");
#endif
        if (_owndata) {
            Allocate();
            Fill(C);
        } else {
            _data = C._data;
        }
    }

    ~NumTns() {
#ifndef RELEASE
        CallStackEntry entry("NumTns::~NumTns");
#endif
        if (_owndata)
            Deallocate();
    }

    NumTns& operator=(const NumTns& C) {
#ifndef RELEASE
        CallStackEntry entry("NumTns::operator=");
#endif
        if (_owndata)
            Deallocate();
        _m = C._m;
        _n = C._n;
        _p = C._p;
        _owndata = C._owndata;
        if (_owndata) {
            Allocate();
            Fill(C);
        } else {
            _data = C._data;
        }
        return *this;
    }

    void Resize(int m, int n, int p)  {
#ifndef RELEASE
        CallStackEntry entry("NumTns::Resize");
#endif
        assert( _owndata );
        if (_m != m || _n != n || _p != p) {
            Deallocate();
            _m = m;
            _n = n;
            _p = p;
            Allocate();
        }
    }

    const F& operator()(int i, int j, int k) const  {
#ifndef RELEASE
        CallStackEntry entry("NumTns::operator()");

#endif
        if (!( i >= 0 && i < _m && j >= 0 && j < _n && k >= 0 && k < _p)) {
	    std::cout << i << " " << j << " " << k << std::endl;
	}

        assert( i >= 0 && i < _m && j >= 0 && j < _n && k >= 0 && k < _p);
        return _data[i + j * _m + k * _m * _n];
    }
    F& operator()(int i, int j, int k)  {
#ifndef RELEASE
        CallStackEntry entry("NumTns::operator()");
#endif
        if (!( i >= 0 && i < _m && j >= 0 && j < _n && k >= 0 && k < _p)) {
	    std::cout << i << " " << j << " " << k << std::endl;
	}
        assert( i >= 0 && i < _m && j >= 0 && j < _n && k >= 0 && k < _p);
        return _data[i + j * _m + k * _m * _n];
    }

    const F& operator()(Index3 ind) const {
#ifndef RELEASE
        CallStackEntry entry("NumTns::operator()");
#endif
	return this->operator()(ind(0), ind(1), ind(2));
    }

    F& operator()(Index3 ind) {
#ifndef RELEASE
        CallStackEntry entry("NumTns::operator()");
#endif
	return this->operator()(ind(0), ind(1), ind(2));
    }

    F* data() const { return _data; }
    int m() const { return _m; }
    int n() const { return _n; }
    int p() const { return _p; }

private:
    int _m, _n, _p;
    bool _owndata;
    F* _data;

    inline bool ValidDimensions() const { return _m > 0 && _n > 0 && _p > 0; }

    void Allocate() {
#ifndef RELEASE
        CallStackEntry entry("NumTns::Allocate");
#endif
        if (ValidDimensions()) {
            _data = new F[_m * _n * _p];
            assert( _data != NULL );
        } else {
            _data = NULL;
        }
    }

    void Deallocate() {
#ifndef RELEASE
        CallStackEntry entry("NumTns::Deallocate");
#endif
        if (ValidDimensions()) {
            delete[] _data;
            _data = NULL;
        }
    }

    void Fill(const NumTns& C) {
#ifndef RELEASE
        CallStackEntry entry("NumTns::Fill");
#endif
        if (ValidDimensions()) {
            for (int i = 0; i < _m * _n * _p; ++i) {
                _data[i] = C._data[i];
            }
        }
    }
};

template <class F> inline std::ostream& operator<<(std::ostream& os,
                                                   const NumTns<F>& tns) {
#ifndef RELEASE
    CallStackEntry entry("operator<<");
#endif
    os << tns.m() << " " << tns.n() << " " << tns.p() << std::endl;
    os.setf(std::ios_base::scientific, std::ios_base::floatfield);
    for (int i = 0; i < tns.m(); ++i) {
        for (int j = 0; j < tns.n(); ++j) {
            for (int k = 0; k < tns.p(); ++k) {
                os << " " << tns(i,j,k);
            }
            os << std::endl;
        }
        os << std::endl;
    }
    return os;
}

template <class F> inline void SetValue(NumTns<F>& T, F val) {
#ifndef RELEASE
    CallStackEntry entry("setvalue");
#endif
    for (int i = 0; i < T.m(); ++i) {
        for (int j = 0; j < T.n(); ++j) {
            for (int k = 0; k < T.p(); ++k) {
                T(i,j,k) = val;
            }
        }
    }
  return;
}

template <class F> inline double Energy(NumTns<F>& T) {
#ifndef RELEASE
    CallStackEntry entry("energy");
#endif
  double sum = 0;
  for (int i = 0; i < T.m(); ++i) {
      for (int j = 0; j < T.n(); ++j) {
          for (int k = 0; k < T.p(); ++k) {
              sum += abs(T(i,j,k) * T(i,j,k));
          }
      }
  }
  return sum;
}

template <class F> inline double NumTnsSum(NumTns<F>& T) {
#ifndef RELEASE
    CallStackEntry entry("energy");
#endif
  double sum = 0;
  for (int i = 0; i < T.m(); ++i) {
      for (int j = 0; j < T.n(); ++j) {
          for (int k = 0; k < T.p(); ++k) {
              sum += T(i,j,k);
          }
      }
  }
  return sum;
}

typedef NumTns<bool>   BolNumTns;
typedef NumTns<int>    IntNumTns;
typedef NumTns<double> DblNumTns;
typedef NumTns<std::complex<double> >    CpxNumTns;

}
#endif  // _NUMTNS_HPP_
