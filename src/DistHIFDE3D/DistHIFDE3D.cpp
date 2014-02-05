#include "DHIF.hpp"

#include <bitset>

namespace DHIF {

//--------------------------------------------------------------------------//
// Public non-static routines                                               //
//--------------------------------------------------------------------------//
template<typename Scalar>
DistHIFDE3D<Scalar>::DistHIFDE3D
( int xSize, int ySize, int zSize )
: xOffset_(1),
  yOffset_(1),
  zOffset_(1),
  xSize_(xSize),
  ySize_(ySize),
  zSize_(zSize),
  own_(true), level_(0)
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::DistHIFDE3D");
#endif
    numLevels_ = teams_->NumLevels();
}
template<typename Scalar>
DistHIFDE3D<Scalar>::DistHIFDE3D
( int numLevels, int level, int xOffset, int yOffset, int zOffset,
  int xSize, int ySize, int zSize, bool own )
: numLevels_(numLevels),
  xOffset_(xOffset),
  yOffset_(yOffset),
  zOffset_(zOffset),
  xSize_(xSize),
  ySize_(ySize),
  zSize_(zSize),
  own_(own), level_(level)
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::DistHIFDE3D");
#endif
    const int numTeamLevels = teams_->NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error
            ("Too many processes for this small size domain");
}


template<typename Scalar>
DistHIFDE3D<Scalar>::~DistHIFDE3D()
{ Clear(); }

template<typename Scalar>
void
DistHIFDE3D<Scalar>::Clear()
{
    //TODO: Clear all the data in the class.
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar>
void
DistHIFDE3D<Scalar>::BuildTree
( const NumTns<Scalar>& A, const NumTns<Scalar>& V )
{
#ifndef RELEASE
    CallStackEntry entry("DistHIFDE3D::BuildTree");
#endif
    mpi::Comm team = teams_->Team(level_);
    const int teamSize = mpi::CommSize( team );
    const int teamRank = mpi::CommRank( team );

    int subteam = -1;
    if( teamSize >= 8 )
        subteam = teamRank/(teamSize/8);

    if( numLevels_ == 0 )
    {
        // Set degreeList containing the position of degree of freedoms
        degreeList_.Resize(DOF_);
        int it=0;
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
                for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
                {
                    Index3 tmp(itx,ity,itz);
                    degreeList_[it] = tmp;
                    it++;
                }

        // Set self interaction matrix A_self_, the order is same as
        // in degreeList_
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
                for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
                {
                    Scalar cv = V(itx,ity,itz);

                    Index3 cp(itx,ity,itz);
                    Index3 cnp;

                    cnp = cp;
                    cnp[0] -= 1;
                    if( itx != xOffset_ )
                        A_self_.Set
                        ( Tns2Gen(cp),
                          Tns2Gen(cnp),
                          -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[0] += 1;
                    if( itx != xOffset_+xSize_-1 )
                        A_self_.Set
                        ( Tns2Gen(cp),
                          Tns2Gen(cnp),
                          -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[1] -= 1;
                    if( ity != yOffset_ )
                        A_self_.Set
                        ( Tns2Gen(cp),
                          Tns2Gen(cnp),
                          -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[1] += 1;
                    if( ity != yOffset_+ySize_-1 )
                        A_self_.Set
                        ( Tns2Gen(cp),
                          Tns2Gen(cnp),
                          -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[2] -= 1;
                    if( itz != zOffset_ )
                        A_self_.Set
                        ( Tns2Gen(cp),
                          Tns2Gen(cnp),
                          -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[2] += 1;
                    if( itz != zOffset_+zSize_-1 )
                        A_self_.Set
                        ( Tns2Gen(cp),
                          Tns2Gen(cnp),
                          -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);
                }
                // TODO: Create the list with boundary points

        return;
    }
    else
    {
    }

    for(int itx=0; itx<2; ++itx)
        for(int ity=0; ity<2; ++ity)
            for(int itz=0; itz<2; ++itz)
            {
                std::bitset<3> iter;
                iter.set(0,itx);
                iter.set(1,ity);
                iter.set(2,itz);
                const int sonOrder = (int)iter.to_ulong();
                bool ownson = false;
                if( subteam < 0 || subteam == sonOrder )
                    ownson = true;
                // The middle point on each side will be used as face
                // These faces are stored on current level
                // In first half of next level (itx==0),
                // xOffset is same, xSize is (xSize_-1)/2
                // In second half of the next level (itx==1),
                // xOffset is xOffset_+(xSize_+1)/2
                // xSize is xSize_/2
                node_.children[sonOrder] =
                new DistHIFDE3D<Scalar>
                ( numLevels_-1, level_+1,
                  xOffset_+itx*(xSize_+1)/2,
                  yOffset_+ity*(ySize_+1)/2,
                  zOffset_+itz*(zSize_+1)/2,
                  (xSize_+itx-1)/2, (ySize_+ity-1)/2, (zSize_+itz-1)/2,
                  ownson );
                if( ownson )
                    node_.Child(sonOrder).BuildTree(A,V);
            }
}

template class DistHIFDE3D<float>;
template class DistHIFDE3D<double>;
template class DistHIFDE3D<std::complex<float> >;
template class DistHIFDE3D<std::complex<double> >;

} // namespace DHIF
