#include "DHIF.hpp"

#include <bitset>

namespace DHIF {

//--------------------------------------------------------------------------//
// Public non-static routines                                               //
//--------------------------------------------------------------------------//
template<typename Scalar>
DistHIFDE3D<Scalar>::DistHIFDE3D
( int xSize, int ySize, int zSize )
: xOffset_(1), yOffset_(1), zOffset_(1),
  xSize_(xSize), ySize_(ySize), zSize_(zSize),
  DOF_(xSize*ySize*zSize+xSize*ySize+xSize*zSize+ySize*zSize),
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
  xOffset_(xOffset), yOffset_(yOffset), zOffset_(zOffset),
  xSize_(xSize), ySize_(ySize), zSize_(zSize),
  DOF_(xSize*ySize*zSize+xSize*ySize+xSize*zSize+ySize*zSize),
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

    grid_ = new Grid( team );

    int subteam = -1;
    if( teamSize >= 8 )
        subteam = teamRank/(teamSize/8);

    if( numLevels_ == 0 )
    {
        // Push interior points into globalPosList
        DistMatrix<Index3d,STAR,VC>& globalPosI = globalPosList_[INTERIOR];
        globalPosI.SetGrid( grid_ );
        globalPosI.ResizeTo( 1, xSize_*ySize_*zSize_ );
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
                for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
                {
                    Index3 tmp(itx,ity,itz);
                    globalPosI[Tns2PT(tmp)] = tmp;
                }

        // Push UP frontal points into globalPosList
        DistMatrix<Index3d,STAR,VC>& globalPosU = globalPosList_[UP];
        globalPosU.SetGrid( grid_ );
        globalPosU.ResizeTo( 1, xSize_*ySize_ );
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            {
                Index3 tmp(itx,ity,zOffset_-1);
                globalPosU[Tns2PT(tmp)] = tmp;
            }

        // Push LEFT frontal points into globalPosList
        DistMatrix<Index3d,STAR,VC>& globalPosL = globalPosList_[LEFT];
        globalPosL.SetGrid( grid_ );
        globalPosL.ResizeTo( 1, xSize_*zSize_ );
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            {
                Index3 tmp(itx,yOffset_-1,itz);
                globalPosL[Tns2PT(tmp)] = tmp;
            }

        // Push FRONT frontal points into globalPosList
        DistMatrix<Index3d,STAR,VC>& globalPosF = globalPosList_[FRONT];
        globalPosF.SetGrid( grid_ );
        globalPosF.ResizeTo( 1, ySize_*zSize_ );
        for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            {
                Index3 tmp(xOffset_-1,ity,itz);
                globalPosF[Tns2PT(tmp)] = tmp;
            }

        std::vector<int> size(8);
        size[0] = xSize_*ySize_*zSize_;
        size[1] = xSize_*ySize_;
        size[2] = xSize_*ySize_;
        size[3] = xSize_*zSize_;
        size[4] = xSize_*zSize_;
        size[5] = ySize_*zSize_;
        size[6] = ySize_*zSize_;
        size[7] = 4*(xSize_+ySize_+zSize_);


        // Initial INTERIOR relation
        for( int i=0; i<8; ++i )
        {
            DistMatrix<Scalar,STAR,VC>& A_I = A_[i][INTERIOR];
            A_I.SetGrid( grid_ );
            A_I.ResizeTo( size[i], xSize_*ySize_*zSize_ );
        }

        // Initial UP relation
        for( int i=0; i<8; ++i )
        {
            DistMatrix<Scalar,STAR,VC>& A_U = A_[i][UP];
            A_U.SetGrid( grid_ );
            A_U.ResizeTo( size[i], xSize_*ySize_ );
        }

        // Initial LEFT relation
        for( int i=0; i<8; ++i )
        {
            DistMatrix<Scalar,STAR,VC>& A_L = A_[i][LEFT];
            A_L.SetGrid( grid_ );
            A_L.ResizeTo( size[i], xSize_*zSize_ );
        }

        // Initial FRONT relation
        for( int i=0; i<8; ++i )
        {
            DistMatrix<Scalar,STAR,VC>& A_F = A_[i][FRONT];
            A_F.SetGrid( grid_ );
            A_F.ResizeTo( size[i], ySize_*zSize_ );
        }

        // Set Interior interaction matrix A_I, the order is same as
        // globalPosList_
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
                for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
                {
                    Scalar cv = V(itx,ity,itz);

                    Index3 cp(itx,ity,itz);
                    Index3 cnp;
                    PointType cpPT = Tns2PT(cp);
                    int cpIdx = Tns2Gen(cp);

                    cnp = cp;
                    cnp[0] = cnp[0]-1;
                    A_[Tns2PT(cnp)][cpPT].Set
                    ( Tns2Gen(cnp), cpIdx,
                      -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[0] = (cnp[0]+1)%xSizeGlobal_;
                    A_[Tns2PT(cnp)][cpPT].Set
                    ( Tns2Gen(cnp), cpIdx,
                      -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[1] = cnp[1]-1;
                    A_[Tns2PT(cnp)][cpPT].Set
                    ( Tns2Gen(cnp), cpIdx,
                      -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[1] = (cnp[1]+1)%ySizeGlobal_;
                    A_[Tns2PT(cnp)][cpPT].Set
                    ( Tns2Gen(cnp), cpIdx,
                      -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[2] = cnp[2]-1;
                    A_[Tns2PT(cnp)][cpPT].Set
                    ( Tns2Gen(cnp), cpIdx,
                      -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    cnp = cp;
                    cnp[2] = (cnp[2]+1)%zSizeGlobal_;
                    A_[Tns2PT(cnp)][cpPT].Set
                    ( Tns2Gen(cnp), cpIdx,
                      -(A(cp)+A(cnp))/Scalar(2.0));
                    cv += (A(cp)+A(cnp))/Scalar(2.0);

                    A_[cpPT][cpPT].Set
                    ( cpIdx, cpIdx, cv );
                }

        // Set UP frontal interaction matrix A_U, the order is same as
        // globalPosList_
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            {
                int itz = zOffset_-1;
                Scalar cv = V(itx,ity,itz);

                Index3 cp(itx,ity,itz);
                Index3 cnp;
                PointType cpPT = Tns2PT(cp);
                int cpIdx = Tns2Gen(cp);

                cnp = cp;
                cnp[0] = cnp[0]-1;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[0] = (cnp[0]+1)%xSizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[1] = cnp[1]-1;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[1] = (cnp[1]+1)%ySizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[2] = (cnp[2]+1)%zSizeGlobal_;
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[2] = (cnp[2]+1)%zSizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                A_[cpPT][cpPT].Set
                ( cpIdx, cpIdx, cv );
            }

        // Set LEFT frontal interaction matrix A_L, the order is same as
        // globalPosList_
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            {
                int ity = yOffset_-1;
                Scalar cv = V(itx,ity,itz);

                Index3 cp(itx,ity,itz);
                Index3 cnp;
                PointType cpPT = Tns2PT(cp);
                int cpIdx = Tns2Gen(cp);

                cnp = cp;
                cnp[0] = cnp[0]-1;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[0] = (cnp[0]+1)%xSizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[1] = (cnp[1]+1)%ySizeGlobal_;
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[1] = (cnp[1]+1)%ySizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[2] = cnp[2]-1;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[2] = (cnp[2]+1)%zSizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                A_[cpPT][cpPT].Set
                ( cpIdx, cpIdx, cv );
            }

        // Set FRONT frontal interaction matrix A_F, the order is same as
        // globalPosList_
        for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            {
                int itx = xOffset_-1;
                Scalar cv = V(itx,ity,itz);

                Index3 cp(itx,ity,itz);
                Index3 cnp;
                PointType cpPT = Tns2PT(cp);
                int cpIdx = Tns2Gen(cp);

                cnp = cp;
                cnp[0] = cnp[0]-1;
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[0] = (cnp[0]+1)%xSizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[1] = cnp[1]-1;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[1] = (cnp[1]+1)%ySizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[2] = cnp[2]-1;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                cnp = cp;
                cnp[2] = (cnp[2]+1)%zSizeGlobal_;
                A_[Tns2PT(cnp)][cpPT].Set
                ( Tns2Gen(cnp), cpIdx,
                  -(A(cp)+A(cnp))/Scalar(2.0));
                cv += (A(cp)+A(cnp))/Scalar(2.0);

                A_[cpPT][cpPT].Set
                ( cpIdx, cpIdx, cv );
            }
        return;
    }
    else if( IsLeader() )
    {
        // Push interior points into globalPosList
        const int xC = xSize_/2 + xOffset_-1;
        const int yC = ySize_/2 + yOffset_-1;
        const int zC = zSize_/2 + zOffset_-1;

        // Push INTERIOR corss into globalPosListsub
        Matrix<Index3d>& globalPosI = globalPosListsub_[INTERIOR];
        globalPosI.ResizeTo( 1, xSize_+ySize_+zSize_-2 );
        {
            int it = 0;
            Index3 tmp(xC,yC,zC);
            globalPosI[it++] = tmp;
        for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            if( itx != xC )
            {
                Index3 tmp(itx,yC,zC);
                globalPosI[it++] = tmp;
            }
        for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            if( ity != yC )
            {
                Index3 tmp(xC,ity,zC);
                globalPosI[it++] = tmp;
            }
        for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            if( itz != zC )
            {
                Index3 tmp(xC,yC,itz);
                globalPosI[it++] = tmp;
            }
        }

        // Push UP frontal points into globalPosList
        Matrix<Index3d>& globalPosU = globalPosListsub_[UP];
        globalPosU.ResizeTo( 1, xSize_+ySize_-1 );
        {
            int it = 0;
            {
                Index3 tmp(xC,yC,zOffset_-1);
                globalPosU[it++] = tmp;
            }
            for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            {
                Index3 tmp(itx,yC,zOffset_-1);
                globalPosU[it++] = tmp;
            }
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            {
                Index3 tmp(xC,ity,zOffset_-1);
                globalPosU[it++] = tmp;
            }
        }

        // Push LEFT frontal points into globalPosList
        Matrix<Index3d>& globalPosL = globalPosListsub_[LEFT];
        globalPosL.ResizeTo( 1, xSize_+zSize_-1 );
        {
            int it = 0;
            {
                Index3 tmp(xC,yOffset_-1,zC);
                globalPosL[it++] = tmp;
            }
            for(int itx=xOffset_; itx<xOffset_+xSize_; ++itx)
            {
                Index3 tmp(itx,yOffset_-1,zC);
                globalPosL[it++] = tmp;
            }
            for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            {
                Index3 tmp(xC,yOffset_-1,itz);
                globalPosL[it++] = tmp;
            }
        }

        // Push FRONT frontal points into globalPosList
        Matrix<Index3d>& globalPosF = globalPosListsub_[FRONT];
        globalPosF.ResizeTo( 1, ySize_+zSize_-1 );
        {
            int it = 0;
            {
                Index3 tmp(xOffset_-1,yC,zC);
                globalPosF[it++] = tmp;
            }
            for(int ity=yOffset_; ity<yOffset_+ySize_; ++ity)
            {
                Index3 tmp(xOffset_-1,ity,zC);
                globalPosF[it++] = tmp;
            }
            for(int itz=zOffset_; itz<zOffset_+zSize_; ++itz)
            {
                Index3 tmp(xOffset_-1,yC,itz);
                globalPosF[it++] = tmp;
            }
        }

        // Initial INTERIOR relation
        Matrix<Scalar>& AII = Asub_[INTERIOR];
        AII.SetGrid( grid_ );
        AII.ResizeTo( xSize_+ySize_+zSize_+4, xSize_+ySize_+zSize_-2 );

        // Initial UP relation
        Matrix<Scalar>& AUU = Asub_[UP];
        AUU.SetGrid( grid_ );
        AUU.ResizeTo( xSize_+ySize_+3, xSize_+ySize_-1 );

        // Initial LEFT relation
        Matrix<Scalar>& ALL = Asub_[LEFT];
        ALL.SetGrid( grid_ );
        ALL.ResizeTo( xSize_+zSize_+3, xSize_+zSize_-1 );

        // Initial FRONT relation
        Matrix<Scalar>& AFF = Asub_[FRONT];
        AFF.SetGrid( grid_ );
        AFF.ResizeTo( ySize_+zSize_+3, ySize_+zSize_-1 );

        // Set Interior interaction matrix A_I, the order is same as
        // globalPosList_
        for( int i=0; i < globalPosI.Width(); ++i )
        {
            int itx = globalPosI.Get(1,i)[0];
            int ity = globalPosI.Get(1,i)[1];
            int itz = globalPosI.Get(1,i)[2];
            int cnploc = 0;
            Scalar cv = V(itx,ity,itz);

            Index3 cp(itx,ity,itz);
            Index3 cnp;

            cnp = cp;
            cnp[0] = cnp[0]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AII.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[0] = (cnp[0]+1)%xSizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AII.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = cnp[1]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AII.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = (cnp[1]+1)%ySizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AII.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = cnp[2]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AII.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = (cnp[2]+1)%zSizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AII.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            AII.Set( i, i, cv );
        }

        // Set UP frontal interaction matrix A_U, the order is same as
        // globalPosList_
        for( int i=0; i < globalPosU.Width(); ++i )
        {
            int itx = globalPosU.Get(1,i)[0];
            int ity = globalPosU.Get(1,i)[1];
            int itz = globalPosU.Get(1,i)[2];
            int cnploc = 0;
            Scalar cv = V(itx,ity,itz);

            Index3 cp(itx,ity,itz);
            Index3 cnp;

            cnp = cp;
            cnp[0] = cnp[0]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AUU.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[0] = (cnp[0]+1)%xSizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AUU.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = cnp[1]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AUU.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = (cnp[1]+1)%ySizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AUU.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = cnp[2]-1;
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = (cnp[2]+1)%zSizeGlobal_;
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            AUU.Set( i, i, cv );
        }

        // Set LEFT frontal interaction matrix A_L, the order is same as
        // globalPosList_
        for( int i=0; i < globalPosL.Width(); ++i )
        {
            int itx = globalPosL.Get(1,i)[0];
            int ity = globalPosL.Get(1,i)[1];
            int itz = globalPosL.Get(1,i)[2];
            int cnploc = 0;
            Scalar cv = V(itx,ity,itz);

            Index3 cp(itx,ity,itz);
            Index3 cnp;

            cnp = cp;
            cnp[0] = cnp[0]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                ALL.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[0] = (cnp[0]+1)%xSizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                ALL.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = cnp[1]-1;
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = (cnp[1]+1)%ySizeGlobal_;
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = cnp[2]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                ALL.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = (cnp[2]+1)%zSizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                ALL.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            ALL.Set( i, i, cv );
        }

        // Set FRONT frontal interaction matrix A_F, the order is same as
        // globalPosList_
        for( int i=0; i < globalPosF.Width(); ++i )
        {
            int itx = globalPosF.Get(1,i)[0];
            int ity = globalPosF.Get(1,i)[1];
            int itz = globalPosF.Get(1,i)[2];
            int cnploc = 0;
            Scalar cv = V(itx,ity,itz);

            Index3 cp(itx,ity,itz);
            Index3 cnp;

            cnp = cp;
            cnp[0] = cnp[0]-1;
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[0] = (cnp[0]+1)%xSizeGlobal_;
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = cnp[1]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AFF.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[1] = (cnp[1]+1)%ySizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AFF.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = cnp[2]-1;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AFF.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            cnp = cp;
            cnp[2] = (cnp[2]+1)%zSizeGlobal_;
            cnploc = CrossTns2Gen( cnp, cp );
            if( cnploc >= 0 )
                AFF.Set( cnploc, i,
                    -(A(cp)+A(cnp))/Scalar(2.0));
            cv += (A(cp)+A(cnp))/Scalar(2.0);

            AFF.Set( i, i, cv );
        }
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
