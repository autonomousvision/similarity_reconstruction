#include "obb_intersection.h"
#include <Eigen/Eigen>
#include "2dobb.h"
#include "utility/utility.h"
#include "utility/oriented_boundingbox.h"

namespace tsdf_test {
//check if two oriented bounding boxes overlap
#define VECTOR Eigen::Vector3f
#define SCALAR float
/*
 *  a: half sidelength
 * Pa: center
*/
bool OBBOverlap
(        //A
        VECTOR&	a,	//extents
        VECTOR&	Pa,	//position
        VECTOR*	A,	//orthonormal basis
        //B
        VECTOR&	b,	//extents
        VECTOR&	Pb,	//position
        VECTOR*	B	//orthonormal basis
);

bool TestOBBsIntersection3D(const cpu_tsdf::OrientedBoundingBox &obb, const std::vector<cpu_tsdf::OrientedBoundingBox> &test_obbs)
{
    for (int i = 0; i < test_obbs.size(); ++i)
    {
        if (TestOBBIntersection3D(obb, test_obbs[i]))
            return true;
    }
    return false;
}


bool TestOBBIntersection3D(const cpu_tsdf::OrientedBoundingBox &a, const cpu_tsdf::OrientedBoundingBox &b)
{
    Eigen::Vector3f extent_a = a.bb_sidelengths/2.0;
    Eigen::Vector3f pos_a = a.bb_offset + a.bb_orientation * extent_a;
    Eigen::Vector3f basis_a[3] = {a.bb_orientation.col(0), a.bb_orientation.col(1), a.bb_orientation.col(2)};

    Eigen::Vector3f extent_b = b.bb_sidelengths/2.0;
    Eigen::Vector3f pos_b = b.bb_offset + b.bb_orientation * extent_b;
    Eigen::Vector3f basis_b[3] = {b.bb_orientation.col(0), b.bb_orientation.col(1), b.bb_orientation.col(2)};

    return OBBOverlap(extent_a, pos_a, basis_a,
                      extent_b, pos_b, basis_b);
}

bool TestOBBsIntersection3D(const tsdf_utility::OrientedBoundingBox &obb, const std::vector<tsdf_utility::OrientedBoundingBox> &test_obbs)
{
    for (int i = 0; i < test_obbs.size(); ++i)
    {
        if (TestOBBIntersection3D(obb, test_obbs[i]))
            return true;
    }
    return false;
}

bool TestOBBIntersection3D(
        const tsdf_utility::OrientedBoundingBox& a,
        const tsdf_utility::OrientedBoundingBox& b
        ) {
    Eigen::Vector3f extent_a = a.SideLengths()/2.0;
    // Eigen::Vector3f pos_a = a.Offset()+ a.Orientations() * extent_a;
    Eigen::Vector3f pos_a = a.Offset()+ a.Orientations() * extent_a;
    Eigen::Matrix3f orient_a = a.Orientations();
    Eigen::Vector3f basis_a[3] = {orient_a.col(0), orient_a.col(1), orient_a.col(2)};

    Eigen::Vector3f extent_b = b.SideLengths()/2.0;
    // Eigen::Vector3f pos_b = b.Offset()+ b.Orientbtions() * extent_b;
    Eigen::Vector3f pos_b = b.Offset()+ b.Orientations() * extent_b;
    Eigen::Matrix3f orient_b = b.Orientations();
    Eigen::Vector3f basis_b[3] = {orient_b.col(0), orient_b.col(1), orient_b.col(2)};

    //Eigen::Vector3f extent_b = b.SideLengths()/2.0;
    //Eigen::Vector3f pos_b = b.bb_offset + b.bb_orientation * extent_b;
    //Eigen::Vector3f basis_b[3] = {b.bb_orientation.col(0), b.bb_orientation.col(1), b.bb_orientation.col(2)};

    return OBBOverlap(extent_a, pos_a, basis_a,
                      extent_b, pos_b, basis_b);
}


bool OBBsLargeOverlapArea(const cpu_tsdf::OrientedBoundingBox &obb1, const std::vector<cpu_tsdf::OrientedBoundingBox> &obbs, int *intersected_box, float *intersect_area, const float thresh)
{
    float fintersect_area = 0;
    for (int i = 0; i < obbs.size(); ++i)
    {
        float cur_area = OBBOverlapArea(obb1, obbs[i]) ;
        if (cur_area > thresh)
        {
            if (cur_area > fintersect_area)
            {
                fintersect_area = cur_area;
                if (intersected_box) *intersected_box = i;
            }
        }
    }
    if (fintersect_area > 0)
    {
        if (intersect_area) *intersect_area = fintersect_area;
        return true;
    }
    else
    {
        if (intersect_area) *intersect_area = 0;
        if (intersected_box) *intersected_box = -1;
        return false;
    }

}


double OBBOverlapArea(const cpu_tsdf::OrientedBoundingBox &obb1, const cpu_tsdf::OrientedBoundingBox &obb2)
{
    if (!TestOBBIntersection3D(obb1, obb2)) return 0;
    Eigen::Matrix4f Tr1, Tr2;
    Tr1.setZero();
    Tr1.block(0, 0, 3, 3) = obb1.bb_orientation * obb1.bb_sidelengths.asDiagonal();
    Tr1.block(0, 3, 3, 1) = obb1.BoxCenter();
    Tr1(3, 3) = 1;

    Tr2.setZero();
    Tr2.block(0, 0, 3, 3) = obb2.bb_orientation * obb2.bb_sidelengths.asDiagonal();
    Tr2.block(0, 3, 3, 1) = obb2.BoxCenter();
    Tr2(3, 3) = 1;

    return (unitOverlap(Tr1.inverse() * Tr2) + unitOverlap(Tr2.inverse() * Tr1))/2.0;
}


double unitOverlap(const Eigen::Matrix4f &Tr)
{
    static const int sample_num = 31;
    static const int total_num = sample_num * sample_num * sample_num;
    static std::vector<Eigen::Vector4f> sample_pts;
    if (sample_pts.size() != total_num)
    {
        sample_pts.resize(total_num);
        int cnt = -1;
        for (int x = 0; x < sample_num; ++x)
            for (int y = 0; y < sample_num; ++y)
                for (int z = 0; z < sample_num; ++z)
                {
                    cnt++;
                    sample_pts[cnt] = (Eigen::Vector4f(x, y, z, 0) - Eigen::Vector4f(sample_num/2, sample_num/2, sample_num/2, 0)) / double((sample_num-1));
                    sample_pts[cnt](3) = 1;
                }
    }
    int overlap_cnt = 0;
    for (int i = 0; i < sample_pts.size(); ++i)
    {
        Eigen::Vector4f proj_pt = Tr * sample_pts[i];
        Eigen::Vector3f normed_pt = Eigen::Vector3f(proj_pt[0]/proj_pt[3], proj_pt[1]/proj_pt[3], proj_pt[2]/proj_pt[3]);
        if (fabs(normed_pt[0]) <= 0.5 && fabs(normed_pt[1]) <= 0.5 && fabs(normed_pt[2]) <= 0.5) overlap_cnt++;
    }
    return (double)overlap_cnt/total_num;
}

//check if two oriented bounding boxes overlap
/*
 *  a: half sidelength
 * Pa: center
*/
bool OBBOverlap
(        //A
        VECTOR&	a,	//extents
        VECTOR&	Pa,	//position
        VECTOR*	A,	//orthonormal basis
        //B
        VECTOR&	b,	//extents
        VECTOR&	Pb,	//position
        VECTOR*	B	//orthonormal basis
)
{
    //translation, in parent frame
    VECTOR v = Pb - Pa;
    //translation, in A's frame
    VECTOR T( v.dot(A[0]), v.dot(A[1]), v.dot(A[2]) );

    //B's basis with respect to A's local frame
    SCALAR R[3][3];
    float ra, rb, t;
    long i, k;

    //calculate rotation matrix
    for( i=0 ; i<3 ; i++ )
        for( k=0 ; k<3 ; k++ )
            R[i][k] = A[i].dot(B[k]);
    /*ALGORITHM: Use the separating axis test for all 15 potential
separating axes. If a separating axis could not be found, the two
boxes overlap. */

    //A's basis vectors
    for( i=0 ; i<3 ; i++ )
    {
        ra = a[i];

        rb =
                b[0]*fabs(R[i][0]) + b[1]*fabs(R[i][1]) + b[2]*fabs(R[i][2]);

        t = fabs(T[i]);

        if( t > ra + rb )
            return false;
    }

    //B's basis vectors
    for( k=0 ; k<3 ; k++ )
    {
        ra =
                a[0]*fabs(R[0][k]) + a[1]*fabs(R[1][k]) + a[2]*fabs(R[2][k]);
        rb = b[k];

        t =
                fabs( T[0]*R[0][k] + T[1]*R[1][k] +
                T[2]*R[2][k] );

        if( t > ra + rb )
            return false;
    }

    //9 cross products

    //L = A0 x B0
    ra =
            a[1]*fabs(R[2][0]) + a[2]*fabs(R[1][0]);

    rb =
            b[1]*fabs(R[0][2]) + b[2]*fabs(R[0][1]);

    t =
            fabs( T[2]*R[1][0] -
            T[1]*R[2][0] );

    if( t > ra + rb )
        return false;

    //L = A0 x B1
    ra =
            a[1]*fabs(R[2][1]) + a[2]*fabs(R[1][1]);

    rb =
            b[0]*fabs(R[0][2]) + b[2]*fabs(R[0][0]);

    t =
            fabs( T[2]*R[1][1] -
            T[1]*R[2][1] );

    if( t > ra + rb )
        return false;

    //L = A0 x B2
    ra =
            a[1]*fabs(R[2][2]) + a[2]*fabs(R[1][2]);

    rb =
            b[0]*fabs(R[0][1]) + b[1]*fabs(R[0][0]);

    t =
            fabs( T[2]*R[1][2] -
            T[1]*R[2][2] );

    if( t > ra + rb )
        return false;

    //L = A1 x B0
    ra =
            a[0]*fabs(R[2][0]) + a[2]*fabs(R[0][0]);

    rb =
            b[1]*fabs(R[1][2]) + b[2]*fabs(R[1][1]);

    t =
            fabs( T[0]*R[2][0] -
            T[2]*R[0][0] );

    if( t > ra + rb )
        return false;

    //L = A1 x B1
    ra =
            a[0]*fabs(R[2][1]) + a[2]*fabs(R[0][1]);

    rb =
            b[0]*fabs(R[1][2]) + b[2]*fabs(R[1][0]);

    t =
            fabs( T[0]*R[2][1] -
            T[2]*R[0][1] );

    if( t > ra + rb )
        return false;

    //L = A1 x B2
    ra =
            a[0]*fabs(R[2][2]) + a[2]*fabs(R[0][2]);

    rb =
            b[0]*fabs(R[1][1]) + b[1]*fabs(R[1][0]);

    t =
            fabs( T[0]*R[2][2] -
            T[2]*R[0][2] );

    if( t > ra + rb )
        return false;

    //L = A2 x B0
    ra =
            a[0]*fabs(R[1][0]) + a[1]*fabs(R[0][0]);

    rb =
            b[1]*fabs(R[2][2]) + b[2]*fabs(R[2][1]);

    t =
            fabs( T[1]*R[0][0] -
            T[0]*R[1][0] );

    if( t > ra + rb )
        return false;

    //L = A2 x B1
    ra =
            a[0]*fabs(R[1][1]) + a[1]*fabs(R[0][1]);

    rb =
            b[0] *fabs(R[2][2]) + b[2]*fabs(R[2][0]);

    t =
            fabs( T[1]*R[0][1] -
            T[0]*R[1][1] );

    if( t > ra + rb )
        return false;

    //L = A2 x B2
    ra =
            a[0]*fabs(R[1][2]) + a[1]*fabs(R[0][2]);

    rb =
            b[0]*fabs(R[2][1]) + b[1]*fabs(R[2][0]);

    t =
            fabs( T[1]*R[0][2] -
            T[0]*R[1][2] );

    if( t > ra + rb )
        return false;

    /*no separating axis found,
the two boxes overlap */

    return true;

}


bool TestOBBsIntersection2D(const tsdf_utility::OrientedBoundingBox& obb1, const tsdf_utility::OrientedBoundingBox& obb2)
{
    const Eigen::Vector3f obb1_center = obb1.BottomCenter();
    const Eigen::Vector3f obb1_sidelengths = obb1.SideLengths();
    const Eigen::Matrix3f obb1_orientation = obb1.Orientations();
    const Eigen::Vector2f obb1_center_2d(obb1_center[0], obb1_center[1]);
    const Eigen::Vector2f obb1_axis_x(obb1_orientation(0, 0), obb1_orientation(1, 0));
    const Eigen::Vector2f obb1_axis_y(obb1_orientation(0, 1), obb1_orientation(1, 1));
    const Eigen::Vector2f obb1_side2d(obb1_sidelengths[0], obb1_sidelengths[1]);
    OBB2D obb1_2d(obb1_center_2d, obb1_axis_x, obb1_axis_y, obb1_side2d);

    const Eigen::Vector3f obb2_center = obb2.BottomCenter();
    const Eigen::Vector3f obb2_sidelengths = obb2.SideLengths();
    const Eigen::Matrix3f obb2_orientation = obb2.Orientations();
    const Eigen::Vector2f obb2_center_2d(obb2_center[0], obb2_center[1]);
    const Eigen::Vector2f obb2_axis_x(obb2_orientation(0, 0), obb2_orientation(1, 0));
    const Eigen::Vector2f obb2_axis_y(obb2_orientation(0, 1), obb2_orientation(1, 1));
    const Eigen::Vector2f obb2_side2d(obb2_sidelengths[0], obb2_sidelengths[1]);
    OBB2D obb2_2d(obb2_center_2d, obb2_axis_x, obb2_axis_y, obb2_side2d);
    return obb1_2d.overlaps(obb2_2d);
}


double OBBIntersectionVolume(const tsdf_utility::OrientedBoundingBox& obb1, const tsdf_utility::OrientedBoundingBox& obb2)
{
    // if (!TestOBBIntersection(obb1, obb2)) return 0;
    Eigen::Matrix4f Tr1, Tr2;
    Tr1.setIdentity();
    Tr1.block(0, 0, 3, 4) = obb1.AffineTransformOriginAsCenter().matrix();

    Tr2.setIdentity();
    Tr2.block(0, 0, 3, 4) = obb2.AffineTransformOriginAsCenter().matrix();

    return (unitOverlap(Tr1.inverse() * Tr2) * obb2.Volume() + unitOverlap(Tr2.inverse() * Tr1) * obb1.Volume())/2.0;
}

double OBBIOU(const tsdf_utility::OrientedBoundingBox& obb1, const tsdf_utility::OrientedBoundingBox& obb2)
{
    // if (!TestOBBIntersection(obb1, obb2)) return 0;
    Eigen::Matrix4f Tr1, Tr2;
    Tr1.setIdentity();
    Tr1.block(0, 0, 3, 4) = obb1.AffineTransformOriginAsCenter().matrix();
    double volume1 = obb1.Volume();

    Tr2.setIdentity();
    Tr2.block(0, 0, 3, 4) = obb2.AffineTransformOriginAsCenter().matrix();
    double volume2 = obb2.Volume();

    double overlap_volume = (unitOverlap(Tr1.inverse() * Tr2) * volume2 + unitOverlap(Tr2.inverse() * Tr1) * volume1)/2.0;
    return overlap_volume / (volume1 + volume2 - overlap_volume);
}

double OBBRelativeIntersectionVolume(const tsdf_utility::OrientedBoundingBox& obb1, const tsdf_utility::OrientedBoundingBox& obb2)
{
    // if (!TestOBBIntersection(obb1, obb2)) return 0;
    Eigen::Matrix4f Tr1, Tr2;
    Tr1.setIdentity();
    Tr1.block(0, 0, 3, 4) = obb1.transform().matrix();

    Tr2.setIdentity();
    Tr2.block(0, 0, 3, 4) = obb2.transform().matrix();

    return (unitOverlap(Tr1.inverse() * Tr2) *  + unitOverlap(Tr2.inverse() * Tr1))/2.0;
}

double OBBOverlapIOU2D(const tsdf_utility::OrientedBoundingBox &obb1, const tsdf_utility::OrientedBoundingBox &obb2)
{
    Eigen::Matrix3f Tr1, Tr2;
    Tr1 = obb1.Affine2D().matrix();
    Tr2 = obb2.Affine2D().matrix();
    Eigen::Vector3f tr1_sides = Tr1.colwise().norm();
    Eigen::Vector3f tr2_sides = Tr2.colwise().norm();
    float tr1_area = tr1_sides[0] * tr1_sides[1];
    float tr2_area = tr2_sides[0] * tr2_sides[1];
    float overlap_area = (UnitOverlap2D(Tr1.inverse() * Tr2) * tr1_area + UnitOverlap2D(Tr2.inverse() * Tr1) * tr2_area) / 2.0;
    // return (UnitOverlap2D(Tr1.inverse() * Tr2) + UnitOverlap2D(Tr2.inverse() * Tr1)) / 2.0;
     float iou = overlap_area / (tr1_area + tr2_area - overlap_area);
     return iou;
}

double UnitOverlap2D(const Eigen::Matrix3f &Tr)
{
    static const int sample_num = 31;
    static const int total_num = sample_num * sample_num;
    static std::vector<Eigen::Vector2f> sample_pts;
    if (sample_pts.size() != total_num)  // entering the function for the 1st time
    {
        sample_pts.resize(total_num);
        int cnt = 0;
        for (int x = 0; x < sample_num; ++x)
            for (int y = 0; y < sample_num; ++y)
                {
                    sample_pts[cnt++] = (Eigen::Vector2f(x, y) - Eigen::Vector2f(sample_num/2, sample_num/2)) / float(sample_num-1);
                }
    }
    int overlap_cnt = 0;
    for (int i = 0; i < sample_pts.size(); ++i)
    {
        Eigen::Vector3f proj_pt = Tr * sample_pts[i].homogeneous();
        Eigen::Vector2f normed_pt = proj_pt.hnormalized();
        if (fabs(normed_pt[0]) <= 0.5 && fabs(normed_pt[1]) <= 0.5) overlap_cnt++;
    }
    return (double)overlap_cnt/total_num;
}

bool OBBSimilarity(const tsdf_utility::OrientedBoundingBox &obb1, const tsdf_utility::OrientedBoundingBox &obb2, float *similarity)
{
    const Eigen::Vector3f obb1_orientation = obb1.Orientations().col(0);
    const Eigen::Vector3f obb2_orientation = obb2.Orientations().col(0);
    float orient_cos = obb1_orientation.dot(obb2_orientation);
    if (!TestOBBsIntersection2D(obb1, obb2) ||  orient_cos < M_SQRT1_2) {
        *similarity = 0;
        return false;
    }
    float overlap_percent = OBBIOU(obb1, obb2);
    *similarity = overlap_percent * orient_cos;
    return true;
}

}  // end namespace
