/*
 * Chen Zhou (zhouch@pku.edu.cn)
 */
#include "2dobb.h"

bool OBB2D::overlaps1Way(const OBB2D &other) const {
    for (int a = 0; a < 2; ++a) {

        double t = other.corner[0].dot(axis[a]);

        // Find the extent of box 2 on axis a
        double tMin = t;
        double tMax = t;

        for (int c = 1; c < 4; ++c) {
            t = other.corner[c].dot(axis[a]);

            if (t < tMin) {
                tMin = t;
            } else if (t > tMax) {
                tMax = t;
            }
        }

        // We have to subtract off the origin

        // See if [tMin, tMax] intersects [0, 1]
        if ((tMin > 1 + origin[a]) || (tMax < origin[a])) {
            // There was no intersection along this dimension;
            // the boxes cannot possibly overlap.
            return false;
        }
    }

    // There was no dimension along which there is no intersection.
    // Therefore the boxes overlap.
    return true;
}


//void OBB2D::computeAxes() {
//    axis[0] = corner[1] - corner[0];
//    axis[1] = corner[3] - corner[0];
//
//    // Make the length of each axis 1/edge length so we know any
//    // dot product must be less than 1 to fall within the edge.
//
//    for (int a = 0; a < 2; ++a) {
//        axis[a] /= axis[a].squaredLength();
//        origin[a] = corner[0].dot(axis[a]);
//    }
//}
//
//
//OBB2D::OBB2D(const OBB2D::Vector2 &center, const double w, const double h, double angle)
//{
//    Vector2 X( cos(angle), sin(angle));
//    Vector2 Y(-sin(angle), cos(angle));
//
//    X *= w / 2;
//    Y *= h / 2;
//
//    corner[0] = center - X - Y;
//    corner[1] = center + X - Y;
//    corner[2] = center + X + Y;
//    corner[3] = center - X + Y;
//
//    computeAxes();
//}

OBB2D::OBB2D(const OBB2D::Vector2 &center, const OBB2D::Vector2& unit_axis_x, const OBB2D::Vector2& unit_axis_y, const OBB2D::Vector2& size)
{
    Vector2 X = unit_axis_x;
    Vector2 Y = unit_axis_y;

    float w = size[0];
    float h = size[1];

    X *= w / 2;
    Y *= h / 2;

    corner[0] = center - X - Y;
    corner[1] = center + X - Y;
    corner[2] = center + X + Y;
    corner[3] = center - X + Y;

    axis[0] = unit_axis_x / w;
    axis[1] = unit_axis_y / h;
    origin[0] = corner[0].dot(axis[0]);
    origin[1] = corner[0].dot(axis[1]);
}

//void OBB2D::moveTo(const OBB2D::Vector2 &center) {
//    Vector2 centroid = (corner[0] + corner[1] + corner[2] + corner[3]) / 4;
//
//    Vector2 translation = center - centroid;
//
//    for (int c = 0; c < 4; ++c) {
//        corner[c] += translation;
//    }
//
//    computeAxes();
//}
