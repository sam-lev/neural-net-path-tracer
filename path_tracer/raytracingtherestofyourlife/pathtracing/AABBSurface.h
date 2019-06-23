#ifndef AABBSURFACE_H
#define AABBSURFACE_H
#include <vtkm/rendering/raytracing/RayTracingTypeDefs.h>
namespace detail
{

#define QUAD_AABB_EPSILON 1.0e-4f
class FindQuadAABBs : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  FindQuadAABBs() {}
  typedef void ControlSignature(FieldIn<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                WholeArrayIn<vtkm::rendering::raytracing::Vec3RenderingTypes>);
  typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8);
  template <typename PointPortalType>
  VTKM_EXEC void operator()(const vtkm::Vec<vtkm::Id, 5> quadId,
                            vtkm::Float32& xmin,
                            vtkm::Float32& ymin,
                            vtkm::Float32& zmin,
                            vtkm::Float32& xmax,
                            vtkm::Float32& ymax,
                            vtkm::Float32& zmax,
                            const PointPortalType& points) const
  {
    // cast to Float32
    vtkm::Vec<vtkm::Float32, 3> q, r, s, t;

    q = static_cast<vtkm::Vec<vtkm::Float32, 3>>(points.Get(quadId[1]));
    r = static_cast<vtkm::Vec<vtkm::Float32, 3>>(points.Get(quadId[2]));
    s = static_cast<vtkm::Vec<vtkm::Float32, 3>>(points.Get(quadId[3]));
    t = static_cast<vtkm::Vec<vtkm::Float32, 3>>(points.Get(quadId[4]));

    xmin = q[0];
    ymin = q[1];
    zmin = q[2];
    xmax = xmin;
    ymax = ymin;
    zmax = zmin;
    xmin = vtkm::Min(xmin, r[0]);
    ymin = vtkm::Min(ymin, r[1]);
    zmin = vtkm::Min(zmin, r[2]);
    xmax = vtkm::Max(xmax, r[0]);
    ymax = vtkm::Max(ymax, r[1]);
    zmax = vtkm::Max(zmax, r[2]);
    xmin = vtkm::Min(xmin, s[0]);
    ymin = vtkm::Min(ymin, s[1]);
    zmin = vtkm::Min(zmin, s[2]);
    xmax = vtkm::Max(xmax, s[0]);
    ymax = vtkm::Max(ymax, s[1]);
    zmax = vtkm::Max(zmax, s[2]);
    xmin = vtkm::Min(xmin, t[0]);
    ymin = vtkm::Min(ymin, t[1]);
    zmin = vtkm::Min(zmin, t[2]);
    xmax = vtkm::Max(xmax, t[0]);
    ymax = vtkm::Max(ymax, t[1]);
    zmax = vtkm::Max(zmax, t[2]);

    vtkm::Float32 xEpsilon, yEpsilon, zEpsilon;
    const vtkm::Float32 minEpsilon = 1e-6f;
    xEpsilon = vtkm::Max(minEpsilon, QUAD_AABB_EPSILON * (xmax - xmin));
    yEpsilon = vtkm::Max(minEpsilon, QUAD_AABB_EPSILON * (ymax - ymin));
    zEpsilon = vtkm::Max(minEpsilon, QUAD_AABB_EPSILON * (zmax - zmin));

    xmin -= xEpsilon;
    ymin -= yEpsilon;
    zmin -= zEpsilon;
    xmax += xEpsilon;
    ymax += yEpsilon;
    zmax += zEpsilon;
  }

}; //class FindAABBs

class FindSphereAABBs : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  FindSphereAABBs() {}
  typedef void ControlSignature(FieldIn<>,
                                FieldIn<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                FieldOut<>,
                                WholeArrayIn<vtkm::rendering::raytracing::Vec3RenderingTypes>);
  typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7, _8, _9);
  template <typename PointPortalType>
  VTKM_EXEC void operator()(const vtkm::Id pointId,
                            const vtkm::Float32& radius,
                            vtkm::Float32& xmin,
                            vtkm::Float32& ymin,
                            vtkm::Float32& zmin,
                            vtkm::Float32& xmax,
                            vtkm::Float32& ymax,
                            vtkm::Float32& zmax,
                            const PointPortalType& points) const
  {
    // cast to Float32
    vtkm::Vec<vtkm::Float32, 3> point;
    vtkm::Vec<vtkm::Float32, 3> temp;
    point = static_cast<vtkm::Vec<vtkm::Float32, 3>>(points.Get(pointId));

    temp[0] = radius;
    temp[1] = 0.f;
    temp[2] = 0.f;

    vtkm::Vec<vtkm::Float32, 3> p = point + temp;
    //set first point to max and min
    xmin = p[0];
    xmax = p[0];
    ymin = p[1];
    ymax = p[1];
    zmin = p[2];
    zmax = p[2];

    p = point - temp;
    xmin = vtkm::Min(xmin, p[0]);
    xmax = vtkm::Max(xmax, p[0]);
    ymin = vtkm::Min(ymin, p[1]);
    ymax = vtkm::Max(ymax, p[1]);
    zmin = vtkm::Min(zmin, p[2]);
    zmax = vtkm::Max(zmax, p[2]);

    temp[0] = 0.f;
    temp[1] = radius;
    temp[2] = 0.f;

    p = point + temp;
    xmin = vtkm::Min(xmin, p[0]);
    xmax = vtkm::Max(xmax, p[0]);
    ymin = vtkm::Min(ymin, p[1]);
    ymax = vtkm::Max(ymax, p[1]);
    zmin = vtkm::Min(zmin, p[2]);
    zmax = vtkm::Max(zmax, p[2]);

    p = point - temp;
    xmin = vtkm::Min(xmin, p[0]);
    xmax = vtkm::Max(xmax, p[0]);
    ymin = vtkm::Min(ymin, p[1]);
    ymax = vtkm::Max(ymax, p[1]);
    zmin = vtkm::Min(zmin, p[2]);
    zmax = vtkm::Max(zmax, p[2]);

    temp[0] = 0.f;
    temp[1] = 0.f;
    temp[2] = radius;

    p = point + temp;
    xmin = vtkm::Min(xmin, p[0]);
    xmax = vtkm::Max(xmax, p[0]);
    ymin = vtkm::Min(ymin, p[1]);
    ymax = vtkm::Max(ymax, p[1]);
    zmin = vtkm::Min(zmin, p[2]);
    zmax = vtkm::Max(zmax, p[2]);

    p = point - temp;
    xmin = vtkm::Min(xmin, p[0]);
    xmax = vtkm::Max(xmax, p[0]);
    ymin = vtkm::Min(ymin, p[1]);
    ymax = vtkm::Max(ymax, p[1]);
    zmin = vtkm::Min(zmin, p[2]);
    zmax = vtkm::Max(zmax, p[2]);
  }
}; //class FindAABBs

}
#endif
