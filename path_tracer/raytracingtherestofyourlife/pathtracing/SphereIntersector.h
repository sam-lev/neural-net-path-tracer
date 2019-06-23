//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2015 UT-Battelle, LLC.
//  Copyright 2015 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================
#ifndef vtk_m_rendering_pathtracing_Sphere_Intersector_h
#define vtk_m_rendering_pathtracing_Sphere_Intersector_h

#include <vtkm/rendering/raytracing/ShapeIntersector.h>

namespace vtkm
{
namespace rendering
{
namespace pathtracing
{

class SphereIntersector : public vtkm::rendering::raytracing::ShapeIntersector
{
protected:
  vtkm::cont::ArrayHandle<vtkm::Id> PointIds;
  vtkm::cont::ArrayHandle<vtkm::Float32> Radii;
  vtkm::cont::ArrayHandle<vtkm::Id> MatIdx, TexIdx;
  vtkm::cont::ArrayHandle<vtkm::Int32> MatIdArray, TexIdArray;

public:
  using ScatterRecord = vtkm::cont::ArrayHandleCompositeVector<vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>>;
  using HitRecord = vtkm::cont::ArrayHandleCompositeVector<vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>>;
  using IdArray = vtkm::cont::ArrayHandle<vtkm::Int32>;
  using HitId = vtkm::cont::ArrayHandleCompositeVector<IdArray, IdArray>;

  SphereIntersector();
  virtual ~SphereIntersector() override;

  void SetData(const vtkm::cont::CoordinateSystem& coords,
               vtkm::cont::ArrayHandle<vtkm::Id> pointIds,
               vtkm::cont::ArrayHandle<vtkm::Float32> radii,
               vtkm::cont::ArrayHandle<vtkm::Id> &matIdx,
               vtkm::cont::ArrayHandle<vtkm::Id> &texIdx,
               IdArray &matIdArray,
               IdArray &texIdArray);


  void IntersectRays(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays, bool returnCellIndex = false) override;


  void IntersectRays(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays, bool returnCellIndex = false) override;

  template <typename Precision>
  void IntersectRaysImp(vtkm::rendering::raytracing::Ray<Precision>& rays, bool returnCellIndex);


//  template <typename Precision>
//  void IntersectionDataImp(vtkm::rendering::raytracing::Ray<Precision>& rays,
//                           const vtkm::cont::Field* scalarField,
//                           const vtkm::Range& scalarRange);

  void IntersectionData(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays,
                        const vtkm::cont::Field* scalarField,
                        const vtkm::Range& scalarRange) override;

  void IntersectionData(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays,
                        const vtkm::cont::Field* scalarField,
                        const vtkm::Range& scalarRange) override;

  vtkm::Id GetNumberOfShapes() const override;
}; // class ShapeIntersector
}
}
} //namespace vtkm::rendering::raytracing
#endif //vtk_m_rendering_raytracing_Shape_Intersector_h
