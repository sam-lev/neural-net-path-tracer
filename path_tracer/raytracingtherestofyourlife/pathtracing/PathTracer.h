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
#ifndef vtk_m_rendering_pathtracing_PathTracer_h
#define vtk_m_rendering_pathtracing_PathTracer_h

#include <vector>

#include <vtkm/cont/DataSet.h>

#include "Camera.h"
#include <vtkm/rendering/raytracing/TriangleIntersector.h>
namespace vtkm
{
namespace rendering
{
namespace pathtracing
{

class VTKM_RENDERING_EXPORT PathTracer
{
protected:
  std::vector<vtkm::rendering::raytracing::ShapeIntersector*> Intersectors;
  Camera camera;
  const vtkm::cont::Field* ScalarField;
  vtkm::cont::ArrayHandle<vtkm::Float32> Scalars;
  vtkm::Id NumberOfShapes;
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4>> ColorMap;
  vtkm::Range ScalarRange;
  bool Shade;

  template <typename Precision>
  void RenderOnDevice(vtkm::rendering::raytracing::Ray<Precision>& rays);

public:
  VTKM_CONT
  PathTracer();
  VTKM_CONT
  ~PathTracer();

  VTKM_CONT
  Camera& GetCamera();

  VTKM_CONT
  void AddShapeIntersector(vtkm::rendering::raytracing::ShapeIntersector* intersector);

  VTKM_CONT
  void SetField(const vtkm::cont::Field& scalarField, const vtkm::Range& scalarRange);

  VTKM_CONT
  void SetColorMap(const vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4>>& colorMap);

  VTKM_CONT
  void SetShadingOn(bool on);

  VTKM_CONT
  void Render(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays);

  VTKM_CONT
  void Render(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays);

  VTKM_CONT
  vtkm::Id GetNumberOfShapes() const;

  VTKM_CONT
  void Clear();

}; //class PathTracer
}
}
} // namespace vtkm::rendering::raytracing
#endif //vtk_m_rendering_raytracing_PathTracer_h
