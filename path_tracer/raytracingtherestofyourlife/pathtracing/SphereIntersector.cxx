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
#include "SphereIntersector.h"
#include <vtkm/VectorAnalysis.h>
#include <vtkm/cont/Algorithm.h>
#include "BVHTraverser.h"
#include <vtkm/rendering/raytracing/RayOperations.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/DispatcherMapTopology.h>
#include "Surface.h"
#include "AABBSurface.h"
namespace vtkm
{
namespace rendering
{
namespace pathtracing
{


SphereIntersector::SphereIntersector()
  : ShapeIntersector()
{
}

SphereIntersector::~SphereIntersector()
{
}

void SphereIntersector::SetData(const vtkm::cont::CoordinateSystem& coords,
                                vtkm::cont::ArrayHandle<vtkm::Id> pointIds,
                                vtkm::cont::ArrayHandle<vtkm::Float32> radii,
                                vtkm::cont::ArrayHandle<vtkm::Id> &matIdx,
                                vtkm::cont::ArrayHandle<vtkm::Id> &texIdx,
                                IdArray &matIdArray,
                                IdArray &texIdArray)

{
  this->MatIdArray = matIdArray;
  this->TexIdArray = texIdArray;
  this->MatIdx = matIdx;
  this->TexIdx = texIdx;

  this->PointIds = pointIds;
  this->Radii = radii;
  this->CoordsHandle = coords;
  vtkm::rendering::raytracing::AABBs AABB;
  vtkm::worklet::DispatcherMapField<::detail::FindSphereAABBs>(::detail::FindSphereAABBs())
    .Invoke(PointIds,
            Radii,
            AABB.xmins,
            AABB.ymins,
            AABB.zmins,
            AABB.xmaxs,
            AABB.ymaxs,
            AABB.zmaxs,
            CoordsHandle);

  this->SetAABBs(AABB);
}

void SphereIntersector::IntersectRays(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays, bool returnCellIndex)
{
  IntersectRaysImp(rays, returnCellIndex);
}

void SphereIntersector::IntersectRays(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays, bool returnCellIndex)
{
  //IntersectRaysImp(rays, returnCellIndex);
}

template <typename Precision>
void SphereIntersector::IntersectRaysImp(vtkm::rendering::raytracing::Ray<Precision>& rays, bool vtkmNotUsed(returnCellIndex))
{

  SphereExecWrapper sphereIntersect(PointIds, Radii, MatIdx, TexIdx);
  vtkm::rendering::pathtracing::BVHTraverser traverser;
  auto hrecs = HitRecord(rays.U, rays.V, rays.Distance, rays.NormalX, rays.NormalY, rays.NormalZ, rays.IntersectionX, rays.IntersectionY, rays.IntersectionZ);
  auto hids = HitId(MatIdArray, TexIdArray);
  auto tmin = rays.MinDistance;

  traverser.IntersectRays(rays, this->BVH, hrecs, hids, tmin, sphereIntersect, CoordsHandle);

}

//template <typename Precision>
//void SphereIntersector::IntersectionDataImp(vtkm::rendering::raytracing::Ray<Precision>& rays,
//                                            const vtkm::cont::Field* scalarField,
//                                            const vtkm::Range& scalarRange)
//{
//  ShapeIntersector::IntersectionPoint(rays);

//  bool isSupportedField =
//    (scalarField->GetAssociation() == vtkm::cont::Field::Association::POINTS ||
//     scalarField->GetAssociation() == vtkm::cont::Field::Association::CELL_SET);
//  if (!isSupportedField)
//    throw vtkm::cont::ErrorBadValue(
//      "SphereIntersector: Field not accociated with a cell set or field");

//  vtkm::worklet::DispatcherMapField<detail::CalculateNormals>(detail::CalculateNormals())
//    .Invoke(rays.HitIdx,
//            rays.Intersection,
//            rays.NormalX,
//            rays.NormalY,
//            rays.NormalZ,
//            CoordsHandle,
//            PointIds);

//  vtkm::worklet::DispatcherMapField<detail::GetScalar<Precision>>(
//    detail::GetScalar<Precision>(vtkm::Float32(scalarRange.Min), vtkm::Float32(scalarRange.Max)))
//    .Invoke(rays.HitIdx, rays.Scalar, *scalarField, PointIds);
//}

void SphereIntersector::IntersectionData(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays,
                                         const vtkm::cont::Field* scalarField,
                                         const vtkm::Range& scalarRange)
{
//  IntersectionDataImp(rays, scalarField, scalarRange);
}

void SphereIntersector::IntersectionData(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays,
                                         const vtkm::cont::Field* scalarField,
                                         const vtkm::Range& scalarRange)
{
//  IntersectionDataImp(rays, scalarField, scalarRange);
}

vtkm::Id SphereIntersector::GetNumberOfShapes() const
{
  return PointIds.GetNumberOfValues();
}
}
}
} //namespace vtkm::rendering::raytracing
