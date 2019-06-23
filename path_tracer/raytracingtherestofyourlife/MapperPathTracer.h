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
#ifndef vtk_m_rendering_MapperPathTracer_h
#define vtk_m_rendering_MapperPathTracer_h

#include <vtkm/cont/ColorTable.h>
#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/Mapper.h>
#include "raytracing/Ray.h"
#include <vtkm/rendering/raytracing/ChannelBuffer.h>
#include <memory>
#include "pathtracing/PathAlgorithms.h"
#include "pathtracing/QuadIntersector.h"

namespace vtkm
{
namespace rendering
{

/**
 * \brief MapperQuad renderers quad facess from a cell set via ray tracing.
 *        As opposed to breaking quads into two trianges, scalars are
 *        interpolated using all 4 points of the quad resulting in more
 *        accurate interpolation.
 */
class VTKM_RENDERING_EXPORT MapperPathTracer : public Mapper
{
public:
  using MyAlgos = ::details::PathAlgorithms<vtkm::cont::DeviceAdapterAlgorithm<VTKM_DEFAULT_DEVICE_ADAPTER_TAG>, VTKM_DEFAULT_DEVICE_ADAPTER_TAG>;

  MapperPathTracer(int sc, int dc,
                   vtkm::cont::ArrayHandle<vtkm::Id> *matIdx,
                   vtkm::cont::ArrayHandle<vtkm::Id> *texIdx,
                   vtkm::cont::ArrayHandle<int> &matType,
                   vtkm::cont::ArrayHandle<int> &texType,
                   vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>> &tex);

  ~MapperPathTracer();

  void SetCanvas(vtkm::rendering::Canvas* canvas) override;
  virtual vtkm::rendering::Canvas* GetCanvas() const override;

  void RenderCells(const vtkm::cont::DynamicCellSet& cellset,
                   const vtkm::cont::CoordinateSystem& coords,
                   const vtkm::cont::Field& scalarField,
                   const vtkm::cont::ColorTable& colorTable,
                   const vtkm::rendering::Camera& camera,
                   const vtkm::Range& scalarRange) override;

  template<typename emittedType,
           typename attenType>
  void intersect(const vtkm::cont::CoordinateSystem &coord,
                 vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                 vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>> &QuadIds,
                 vtkm::cont::ArrayHandle<vtkm::Id> &SphereIds,
                 vtkm::cont::ArrayHandle<vtkm::Float32> &SphereRadii,
                 vtkm::cont::ArrayHandle<vtkm::Int32> &matIdArray,
                 vtkm::cont::ArrayHandle<vtkm::Int32> &texIdArray,
                 vtkm::cont::ArrayHandle<vtkm::Id> *matIdx,
                 vtkm::cont::ArrayHandle<vtkm::Id> *texIdx,
                 vtkm::cont::ArrayHandle<float> &tmin,
                 emittedType &emitted,
                 attenType &attenuation,
                 const vtkm::Id depth) const;

  template<typename HitRecord, typename HitId, typename ScatterRecord,
           typename emittedType>
  void applyMaterials(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                      HitRecord &hrecs,
                      HitId &hids,
                      ScatterRecord &srecs,
                      vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>> tex,
                      vtkm::cont::ArrayHandle<int> matType,
                      vtkm::cont::ArrayHandle<int> texType,
                      emittedType &emitted,
                      vtkm::cont::ArrayHandle<unsigned int> &seeds,
                      vtkm::Id canvasSize,
                      vtkm::Id depth) const;

  template<typename HitRecord, typename ScatterRecord,
           typename attenType, typename GenDirType>
  void applyPDFs(const vtkm::cont::CoordinateSystem &coord,
                 vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>> QuadIds,
                 vtkm::cont::ArrayHandle<vtkm::Id> &SphereIds,
                 vtkm::cont::ArrayHandle<vtkm::Float32> &SphereRadii,
                 vtkm::cont::ArrayHandle<vtkm::Id> *matIdx,
                 vtkm::cont::ArrayHandle<vtkm::Id> *texIdx,
                 vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                 HitRecord &hrecs,
                 ScatterRecord srecs,
                 vtkm::cont::ArrayHandle<vtkm::Float32> &sum_values,
                 GenDirType generated_dir,
                 attenType &attenuation,
                 vtkm::cont::ArrayHandle<unsigned int> &seeds,
                 vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>>  light_box_pointids,
                 vtkm::cont::ArrayHandle<vtkm::Id> light_box_indices,
                 vtkm::cont::ArrayHandle<vtkm::Id> &light_sphere_pointids,
                 vtkm::cont::ArrayHandle<vtkm::Id> & light_sphere_indices,
                 int lightables,
                 vtkm::Id canvasSize,
                 vtkm::Id depth
                 ) const;
  void generateRays(const vtkm::cont::CoordinateSystem &coord,
                    vtkm::cont::ArrayHandle<vtkm::Float32> &SphereRadii,
                    vtkm::cont::ArrayHandle<int> &whichPDF,
                    vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                    vtkm::cont::ArrayHandle<vtkm::UInt32> &seeds,
                    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>>  light_box_pointids,
                    vtkm::cont::ArrayHandle<vtkm::Id> light_box_indices,
                    vtkm::cont::ArrayHandle<vtkm::Id> &light_sphere_pointids,
                    vtkm::cont::ArrayHandle<vtkm::Id> & light_sphere_indices
                    ) const;
  auto extract(const vtkm::cont::DynamicCellSet &cellset) const;
  virtual void StartScene() override;
  virtual void EndScene() override;
  void SetCompositeBackground(bool on);
  vtkm::rendering::Mapper* NewCopy() const override;

  vtkm::cont::ArrayHandleCompositeVector<
        vtkm::cont::ArrayHandle<vtkm::Float32>,
        vtkm::cont::ArrayHandle<vtkm::Float32>,
        vtkm::cont::ArrayHandle<vtkm::Float32>> get_attenuation();

  const int depthcount, samplecount;
  vtkm::cont::ArrayHandle<vtkm::Id> *MatIdx, *TexIdx;
  vtkm::cont::ArrayHandle<int> whichPDF;
  vtkm::cont::ArrayHandle<int> MatType, TexType;
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>> Tex;

  vtkm::rendering::pathtracing::QuadIntersector::ScatterRecord srecs;
  vtkm::rendering::pathtracing::QuadIntersector::HitRecord hrecs;
  vtkm::rendering::pathtracing::QuadIntersector::HitId hids;

private:
  struct InternalsType;
  std::shared_ptr<InternalsType> Internals;

  struct RenderFunctor;

  void RenderCellsImpl(const vtkm::cont::DynamicCellSet& cellset,
                       const vtkm::cont::CoordinateSystem& coords,
                       const vtkm::cont::Field& scalarField,
                       const vtkm::rendering::Camera& camera);

};
}
} //namespace vtkm::rendering

#endif //vtk_m_rendering_MapperQuad_h
