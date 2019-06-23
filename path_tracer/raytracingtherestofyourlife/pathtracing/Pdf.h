#ifndef PDF_H
#define PDF_H

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/CoordinateSystem.h>
#include "raytracing/Ray.h"

class Pdf
{
public:

  using HitRecord = vtkm::cont::ArrayHandleCompositeVector<vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>,
  vtkm::cont::ArrayHandle<vtkm::Float32>>;

  vtkm::cont::ArrayHandle<vtkm::Id> MatIdx, TexIdx;
  vtkm::cont::ArrayHandle<vtkm::Id> light_indices;
  int lightables;
  vtkm::cont::ArrayHandle<vtkm::UInt32> seeds;

  vtkm::cont::CoordinateSystem coordsHandle;
  virtual void apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays) = 0;
  virtual void SetData(const vtkm::cont::CoordinateSystem &coord,
                       vtkm::cont::ArrayHandle<vtkm::Id> matIdx,
                       vtkm::cont::ArrayHandle<vtkm::Id> texIdx,
                       vtkm::cont::ArrayHandle<vtkm::UInt32> &_seeds,
                       vtkm::cont::ArrayHandle<vtkm::Id> &light_box_indices,
                       int _lightables)
  {
    this->MatIdx = matIdx;
    this->TexIdx = texIdx;
    this->light_indices = light_box_indices;
    this->lightables = _lightables;
    this->seeds = _seeds;
    this->coordsHandle = coord;
  }


};

#endif
