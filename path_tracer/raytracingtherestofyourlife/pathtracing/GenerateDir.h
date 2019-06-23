#ifndef GenerateDir_H
#define GenerateDir_H

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/CoordinateSystem.h>
#include "raytracing/Ray.h"

class GenerateDir
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

  vtkm::cont::ArrayHandle<vtkm::Id> light_indices;
  vtkm::cont::ArrayHandle<vtkm::UInt32> seeds;
  vtkm::cont::ArrayHandle<int> whichPdf;

  vtkm::cont::CoordinateSystem coordsHandle;
  virtual void apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays) = 0;
  virtual void SetData(const vtkm::cont::CoordinateSystem &coord,
                       vtkm::cont::ArrayHandle<vtkm::UInt32> &_seeds,
                       vtkm::cont::ArrayHandle<vtkm::Id> &light_box_indices,
                       vtkm::cont::ArrayHandle<int> &_whichPdf)
  {
    this->light_indices = light_box_indices;
    this->seeds = _seeds;
    this->coordsHandle = coord;
    this->whichPdf = _whichPdf;
  }


};

#endif
