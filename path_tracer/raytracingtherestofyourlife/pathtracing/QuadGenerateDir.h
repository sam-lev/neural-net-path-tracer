#ifndef QUADGENERATEDIR_H
#define QUADGENERATEDIR_H
#include "GenerateDir.h"
#include <vtkm/cont/ArrayHandle.h>
#include "raytracing/Ray.h"

#include "PdfWorklet.h"
#include <vtkm/worklet/Invoker.h>


class QuadGenerateDir : public GenerateDir
{
public:
  QuadGenerateDir(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>> _light_pointids)
    : light_pointids(_light_pointids){}

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>>  light_pointids;

  void apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays)
  {
    vtkm::worklet::Invoker Invoke;
    QuadWorkletGenerateDir quadGenDir(2);

    using vec3CompositeType = vtkm::cont::ArrayHandleCompositeVector<
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>>;
    auto generated_dir = vec3CompositeType(
          rays.GetBuffer("generated_dirX").Buffer,rays.GetBuffer("generated_dirY").Buffer,rays.GetBuffer("generated_dirZ").Buffer);

    auto hrecs = HitRecord(rays.U, rays.V, rays.Distance, rays.NormalX, rays.NormalY, rays.NormalZ, rays.IntersectionX, rays.IntersectionY, rays.IntersectionZ);

    Invoke(quadGenDir, this->whichPdf, hrecs, generated_dir, seeds, light_pointids, light_indices, coordsHandle);

  }




};

#endif
