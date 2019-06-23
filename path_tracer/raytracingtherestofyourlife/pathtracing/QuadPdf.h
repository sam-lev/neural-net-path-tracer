#ifndef QUADPDF_H
#define QUADPDF_H
#include "Pdf.h"
#include <vtkm/cont/ArrayHandle.h>
#include "raytracing/Ray.h"

#include "PdfWorklet.h"
#include <vtkm/worklet/Invoker.h>

class QuadPdf : public Pdf
{
public:
  QuadPdf(vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>>  _QuadIds,
          vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>> _light_pointids)
    :QuadIds(_QuadIds)
  , light_pointids(_light_pointids){}

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>>  QuadIds, light_pointids;

  void apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays)
  {
    vtkm::worklet::Invoker Invoke;
    QuadPDFWorklet quadPDFWorklet(this->lightables);
    QuadExecWrapper quadSurf(QuadIds, MatIdx, TexIdx);
    using vec3CompositeType = vtkm::cont::ArrayHandleCompositeVector<
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>>;
    auto generated_dir = vec3CompositeType(
          rays.GetBuffer("generated_dirX").Buffer,rays.GetBuffer("generated_dirY").Buffer,rays.GetBuffer("generated_dirZ").Buffer);

    auto sum_values = rays.GetBuffer("sum_values").Buffer;
    auto hrecs = HitRecord(rays.U, rays.V, rays.Distance, rays.NormalX, rays.NormalY, rays.NormalZ, rays.IntersectionX, rays.IntersectionY, rays.IntersectionZ);

    Invoke(quadPDFWorklet, rays.Origin, rays.Dir,hrecs,
           rays.Status, sum_values, generated_dir, seeds, quadSurf,
            light_pointids, this->light_indices, this->coordsHandle);

  }

};

#endif
