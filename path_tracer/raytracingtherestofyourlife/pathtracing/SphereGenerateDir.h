#ifndef SPHEREGENERATEDIR_H
#define SPHEREGENERATEDIR_H
#include "GenerateDir.h"
#include <vtkm/cont/ArrayHandle.h>
#include "raytracing/Ray.h"

#include "PdfWorklet.h"
#include <vtkm/worklet/Invoker.h>

class SphereGenerateDir : public GenerateDir
{
public:
  SphereGenerateDir(vtkm::cont::ArrayHandle<vtkm::Id> _light_pointids,
                  vtkm::cont::ArrayHandle<vtkm::Float32> &radii)
    : light_pointids(_light_pointids)
  , SphereRadii(radii){}

  vtkm::cont::ArrayHandle<vtkm::Id>  light_pointids;
  vtkm::cont::ArrayHandle<vtkm::Float32> SphereRadii;

  void apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays)
  {
    vtkm::worklet::Invoker Invoke;
    SphereWorkletGenerateDir sphereGenDir(3);

    using vec3CompositeType = vtkm::cont::ArrayHandleCompositeVector<
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>>;
    auto generated_dir = vec3CompositeType(
          rays.GetBuffer("generated_dirX").Buffer,rays.GetBuffer("generated_dirY").Buffer,rays.GetBuffer("generated_dirZ").Buffer);

    auto hrecs = HitRecord(rays.U, rays.V, rays.Distance, rays.NormalX, rays.NormalY, rays.NormalZ, rays.IntersectionX, rays.IntersectionY, rays.IntersectionZ);

    Invoke(sphereGenDir, whichPdf, hrecs, generated_dir, seeds, light_pointids, light_indices, coordsHandle, SphereRadii);

  }




};

#endif
