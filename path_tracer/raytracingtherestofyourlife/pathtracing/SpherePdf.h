#ifndef SpherePdf_H
#define SpherePdf_H
#include "Pdf.h"
#include <vtkm/cont/ArrayHandle.h>
#include "raytracing/Ray.h"

#include "PdfWorklet.h"
#include <vtkm/worklet/Invoker.h>

class SpherePdf : public Pdf
{
public:
  SpherePdf(vtkm::cont::ArrayHandle<vtkm::Id>  _SphereIds,
          vtkm::cont::ArrayHandle<vtkm::Id> _light_pointids,
            vtkm::cont::ArrayHandle<vtkm::Float32> radii)
    :SphereIds(_SphereIds)
  , light_pointids(_light_pointids)
  , SphereRadii(radii){}

  vtkm::cont::ArrayHandle<vtkm::Id>  SphereIds, light_pointids;
  vtkm::cont::ArrayHandle<vtkm::Float32>  SphereRadii;

  void apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays)
  {
    vtkm::worklet::Invoker Invoke;
    SpherePDFWorklet spherePDFWorklet(lightables);
    SphereExecWrapper surf(SphereIds, SphereRadii, MatIdx, TexIdx);
    using vec3CompositeType = vtkm::cont::ArrayHandleCompositeVector<
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>>;
    auto generated_dir = vec3CompositeType(
          rays.GetBuffer("generated_dirX").Buffer,rays.GetBuffer("generated_dirY").Buffer,rays.GetBuffer("generated_dirZ").Buffer);

    auto sum_values = rays.GetBuffer("sum_values").Buffer;
    auto hrecs = HitRecord(rays.U, rays.V, rays.Distance, rays.NormalX, rays.NormalY, rays.NormalZ, rays.IntersectionX, rays.IntersectionY, rays.IntersectionZ);


    Invoke(spherePDFWorklet,
           rays.Origin,
           rays.Dir,hrecs,
           rays.Status,
           sum_values,
           generated_dir,
           seeds,
           surf,
           light_pointids,
           light_indices,
           coordsHandle,
           SphereRadii);
  }

};

#endif
