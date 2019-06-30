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

#include <vtkm/rendering/CanvasRayTracer.h>

#include <vtkm/cont/TryExecute.h>
#include <vtkm/rendering/Canvas.h>
#include <vtkm/rendering/Color.h>
#include <vtkm/rendering/raytracing/Ray.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>

// for scaling depth exponentially
#include <vtkm/Math.h>

#include <iostream>
#include <math.h>

namespace vtkm
{
namespace rendering
{
namespace internal
{
class SurfaceConverter : public vtkm::worklet::WorkletMapField
{
  vtkm::Matrix<vtkm::Float32, 4, 4> ViewProjMat;

public:
  VTKM_CONT
  SurfaceConverter(const vtkm::Matrix<vtkm::Float32, 4, 4> viewProjMat)
    : ViewProjMat(viewProjMat)
  {
  }
  using ControlSignature = void(FieldIn<>,
                                WholeArrayInOut<>,
                                FieldIn<>,
                                FieldIn<>,
                                FieldIn<>,
                                WholeArrayOut<vtkm::ListTagBase<vtkm::Float32>>,
                                WholeArrayOut<vtkm::ListTagBase<vtkm::Vec<vtkm::Float32, 4>>>);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, WorkIndex);
  template <typename Precision,
            typename ColorPortalType,
            typename DepthBufferPortalType,
            typename ColorBufferPortalType>
  VTKM_EXEC void operator()(const vtkm::Id& pixelIndex,
                            ColorPortalType& colorBufferIn,
                            const Precision& inDepth,
                            const vtkm::Vec<Precision, 3>& origin,
                            const vtkm::Vec<Precision, 3>& dir,
                            DepthBufferPortalType& depthBuffer,
                            ColorBufferPortalType& colorBuffer,
                            const vtkm::Id& index) const
  {
    vtkm::Vec<Precision, 3> intersection ;
    if(isinf(inDepth) || isnan(inDepth))
        intersection = origin;
    else
        intersection = origin + inDepth * dir;
    vtkm::Vec<vtkm::Float32, 4> point;

    point[0] = static_cast<vtkm::Float32>(intersection[0]);
    point[1] = static_cast<vtkm::Float32>(intersection[1]);
    point[2] = static_cast<vtkm::Float32>(intersection[2]);
    point[3] = 1.f;

    vtkm::Vec<vtkm::Float32, 4> newpoint;
    newpoint = vtkm::MatrixMultiply(this->ViewProjMat, point);
    newpoint[0] = newpoint[0] / newpoint[3];
    newpoint[1] = newpoint[1] / newpoint[3];
    vtkm::Float32 nan_val = newpoint[2]; //set to 0 and depth 1 if nan or inf
    bool was_nan =false;
    if(newpoint[2] != newpoint[2] ||isinf(newpoint[2]) ){
            newpoint[2] = 0.f;//1.f;//newpoint[3];
            was_nan = true;
    }
    if( isinf(newpoint[3]) || newpoint[3] == 0 || newpoint[3] != newpoint[3]){
        newpoint[3] = 1.f;
    }
    else
        newpoint[2] = newpoint[2] / newpoint[3];

    vtkm::Float32 depth = newpoint[2];

    vtkm::Float32 re_scale = (555.f * vtkm::Abs(555.f - (785.3f*depth +555.f*depth - 785.3f)))/10.f;

    depth = 0.5f*re_scale + .5f;// vtkm::Abs((depth - 0.99999f) / (.9999999f - 0.99999f)); //0.25f * vtkm::Exp(-0.5f * (depth) ) ;// 0.25f * vtkm::Exp((depth) / (800.0f )) + 0.5f;//nonlinearly_scaled_depth(depth);//0.5f * (depth) + 0.5f;

    //std::cout << depth << std::endl;

    vtkm::Vec<vtkm::Float32, 4> color;
    color[0] = static_cast<vtkm::Float32>(colorBufferIn.Get(index * 4 + 0));
    color[1] = static_cast<vtkm::Float32>(colorBufferIn.Get(index * 4 + 1));
    color[2] = static_cast<vtkm::Float32>(colorBufferIn.Get(index * 4 + 2));
    color[3] = static_cast<vtkm::Float32>(colorBufferIn.Get(index * 4 + 3));
    // blend the mapped color with existing canvas color
    vtkm::Vec<vtkm::Float32, 4> inColor = colorBuffer.Get(pixelIndex);


    // if transparency exists, all alphas have been pre-multiplied
    vtkm::Float32 alpha = (1.f - color[3]);
    color[0] = color[0] + inColor[0] * alpha;
    color[1] = color[1] + inColor[1] * alpha;
    color[2] = color[2] + inColor[2] * alpha;
    color[3] = inColor[3] * alpha + color[3];

    // clamp
    for (vtkm::Int32 i = 0; i < 4; ++i)
    {
      color[i] = vtkm::Max(0.f, vtkm::Min(color[i], 1.f));;//vtkm::Min(1.f, vtkm::Max(color[i], 0.f));
    }

    // The existing depth should already been fed into the ray mapper
    // so no color contribution will exist past the existing depth.

    depthBuffer.Set(pixelIndex, depth);
    colorBuffer.Set(pixelIndex, color);
  }
}; //class SurfaceConverter

template <typename Precision>
VTKM_CONT void WriteToCanvas(const vtkm::rendering::raytracing::Ray<Precision>& rays,
                             const vtkm::cont::ArrayHandle<Precision>& colors,
                             const vtkm::rendering::Camera& camera,
                             vtkm::rendering::CanvasRayTracer* canvas)
{
  vtkm::Matrix<vtkm::Float32, 4, 4> viewProjMat =
    vtkm::MatrixMultiply(camera.CreateProjectionMatrix(canvas->GetWidth(), canvas->GetHeight()),
                         camera.CreateViewMatrix());

  vtkm::worklet::DispatcherMapField<SurfaceConverter>(SurfaceConverter(viewProjMat))
    .Invoke(rays.PixelIdx,
            colors,
            rays.Distance,
            rays.Origin,
            rays.Dir,
            canvas->GetDepthBuffer(),
            canvas->GetColorBuffer());

  //Force the transfer so the vectors contain data from device
  canvas->GetColorBuffer().GetPortalControl().Get(0);
  canvas->GetDepthBuffer().GetPortalControl().Get(0);
}

} // namespace internal

CanvasRayTracer::CanvasRayTracer(vtkm::Id width, vtkm::Id height)
  : Canvas(width, height)
{
}

CanvasRayTracer::~CanvasRayTracer()
{
}

void CanvasRayTracer::WriteToCanvas(const vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays,
                                    const vtkm::cont::ArrayHandle<vtkm::Float32>& colors,
                                    const vtkm::rendering::Camera& camera)
{
  internal::WriteToCanvas(rays, colors, camera, this);
}

void CanvasRayTracer::WriteToCanvas(const vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays,
                                    const vtkm::cont::ArrayHandle<vtkm::Float64>& colors,
                                    const vtkm::rendering::Camera& camera)
{
  internal::WriteToCanvas(rays, colors, camera, this);
}

vtkm::rendering::Canvas* CanvasRayTracer::NewCopy() const
{
  return new vtkm::rendering::CanvasRayTracer(*this);
}
}
}
