#ifndef WORKLETS_H
#define WORKLETS_H

#include <vtkm/worklet/Invoker.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include "wangXor.h"

using vec3 = vtkm::Vec<vtkm::Float32, 3>;

class UVGen : public vtkm::worklet::WorkletMapField
{
public:
  vtkm::UInt32 RayCount;
  VTKM_CONT

  UVGen(vtkm::Id _nx,
        vtkm::Id _ny,
        vtkm::UInt32 rayCount)
    : numx(_nx),
      numy(_ny),
      RayCount(rayCount)
  {
  }

  using ControlSignature = void(FieldInOut<>, FieldOut<>);

  using ExecutionSignature = void(WorkIndex, _1, _2);
  template <typename Precision>
  VTKM_EXEC void operator()(vtkm::Id &idx,
                            unsigned int &seed,
                            Precision& uv) const
  {


    vtkm::Id i = idx % numx;
    vtkm::Id j = idx / numx;
    uv[0] = vtkm::Float32(i + xorshiftWang::getRandF(seed)) / numx;
    uv[1] = vtkm::Float32(j + xorshiftWang::getRandF(seed)) / numy;

  }

  const vtkm::Id numx, numy;
};


class RayGen : public vtkm::worklet::WorkletMapField
{

public:
  vtkm::Int32 numx;
  vtkm::Int32 numy;
  vtkm::Int32 Minx;
  vtkm::Int32 Miny;
  vtkm::Int32 SubsetWidth;
  vtkm::Vec<vtkm::Float32, 3> nlook; // normalized look
  vtkm::Vec<vtkm::Float32, 3> delta_x;
  vtkm::Vec<vtkm::Float32, 3> delta_y;
  vtkm::UInt32 RayCount;
  VTKM_CONT
  RayGen(vtkm::Int32 _nx,
                    vtkm::Int32 _ny,
                    vtkm::Float32 fovX,
                    vtkm::Float32 fovY,
                    vtkm::Vec<vtkm::Float32, 3> look,
                    vtkm::Vec<vtkm::Float32, 3> up,
                    vtkm::Float32 _zoom,
                    vtkm::Int32 subsetWidth,
                    vtkm::Int32 minx,
                    vtkm::Int32 miny,
                    vtkm::UInt32 rayCount)
    : numx(_nx)
    , numy(_ny)
    , Minx(minx)
    , Miny(miny)
    , SubsetWidth(subsetWidth)
    , RayCount(rayCount)
  {
    vtkm::Float32 thx = tanf((fovX * vtkm::Pi_180f()) * .5f);
    vtkm::Float32 thy = tanf((fovY * vtkm::Pi_180f()) * .5f);
    if(thx != thx)
        thx = 0.f;
    if(thy != thy)
        thy = 0.f;
    vtkm::Vec<vtkm::Float32, 3> u = vtkm::Cross(look, up);
    vtkm::Normalize(u);

    vtkm::Vec<vtkm::Float32, 3> v = vtkm::Cross(u, look);
    vtkm::Normalize(v);
    delta_x = u * (2 * thx / (float)numx);
    delta_y = v * (2 * thy / (float)numy);

    if (_zoom > 0)
    {
      delta_x[0] = delta_x[0] / _zoom;
      delta_x[1] = delta_x[1] / _zoom;
      delta_x[2] = delta_x[2] / _zoom;
      delta_y[0] = delta_y[0] / _zoom;
      delta_y[1] = delta_y[1] / _zoom;
      delta_y[2] = delta_y[2] / _zoom;
    }
    nlook = look;
    vtkm::Normalize(nlook);
  }

  using ControlSignature = void(FieldOut<>, FieldOut<>, FieldOut<>, FieldInOut<>, FieldOut<>);

  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5);

  template <typename Precision>
  VTKM_EXEC void operator()(vtkm::Id idx,
                            Precision& rayDirX,
                            Precision& rayDirY,
                            Precision& rayDirZ,
                            unsigned int &seed,
                            vtkm::Id& pixelIndex) const
  {

    vtkm::Vec<Precision, 3> ray_dir(rayDirX, rayDirY, rayDirZ);
    int i = vtkm::Int32(idx) % SubsetWidth;
    int j = vtkm::Int32(idx) / SubsetWidth;
    i += Minx;
    j += Miny;
    // Write out the global pixelId
    pixelIndex = static_cast<vtkm::Id>(j * numx + i);
    // ray_dir = nlook + delta_x * ((2.f * Precision(i) - Precision(w)) / 2.0f) +
      // delta_y * ((2.f * Precision(j) - Precision(h)) / 2.0f);

    Precision _randU = xorshiftWang::getRandF(seed);
    Precision _randV = xorshiftWang::getRandF(seed);

    if (RayCount < 2) {_randU = 0.5f; _randV = 0.5f;}

    ray_dir = nlook + delta_x * ((2.f * (Precision(i) + (1.f - _randU)) - Precision(numx)) / 2.0f) +
      delta_y * ((2.f * (Precision(j) + (_randV)) - Precision(numy)) / 2.0f);

      // if (idx == 15427)
      // {
        // DBGVAR(ray_dir);
      // }
    // avoid some numerical issues
    for (vtkm::Int32 d = 0; d < 3; ++d)
    {
      if (ray_dir[d] == 0.f)
        ray_dir[d] += 0.0000001f;
    }
    Precision dot = vtkm::Dot(ray_dir, ray_dir);
    Precision sq_mag = vtkm::Sqrt(dot);

    rayDirX = ray_dir[0] / sq_mag;
    rayDirY = ray_dir[1] / sq_mag;
    rayDirZ = ray_dir[2] / sq_mag;
  }

}; // class perspective ray gen




#endif
