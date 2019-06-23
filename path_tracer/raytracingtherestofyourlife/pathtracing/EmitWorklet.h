#ifndef EMITWORKLET_H
#define EMITWORKLET_H
#include <vtkm/worklet/Invoker.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include "Record.h"
#include "vec3.h"
#include "wangXor.h"

class LambertianWorklet : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  LambertianWorklet(vtkm::Id cs, vtkm::Id d)
      : canvasSize(cs)
      , depth(d)
  {
  }

  template<typename HitRecord, typename ScatterRecord>
  VTKM_EXEC
  bool scatter(const vec3 &origin, const vec3 &direction, const HitRecord& hrec, ScatterRecord& srec, vec3 albedo) const {
      //srec.is_specular = false;
      srec[static_cast<vtkm::Id>(SR::Ax)] = albedo[0];
      srec[static_cast<vtkm::Id>(SR::Ay)] = albedo[1];
      srec[static_cast<vtkm::Id>(SR::Az)] = albedo[2];
      return true;
  }
  template<typename HitRecord>
  VTKM_EXEC
  vec3 emit(const vec3 &origin, const vec3 &direction, const HitRecord& rec, vec3 emit) const { return vec3(0,0,0); }

  using ControlSignature = void(FieldInOut<>, FieldInOut<>, FieldInOut<>,FieldInOut<>, FieldInOut<>, FieldInOut<>, WholeArrayInOut<>,WholeArrayInOut<>, WholeArrayInOut<>, WholeArrayInOut<>);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8,_9, _10);
  template<typename VecArrayType,
          typename ColorArrayType,
          typename MatTypeArray,
          typename TexTypeArray,
          typename HitRecord, typename HitId,
           typename ScatterRecord,
          int FinishedBitIdx = 1,
          int ScatterBitIdx = 3,
           int SpecularBitIdx = 4>
  VTKM_EXEC
  void operator()(vtkm::Id idx,
                  vec3 &origin,
                  vec3 &direction,
                  HitRecord &hrec,
                  HitId &hid,
                  ScatterRecord &srec,
                  vtkm::UInt8 &fin,
                  ColorArrayType col,
                  MatTypeArray matType,
                  TexTypeArray texType,
                  VecArrayType emitted) const
  {
    if (!(fin & (1UL << FinishedBitIdx))){ //fin
      if (fin & (1UL << ScatterBitIdx)){ //scattered
        auto mt = matType.Get(hid[static_cast<vtkm::Id>(HI::M)]);
        if (mt == 0){
          auto tt = texType.Get(hid[static_cast<vtkm::Id>(HI::T)]);
          vec3 em = emit(origin, direction, hrec, col.Get(tt));
          auto sctr = scatter(origin, direction, hrec, srec, col.Get(tt));
          fin |= (sctr << ScatterBitIdx); //scattered
          fin &= ~(1UL << SpecularBitIdx);
          emitted.Set(canvasSize * depth + idx, em);

        }
      }
    }

  }

  const vtkm::Id depth, canvasSize;
};

class DiffuseLightWorklet : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  DiffuseLightWorklet(vtkm::Id cs, vtkm::Id d)
    : canvasSize(cs)
    , depth(d)
  {
  }

  template<typename HitRecord, typename ScatterRecord>
  VTKM_EXEC
  bool scatter(const vec3 &origin, const vec3 &direction, const HitRecord& hrec, ScatterRecord &srec, vtkm::UInt8 &fin, vec3) const {
        return false;}
  template<typename HitRecord>
  VTKM_EXEC
  vec3 emit(const vec3 &origin, const vec3 &direction, const HitRecord& rec, vec3 emit) const {
    vec3 n(rec[static_cast<vtkm::Id>(HR::Nx)], rec[static_cast<vtkm::Id>(HR::Ny)],rec[static_cast<vtkm::Id>(HR::Nz)]);
      if (dot(n, direction) < 0.0)
          return emit;
      else
          return vec3(0,0,0);
  }

  using ControlSignature = void(FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, WholeArrayInOut<>, WholeArrayInOut<>, WholeArrayInOut<>, WholeArrayInOut<>);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10);
  template<typename VecArrayType,
          typename ColorArrayType,
          typename MatTypeArray,
          typename TexTypeArray,
  typename HitRecord, typename HitId, typename ScatterRecord,
  int FinBitIdx = 1,
  int ScatterBitIdx= 3>
  VTKM_EXEC
  void operator()(vtkm::Id idx,
                  vec3 &origin,
                  vec3 &direction,
                  HitRecord &hrec,
                  HitId &hid,
                  ScatterRecord &srec,
                  vtkm::UInt8 &fin,
                  ColorArrayType col,
                  MatTypeArray matType,
                  TexTypeArray texType,
                  VecArrayType emitted) const
  {
    if(!(fin & (1UL << FinBitIdx))){
      if ( (fin & (1UL << ScatterBitIdx))){ //scattered
        auto mt = matType.Get(hid[static_cast<vtkm::Id>(HI::M)]);
        if (mt == 1){
          auto tt = texType.Get(hid[static_cast<vtkm::Id>(HI::T)]);
          vec3 em = emit(origin, direction, hrec, col.Get(tt));
          fin &= (scatter(origin, direction, hrec, srec, fin, col.Get(tt)) << ScatterBitIdx); //scattered
          emitted.Set(canvasSize * depth + idx, em);
        }
      }
    }
  }

  const vtkm::Id depth, canvasSize;

};
class DielectricWorklet : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  DielectricWorklet(vtkm::Id cs, vtkm::Id d, float rid, vtkm::UInt32 rc)
      : canvasSize(cs)
      , depth(d)
      , ref_idx(rid)
      , RayCount(rc)
  {
  }

  VTKM_EXEC
  float schlick(float cosine, float ref_idx) const {
      float r0 = (1-ref_idx) / (1+ref_idx);
      r0 = r0*r0;
      return r0 + (1-r0)*pow((1 - cosine),5);
  }

  VTKM_EXEC
  bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) const {
      vec3 uv = unit_vector(v);
      float dt = dot(uv, n);
      float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
      if (discriminant > 0) {
          refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
          return true;
      }
      else
          return false;
  }

  VTKM_EXEC
  vec3 reflect(const vec3& v, const vec3& n) const {
       return v - 2*dot(v,n)*n;
  }

  template<typename HitRecord,
           typename ScatterRecord>
  VTKM_EXEC
  bool scatter(const vec3 &origin, const vec3 &direction, const HitRecord& hrec, ScatterRecord& srec, vec3 albedo, double _rand) const {
      //srec.attenuation = vec3(1.0, 1.0, 1.0);
    srec[static_cast<vtkm::Id>(SR::Ax)] = 1;
    srec[static_cast<vtkm::Id>(SR::Ay)] = 1;
    srec[static_cast<vtkm::Id>(SR::Az)] = 1;
      vec3 outward_normal;
      vec3 n(hrec[static_cast<vtkm::Id>(HR::Nx)], hrec[static_cast<vtkm::Id>(HR::Ny)],hrec[static_cast<vtkm::Id>(HR::Nz)]);

       vec3 reflected = reflect(direction, n);
       vec3 refracted;
       float ni_over_nt;
       float reflect_prob;
       float cosine;
       if (dot(direction, n) > 0) {
            outward_normal = -n;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(direction, n) * vtkm::RMagnitude(direction);
       }
       else {
            outward_normal = n;
            ni_over_nt = 1.0 / ref_idx;
            cosine = -dot(direction, n) * vtkm::RMagnitude(direction);
       }
       if (refract(direction, outward_normal, ni_over_nt, refracted)) {
          reflect_prob = schlick(cosine, ref_idx);
       }
       else {
          reflect_prob = 1.0;
       }
       if (_rand < reflect_prob) {
          srec[static_cast<vtkm::Id>(SR::Ox)] = hrec[static_cast<vtkm::Id>(HR::Px)];
          srec[static_cast<vtkm::Id>(SR::Oy)] = hrec[static_cast<vtkm::Id>(HR::Py)];
          srec[static_cast<vtkm::Id>(SR::Oz)] = hrec[static_cast<vtkm::Id>(HR::Pz)];
          srec[static_cast<vtkm::Id>(SR::Dx)] = reflected[0];
          srec[static_cast<vtkm::Id>(SR::Dy)] = reflected[1];
          srec[static_cast<vtkm::Id>(SR::Dz)] = reflected[2];
       }
       else {
         srec[static_cast<vtkm::Id>(SR::Ox)] = hrec[static_cast<vtkm::Id>(HR::Px)];
         srec[static_cast<vtkm::Id>(SR::Oy)] = hrec[static_cast<vtkm::Id>(HR::Py)];
         srec[static_cast<vtkm::Id>(SR::Oz)] = hrec[static_cast<vtkm::Id>(HR::Pz)];
         srec[static_cast<vtkm::Id>(SR::Dx)] = refracted[0];
         srec[static_cast<vtkm::Id>(SR::Dy)] = refracted[1];
         srec[static_cast<vtkm::Id>(SR::Dz)] = refracted[2];
       }
       return true;
  }
  template<typename HitRecord>
  VTKM_EXEC
  vec3 emit(const vec3 &origin, const vec3 &direction, const HitRecord& rec, vec3 emit) const { return vec3(0,0,0); }

  using ControlSignature = void(FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, FieldInOut<>, WholeArrayInOut<>, WholeArrayInOut<>, WholeArrayInOut<>, WholeArrayInOut<>);
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);
  template<typename VecArrayType,
          typename ColorArrayType,
          typename MatTypeArray,
          typename TexTypeArray,
          typename HitRecord,
          typename HitId,
           typename ScatterRecord,
          int FinishedBitIdx = 1,
          int ScatterBitIdx = 3,
           int SpecularBitIdx = 4>
  VTKM_EXEC
  void operator()(vtkm::Id idx,
                  unsigned int &seed,
                  vec3 &origin,
                  vec3 &direction,
                  HitRecord &hrec,
                  HitId &hid,
                  ScatterRecord &srec,
                  vtkm::UInt8 &fin,
                  ColorArrayType col,
                  MatTypeArray matType,
                  TexTypeArray texType,
                  VecArrayType emitted) const
  {
    if (!(fin & (1UL << FinishedBitIdx))){
      if ((fin & (1UL << ScatterBitIdx))){ //scattered

        auto mt = matType.Get(hid[static_cast<vtkm::Id>(HI::M)]);
        if (mt == 2){
          auto tt = texType.Get(hid[static_cast<vtkm::Id>(HI::T)]);
          vec3 em = emit(origin, direction, hrec, col.Get(tt));
          auto sctr = scatter(origin, direction, hrec, srec, col.Get(tt), xorshiftWang::getRandF(seed));
          fin |= (sctr << ScatterBitIdx);//vtkm::random::xorshift::getRandF(randState));
          fin |= (1UL << SpecularBitIdx);
          emitted.Set(canvasSize * depth + idx, em);

        }
      }
    }
  }

  const vtkm::Id depth, canvasSize;
  const float ref_idx;
  const vtkm::UInt32 RayCount;

};

#endif
