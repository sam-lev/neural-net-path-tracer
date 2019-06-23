#ifndef PDFWORKLET_H
#define PDFWORKLET_H
#include <float.h>
#include "onb.h"
#include "Surface.h"
#include "Record.h"
#include "wangXor.h"
class WorketletGenerateDir : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_EXEC_CONT
  WorketletGenerateDir(int ts) : type_size(ts){}

  using ControlSignature = void(FieldInOut<>, FieldInOut<>);
  using ExecutionSignature = void( _1, _2);

  VTKM_EXEC
  void operator()(unsigned int &seed, int &which) const {
    which = vtkm::Min(type_size,int(xorshiftWang::getRandF(seed) * type_size+1));
  }
  const int type_size;
};

class CosineWorketletGenerateDir : public vtkm::worklet::WorkletMapField
{
public:
  CosineWorketletGenerateDir(int cur = 0) : current(cur){}

  VTKM_EXEC
  vec3 random(const vec3& o, float r1, float r2,
              float x0, float x1, float z0, float z1, float k) const {
      vec3 random_point = vec3(x0 + r1*(x1-x0), k,  z0 + r2*(z1-z0));
      return random_point - o;
  }

  VTKM_EXEC
  inline vec3 de_nan(const vec3& c) const {
      vec3 temp = c;
      if (!(temp[0] == temp[0])) temp[0] = 0;
      if (!(temp[1] == temp[1])) temp[1] = 0;
      if (!(temp[2] == temp[2])) temp[2] = 0;
      return temp;
  }


  VTKM_EXEC
  inline vec3 random_cosine_direction(float r1, float r2) const {
      float z = sqrt(1-r2);
      float phi = 2*M_PI*r1;
      float x = cos(phi)*2*sqrt(r2);
      float y = sin(phi)*2*sqrt(r2);
      if(x!=x) x=0;
      if(y!=y) y=0;
      if(z!=z) z=0;
      return vec3(x, y, z);
  }
  using ControlSignature = void(FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>
  );
  using ExecutionSignature = void(_1, _2, _3, _4);

  template<typename HitRecord>
  VTKM_EXEC
  void operator()(
      int which,
       HitRecord &hrec,
       vec3 &generated,
      unsigned int &seed
    ) const
  {
    if (which <= current ){
      float r1 =xorshiftWang::getRandF(seed);
      float r2 =xorshiftWang::getRandF(seed);
      onb uvw;

      vec3 hrecn(hrec[static_cast<vtkm::Id>(HR::Nx)], hrec[static_cast<vtkm::Id>(HR::Ny)], hrec[static_cast<vtkm::Id>(HR::Nz)]);
      uvw.build_from_w(hrecn);
      generated = de_nan(uvw.local(random_cosine_direction(r1,r2)));
    }
  }
  int current;
};

class QuadWorkletGenerateDir : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_EXEC_CONT
  QuadWorkletGenerateDir(int cur = 1): current(cur) {}
  VTKM_EXEC
  vec3 random(const vec3& o, unsigned int &seed,
              float x0, float x1, float y0, float y1, float z0, float z1) const {
    auto r1 = xorshiftWang::getRandF(seed);
    auto r2 = xorshiftWang::getRandF(seed);
    auto r3 = xorshiftWang::getRandF(seed);
      vec3 random_point = vec3(x0 + r1*(x1-x0), y0 + r2*(y1 - y0),  z0 + r3*(z1-z0));
      return random_point - o;
  }

  using ControlSignature = void(FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  WholeArrayIn<>,
  WholeArrayIn<>,
  WholeArrayIn<>

  );
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7);

  template<typename PtArrayType, typename HitRecord,
           typename PointIdType, typename QuadIndicesType>
  VTKM_EXEC
  void operator()(int which,
                  HitRecord &hrec,
                  vec3 &generated,
                  unsigned int &seed,
                  const PointIdType &PointIds,
                  const QuadIndicesType &quadIndices,
                  const PtArrayType pts
                  ) const
  {
    if (current == which){
      for (int quadIndex=0; quadIndex<quadIndices.GetNumberOfValues(); quadIndex++){
        auto pointIndex = PointIds.Get(quadIndex);

        auto pt1 = pts.Get(pointIndex[1]);
        auto pt2 = pts.Get(pointIndex[3]);
        float x0 = pt1[0];
        float x1 = pt2[0];
        float z0 = pt1[2];
        float z1 = pt2[2];
        float y0 = pt1[1];
        float y1 = pt1[1];
        vec3 p(hrec[static_cast<vtkm::Id>(HR::Px)], hrec[static_cast<vtkm::Id>(HR::Py)], hrec[static_cast<vtkm::Id>(HR::Pz)]);
        generated = random(p,seed, x0,x1,y0,y1,z0,z1);
      }
    }
  }

  int current;
};
class SphereWorkletGenerateDir : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_EXEC_CONT
  SphereWorkletGenerateDir(int cur = 2): current(cur) {}

  VTKM_EXEC
  inline vec3 de_nan(const vec3& c) const {
      vec3 temp = c;
      if (temp[0] != temp[0]) temp[0] = 0;
      if (temp[1] != temp[1]) temp[1] = 0;
      if (temp[2] != temp[2]) temp[2] = 0;
      return temp;
  }


  VTKM_EXEC
  vec3 random_to_sphere(float radius, float distance_squared, float r1, float r2) const {
  //    float r1 = drand48();
  //    float r2 = drand48();
      float z = 1 + r2*(sqrt(1-radius*radius/distance_squared) - 1);
      float phi = 2*M_PI*r1;
      float x = cos(phi)*sqrt(1-z*z);
      float y = sin(phi)*sqrt(1-z*z);
      if(z!=z) z=0;
      if(x!=x) x=0;
      if(y!=y) y=0;
      return vec3(x, y, z);
  }
  VTKM_EXEC
  vec3 random(const vec3& o, float r1, float r2,
              const vec3 &center, float radius) const {
       vec3 direction = center - o;
       float distance_squared = MagnitudeSquared(direction);
       onb uvw;
       uvw.build_from_w(direction);
       return de_nan(uvw.local(random_to_sphere(radius, distance_squared, r1,r2)));
  }
  using ControlSignature = void(FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  WholeArrayIn<>,
  WholeArrayIn<>,
  WholeArrayIn<>,
  WholeArrayIn<>

  );
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8);

  template<typename PtArrayType,
           typename HitRecord,
           typename PointIdType,
           typename IndexArrayType,
           typename RadiiArrayType>
  VTKM_EXEC
  void operator()(vtkm::Id idx,
          int &which,
           HitRecord &hrec,
           vec3 &generated,
                  unsigned int &seed,
                  const PointIdType &PointIds,
                  const IndexArrayType &idxArray,
                  const PtArrayType &pt1,
                  const RadiiArrayType &radii
           ) const
  {
    if (which == current){
      for (int i = 0; i < idxArray.GetNumberOfValues(); i++){
        auto radius = radii.Get(i);
        auto pointIndex = PointIds.Get(i);

        vec3 p(hrec[static_cast<vtkm::Id>(HR::Px)], hrec[static_cast<vtkm::Id>(HR::Py)], hrec[static_cast<vtkm::Id>(HR::Pz)]);
        generated = random(p, xorshiftWang::getRandF(seed), xorshiftWang::getRandF(seed), pt1.Get(pointIndex), radius);
      }
    }
  }
  int current;
};

class QuadPDFWorklet : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  QuadPDFWorklet(int ls
      )
    :list_size(ls)
  {
  }


  template<typename ExecSurf>
  VTKM_EXEC
  float  pdf_value(const vec3& o, const vec3& v,
                   const vec3 &q, const vec3 &r, const vec3 &s, const vec3 &t,
                   ExecSurf &surf) const {
    vtkm::Vec<vtkm::Float32, 9> rec;
    vtkm::Vec<vtkm::Int8,2> hid;
    if (surf.intersect(o,v, rec, hid, 0.001, FLT_MAX, q,r,s,t, 0)) {
      auto qr = vtkm::Magnitude(r - q);
      auto qt = vtkm::Magnitude(t - q);

      float area = qr * qt;
      auto rect = rec[static_cast<vtkm::Id>(HR::T)];
      float distance_squared = rect * rect * vtkm::MagnitudeSquared(v);
      vec3 n(rec[static_cast<vtkm::Id>(HR::Nx)],rec[static_cast<vtkm::Id>(HR::Ny)],rec[static_cast<vtkm::Id>(HR::Nz)]);
      float cosine = fabs(dot(v, n) * vtkm::RMagnitude(v));
      if(cosine != cosine) cosine = 0;
      return  distance_squared / (cosine * area);
    }
    else
        return 0;
  }


  using ControlSignature = void(FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  ExecObject surf,
  WholeArrayIn<>,
  WholeArrayIn<>,
  WholeArrayIn<>

  );
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11);

  template<typename PtArrayType,
           typename PointIdType,
           typename QuadIndicesType,
          typename FlippedType,
          typename HitRecord,
           typename ExecSurf,
  int ScatterBitIndex = 3>
  VTKM_EXEC
  void operator()(vtkm::Id idx,
                  vec3 &origin,
                  vec3 &direction,
                  HitRecord &hrec,
                  FlippedType &scattered,
                  float &sum_value,
                  vec3 &generated,
                  unsigned int &seed,
                  ExecSurf surf,
                  const PointIdType &PointIds,
                  const QuadIndicesType &quadIndices,
                  const PtArrayType &pts
                  ) const
  {
    if (scattered & (1UL << ScatterBitIndex)){
      float weight = 1.0/list_size;
      //int index = int(xorshiftWang::getRandF(seed) * list_size);
      for (int quadIndex=0; quadIndex<quadIndices.GetNumberOfValues(); quadIndex++){
        auto pointIndex = PointIds.Get(quadIndex);

        vec3 q, r, s, t;
        q = pts.Get(pointIndex[1]);
        r = pts.Get(pointIndex[2]);
        s = pts.Get(pointIndex[3]);
        t = pts.Get(pointIndex[4]);
        vec3 p(hrec[static_cast<vtkm::Id>(HR::Px)], hrec[static_cast<vtkm::Id>(HR::Py)], hrec[static_cast<vtkm::Id>(HR::Pz)]);
        sum_value += weight*pdf_value(p, generated, q,r,s,t, surf);

      }

//      for (int i = 0; i < pt1.GetNumberOfValues(); i++){
//        float x0 = pt1.Get(i)[0];
//        float x1 = pt2.Get(i)[0];
//        float z0 = pt1.Get(i)[2];
//        float z1 = pt2.Get(i)[2];
//        float k = pt1.Get(i)[1];
//        vec3 p(hrec[static_cast<vtkm::Id>(HR::Px)], hrec[static_cast<vtkm::Id>(HR::Py)], hrec[static_cast<vtkm::Id>(HR::Pz)]);
//        sum_value += weight*pdf_value(p, generated, x0,x1,z0,z1,k, surf);
//        //if (!index)

//      }
    }
  }

  float list_size;
};

class SpherePDFWorklet : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  SpherePDFWorklet(int ls
      )
    : list_size(ls)
  {
  }

  template<typename SurfExec>
  VTKM_EXEC
  float  pdf_value(const vec3& o, const vec3& v,
                  vec3 center, float radius,
                   SurfExec surf) const {

    vtkm::Vec<vtkm::Float32, 9> rec, hid;
    int matId,texId;
    if (surf.hit(o, v, rec, hid, 0.001, FLT_MAX, center, radius, matId, texId )) {
        float cos_theta_max = sqrt(1 - radius*radius/vtkm::MagnitudeSquared(center-o));
        if(cos_theta_max != cos_theta_max) cos_theta_max = 0;
        float solid_angle = 2*M_PI*(1-cos_theta_max);
        return  1 / solid_angle;
    }
    else
        return 0;

  }


  using ControlSignature = void(FieldIn<>,
  FieldIn<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  FieldInOut<>,
  ExecObject surf,
  WholeArrayIn<>,
  WholeArrayIn<>,
  WholeArrayIn<>,
  WholeArrayIn<>
  );
  using ExecutionSignature = void(WorkIndex, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12);

  template<typename PtArrayType,
           typename FlippedType,
           typename HitRecord,
           typename SphereExec,
           typename IndexArrayType,
           typename RadiiArrayType,
           typename PointIdType,
           int ScatterBitIndex = 3>
  VTKM_EXEC
  void operator()(vtkm::Id idx,
                  const vec3 &origin,
                  const vec3 &direction,
                  HitRecord &hrec,
                  FlippedType &scattered,
                  float &sum_value,
                  vec3 &generated,
                  unsigned int &seed,
                  SphereExec surf,
                  const PointIdType &PointIds,
                  const IndexArrayType &idxArray,
                  const PtArrayType &pt1,
                  const RadiiArrayType &radii
                  ) const
  {
    if (scattered & (1UL << ScatterBitIndex)){
      float weight = 1.0/list_size;
      int index = int(xorshiftWang::getRandF(seed) *list_size);
      for (int i = 0; i < idxArray.GetNumberOfValues(); i++){
        auto radius = radii.Get(i);
        auto pointIndex = PointIds.Get(i);
        vec3 p(hrec[static_cast<vtkm::Id>(HR::Px)], hrec[static_cast<vtkm::Id>(HR::Py)], hrec[static_cast<vtkm::Id>(HR::Pz)]);
        sum_value += weight*pdf_value(p, generated, pt1.Get(pointIndex), radius, surf);
      }
    }
  }

  float list_size;
};
#endif
