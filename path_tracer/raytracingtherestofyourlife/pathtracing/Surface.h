#ifndef SURFACE_H
#define SURFACE_H
#include "AABBSurface.h"
#include "Record.h"
#include "vec3.h"


template <typename Device>
class QuadLeafIntersector
{
public:
  using Id5Handle = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>>;
  using IdHandle = vtkm::cont::ArrayHandle<vtkm::Id>;
  using FloatHandle = vtkm::cont::ArrayHandle<vtkm::Float32>;
  using IdArrayPortal = typename IdHandle::ExecutionTypes<Device>::PortalConst;
  using Id5ArrayPortal = typename Id5Handle::ExecutionTypes<Device>::PortalConst;
  Id5ArrayPortal PointIds;
  IdArrayPortal MatIdx, TexIdx;

  QuadLeafIntersector(){}
  QuadLeafIntersector(const Id5Handle& pointIds,
                        const IdHandle& matIdx,
                        const IdHandle& texIdx)
    : PointIds(pointIds.PrepareForInput(Device()))
    , MatIdx(matIdx.PrepareForInput(Device()))
    , TexIdx(texIdx.PrepareForInput(Device()))
  {

  }
  template <typename vec3, typename Precision>
  VTKM_EXEC bool hit(const vec3& ray_origin,
                      const vec3& ray_direction,
                      const vec3& v00,
                      const vec3& v10,
                      const vec3& v11,
                      const vec3& v01,
                      Precision& u,
                      Precision& v,
                      Precision& t) const
  {

    /* An Eﬃcient Ray-Quadrilateral Intersection Test
         Ares Lagae Philip Dutr´e
         http://graphics.cs.kuleuven.be/publications/LD05ERQIT/index.html

      v01 *------------ * v11
          |\           |
          |  \         |
          |    \       |
          |      \     |
          |        \   |
          |          \ |
      v00 *------------* v10
      */
    // Rejects rays that are parallel to Q, and rays that intersect the plane of
    // Q either on the left of the line V00V01 or on the right of the line V00V10.

    vec3 E03 = v01 - v00;
    vec3 P = vtkm::Cross(ray_direction, E03);
    vec3 E01 = v10 - v00;
    Precision det = vtkm::dot(E01, P);

    if (vtkm::Abs(det) < vtkm::Epsilon<Precision>())
      return false;
    Precision inv_det = 1.0f / det;
    vec3 T = ray_origin - v00;
    Precision alpha = vtkm::dot(T, P) * inv_det;
    if (alpha < 0.0)
      return false;
    vec3 Q = vtkm::Cross(T, E01);
    Precision beta = vtkm::dot(ray_direction, Q) * inv_det;
    if (beta < 0.0)
      return false;

    if ((alpha + beta) > 1.0f)
    {

      // Rejects rays that intersect the plane of Q either on the
      // left of the line V11V10 or on the right of the line V11V01.

      vec3 E23 = v01 - v11;
      vec3 E21 = v10 - v11;
      vec3 P_prime = vtkm::Cross(ray_direction, E21);
      Precision det_prime = vtkm::dot(E23, P_prime);
      if (vtkm::Abs(det_prime) < vtkm::Epsilon<Precision>())
        return false;
      Precision inv_det_prime = 1.0f / det_prime;
      vec3 T_prime = ray_origin - v11;
      Precision alpha_prime = vtkm::dot(T_prime, P_prime) * inv_det_prime;
      if (alpha_prime < 0.0f)
        return false;
      vec3 Q_prime = vtkm::Cross(T_prime, E23);
      Precision beta_prime = vtkm::dot(ray_direction, Q_prime) * inv_det_prime;
      if (beta_prime < 0.0f)
        return false;
    }

    // Compute the ray parameter of the intersection point, and
    // reject the ray if it does not hit Q.

    t = vtkm::dot(E03, Q) * inv_det;
    if (t < 0.0)
      return false;


    // Compute the barycentric coordinates of V11
    Precision alpha_11, beta_11;
    vec3 E02 = v11 - v00;
    vec3 n = vtkm::Cross(E01, E02);

    if ((vtkm::Abs(n[0]) >= vtkm::Abs(n[1])) && (vtkm::Abs(n[0]) >= vtkm::Abs(n[2])))
    {

      alpha_11 = ((E02[1] * E03[2]) - (E02[2] * E03[1])) / n[0];
      beta_11 = ((E01[1] * E02[2]) - (E01[2] * E02[1])) / n[0];
    }
    else if ((vtkm::Abs(n[1]) >= vtkm::Abs(n[0])) && (vtkm::Abs(n[1]) >= vtkm::Abs(n[2])))
    {

      alpha_11 = ((E02[2] * E03[0]) - (E02[0] * E03[2])) / n[1];
      beta_11 = ((E01[2] * E02[0]) - (E01[0] * E02[2])) / n[1];
    }
    else
    {

      alpha_11 = ((E02[0] * E03[1]) - (E02[1] * E03[0])) / n[2];
      beta_11 = ((E01[0] * E02[1]) - (E01[1] * E02[0])) / n[2];
    }

    // Compute the bilinear coordinates of the intersection point.
    if (vtkm::Abs(alpha_11 - 1.0f) < vtkm::Epsilon<Precision>())
    {

      u = alpha;
      if (vtkm::Abs(beta_11 - 1.0f) < vtkm::Epsilon<Precision>())
        v = beta;
      else
        v = beta / ((u * (beta_11 - 1.0f)) + 1.0f);
    }
    else if (vtkm::Abs(beta_11 - 1.0) < vtkm::Epsilon<Precision>())
    {

      v = beta;
      u = alpha / ((v * (alpha_11 - 1.0f)) + 1.0f);
    }
    else
    {

      Precision A = 1.0f - beta_11;
      Precision B = (alpha * (beta_11 - 1.0f)) - (beta * (alpha_11 - 1.0f)) - 1.0f;
      Precision C = alpha;
      Precision D = (B * B) - (4.0f * A * C);
      Precision QQ = -0.5f * (B + ((B < 0.0f ? -1.0f : 1.0f) * vtkm::Sqrt(D)));
      u = QQ / A;
      if ((u < 0.0f) || (u > 1.0f))
        u = C / QQ;
      v = beta / ((u * (beta_11 - 1.0f)) + 1.0f);
    }

    return true;
  }

  template<typename HitRecord, typename HitId>
  VTKM_EXEC bool intersect(const vec3& origin,
                           const vec3& direction,
                           HitRecord& temp_rec,
                           HitId &hid,
                           float tmin, float tmax,
                           const vec3 &q, const vec3 &r, const vec3 &s, const vec3 &t,
                           const vtkm::Id &index

                     ) const
  {
    auto h = hit(origin, direction, q,r,s,t,
                  temp_rec[static_cast<vtkm::Id>(HR::U)],
                  temp_rec[static_cast<vtkm::Id>(HR::V)],
                  temp_rec[static_cast<vtkm::Id>(HR::T)]);
    h = h && (temp_rec[static_cast<vtkm::Id>(HR::T)] < tmax) &&
        (temp_rec[static_cast<vtkm::Id>(HR::T)] > tmin);
    if (h){

      vec3 normal = vtkm::TriangleNormal(q,r,s);
      vtkm::Normalize(normal);
      if (vtkm::dot(normal, direction) > 0.f)
        normal = -normal;

      vec3 p(origin + direction * temp_rec[static_cast<vtkm::Id>(HR::T)]);
      temp_rec[static_cast<vtkm::Id>(HR::Px)] = p[0];
      temp_rec[static_cast<vtkm::Id>(HR::Py)] = p[1];
      temp_rec[static_cast<vtkm::Id>(HR::Pz)] = p[2];
      temp_rec[static_cast<vtkm::Id>(HR::Nx)] = normal[0];
      temp_rec[static_cast<vtkm::Id>(HR::Ny)] = normal[1];
      temp_rec[static_cast<vtkm::Id>(HR::Nz)] = normal[2];
      hid[static_cast<vtkm::Id>(HI::M)] = MatIdx.Get(index);
      hid[static_cast<vtkm::Id>(HI::T)] = TexIdx.Get(index);
    }

    return h;
  }

  template<typename Precision,
           typename PtArrayType,
           typename HitRecord,
           typename HitId,
           typename LeafPortalType,
           int HitBitIdx = 2,
           int ScatterBitIdx= 3>
  VTKM_EXEC void IntersectLeaf(
                  const vtkm::Int32 &currentNode,
                  const vtkm::Vec<Precision,3> &origin,
                  const vtkm::Vec<Precision,3> &direction,
                  HitRecord &hrec,
                  HitId &hid,
                  float &tmin,
                  float &tmax,
                  vtkm::UInt8 &scattered,
                  const PtArrayType &pts,
                  const LeafPortalType &leafs
      ) const

  {

    if (scattered & (1UL << ScatterBitIdx)){
      const vtkm::Id quadCount = leafs.Get(currentNode);
      for (vtkm::Id i = 1; i <= quadCount; ++i)
      {
        const vtkm::Id quadIndex = leafs.Get(currentNode + i);
        auto pointIndex = PointIds.Get(quadIndex);

        vec3 q, r, s, t;
        q = pts.Get(pointIndex[1]);
        r = pts.Get(pointIndex[2]);
        s = pts.Get(pointIndex[3]);
        t = pts.Get(pointIndex[4]);

        HitRecord temp_rec;

        auto h = intersect(origin,
                           direction,
                           temp_rec,
                           hid,
                           tmin,
                           tmax,
                           q,r,s,t,
                           quadIndex);
        if (h){
          hrec = temp_rec;
          tmax = temp_rec[static_cast<vtkm::Id>(HR::T)];
        }
        scattered |= (h << HitBitIdx);

      }
    }
  }

};
class QuadExecWrapper : public vtkm::cont::ExecutionObjectBase
{
  using IdType = vtkm::Vec<vtkm::Id, 5>;
public:
  vtkm::cont::ArrayHandle<IdType> &PointIds;
  vtkm::cont::ArrayHandle<vtkm::Id> &MatIdx;
  vtkm::cont::ArrayHandle<vtkm::Id> &TexIdx;

  QuadExecWrapper(vtkm::cont::ArrayHandle<IdType> &pointIds,
                    vtkm::cont::ArrayHandle<vtkm::Id> &matIdx,
                    vtkm::cont::ArrayHandle<vtkm::Id> &texIdx

                    )
    : PointIds(pointIds)
    , MatIdx(matIdx)
    , TexIdx(texIdx)

  {
  }

  template <typename Device>
  VTKM_CONT QuadLeafIntersector<Device> PrepareForExecution(Device) const
  {
    return QuadLeafIntersector<Device>(PointIds, MatIdx, TexIdx);
  }
};


template <typename Device>
class SphereLeafIntersector
{
public:
  using IdHandle = vtkm::cont::ArrayHandle<vtkm::Id>;
  using FloatHandle = vtkm::cont::ArrayHandle<vtkm::Float32>;
  using IdArrayPortal = typename IdHandle::ExecutionTypes<Device>::PortalConst;
  using FloatArrayPortal = typename FloatHandle::ExecutionTypes<Device>::PortalConst;
  IdArrayPortal PointIds;
  IdArrayPortal MatIdx, TexIdx;
  FloatArrayPortal Radii;

  SphereLeafIntersector(){}
  SphereLeafIntersector(const IdHandle& pointIds,
                        const FloatHandle &radii,
                        const IdHandle& matIdx,
                        const IdHandle& texIdx)
    : PointIds(pointIds.PrepareForInput(Device()))
    , Radii(radii.PrepareForInput(Device()))
    , MatIdx(matIdx.PrepareForInput(Device()))
    , TexIdx(texIdx.PrepareForInput(Device()))
  {

  }

  VTKM_EXEC
  void get_sphere_uv(const vec3& p, float& u, float& v) const {
      float phi = atan2(p[2], p[0]);
      if(phi!=phi) phi =0;
      float theta = asin(p[1]);
      if(theta!=theta) theta = 0;
      u = 1-(phi + M_PI) / (2*M_PI);
      v = (theta + M_PI/2) / M_PI;
  }

  template<typename HitRecord, typename HitId>
  VTKM_EXEC
  bool hit(const vec3& origin, const vec3 &direction,
           HitRecord& rec, HitId &hid, float tmin, float tmax,
           vec3 center, float radius,
           int matId, int texId) const {
      vec3 oc = origin - center;
      float a = dot(direction, direction);
      float b = dot(oc, direction);
      float c = dot(oc, oc) - radius*radius;
      float discriminant = b*b - a*c;
      if (discriminant > 0) {
          float temp = (-b - sqrt(b*b-a*c))/a;
          if(temp !=temp) temp=0;
          if (temp < tmax && temp > tmin) {
              rec[static_cast<vtkm::Id>(HR::T)] = temp;
              auto p = origin + direction * rec[static_cast<vtkm::Id>(HR::T)];
              rec[static_cast<vtkm::Id>(HR::Px)] = p[0];
              rec[static_cast<vtkm::Id>(HR::Py)] = p[1];
              rec[static_cast<vtkm::Id>(HR::Pz)] = p[2];
              get_sphere_uv((p-center)/radius, rec[static_cast<vtkm::Id>(HR::U)], rec[static_cast<vtkm::Id>(HR::V)]);
              auto n = (p - center) / radius;
              rec[static_cast<vtkm::Id>(HR::Nx)] = n[0];
              rec[static_cast<vtkm::Id>(HR::Ny)] = n[1];
              rec[static_cast<vtkm::Id>(HR::Nz)] = n[2];
              hid[static_cast<vtkm::Id>(HI::M)] = matId;
              hid[static_cast<vtkm::Id>(HI::T)] = texId;

              return true;
          }
          temp = (-b + sqrt(b*b-a*c))/a;
          if(temp !=temp) temp=0;
          if (temp < tmax && temp > tmin) {
              rec[static_cast<vtkm::Id>(HR::T)] = temp;
              auto p = origin + direction * (rec[static_cast<vtkm::Id>(HR::T)]);
              rec[static_cast<vtkm::Id>(HR::Px)] = p[0];
              rec[static_cast<vtkm::Id>(HR::Py)] = p[1];
              rec[static_cast<vtkm::Id>(HR::Pz)] = p[2];
              get_sphere_uv((p-center)/radius, rec[static_cast<vtkm::Id>(HR::U)], rec[static_cast<vtkm::Id>(HR::V)]);
              auto n = (p - center) / radius;
              rec[static_cast<vtkm::Id>(HR::Nx)] = n[0];
              rec[static_cast<vtkm::Id>(HR::Ny)] = n[1];
              rec[static_cast<vtkm::Id>(HR::Nz)] = n[2];
              hid[static_cast<vtkm::Id>(HI::M)] = matId;
              hid[static_cast<vtkm::Id>(HI::T)] = texId;

              return true;
          }
      }
      return false;
  }

  template<typename Precision,
           typename PtArrayType,
            typename HitRecord,
            typename HitId,
            typename LeafPortalType,
            int HitBitIdx = 2,
            int ScatterBitIdx= 3>
  VTKM_EXEC void IntersectLeaf(
                        const vtkm::Int32 &currentNode,
                        const vtkm::Vec<Precision,3> &origin,
                        const vtkm::Vec<Precision,3> &direction,
                        HitRecord &hrec,
                        HitId &hid,
                        float &tmin,
                        float &tmax,
                        vtkm::UInt8 &scattered,
                        const PtArrayType &pts,
                        const LeafPortalType &leafs) const
  {
    if (scattered & (1UL << ScatterBitIdx)){
      const vtkm::Id sphereCount = leafs.Get(currentNode);
      for (vtkm::Id i = 1; i <= sphereCount; ++i)
      {
        const vtkm::Id sphereIndex = leafs.Get(currentNode + i);
        auto radius = Radii.Get(sphereIndex);
        auto pointIndex = PointIds.Get(sphereIndex);
        vec3 pt = pts.Get(pointIndex);

        HitRecord  temp_rec;
        HitId temp_hid;
        auto h =   hit(origin, direction, temp_rec, temp_hid, tmin, tmax,
                       pt, radius,MatIdx.Get(sphereIndex),TexIdx.Get(sphereIndex));
        if (h){
          tmax = temp_rec[static_cast<vtkm::Id>(HR::T)];
          hrec = temp_rec;
          hid = temp_hid;
        }
        scattered |= (h << HitBitIdx);
      }
    }
  }
};


class SphereExecWrapper : public vtkm::cont::ExecutionObjectBase
{
public:
  vtkm::cont::ArrayHandle<vtkm::Id> PointIds;
  vtkm::cont::ArrayHandle<vtkm::Float32> Radii;
  vtkm::cont::ArrayHandle<vtkm::Id> MatIdx;
  vtkm::cont::ArrayHandle<vtkm::Id> TexIdx;

  SphereExecWrapper(vtkm::cont::ArrayHandle<vtkm::Id> &pointIds,
                    vtkm::cont::ArrayHandle<vtkm::Float32> radii,
                    vtkm::cont::ArrayHandle<vtkm::Id> &matIdx,
                    vtkm::cont::ArrayHandle<vtkm::Id> &texIdx

                    )
    : PointIds(pointIds)
    , Radii(radii)
    , MatIdx(matIdx)
    , TexIdx(texIdx)

  {
  }

  template <typename Device>
  VTKM_CONT SphereLeafIntersector<Device> PrepareForExecution(Device) const
  {
    return SphereLeafIntersector<Device>(PointIds, Radii, MatIdx, TexIdx);
  }
};


#endif
