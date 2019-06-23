//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef VEC3H
#define VEC3H

#include <vtkm/Types.h>
#include <vtkm/VectorAnalysis.h>

#include <math.h>
#include <stdlib.h>
#include <iostream>

using vec3 = vtkm::Vec<vtkm::Float32, 3>;

VTKM_EXEC_CONT
inline float dot(const vec3 &v1, const vec3 &v2) {
    //return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
  return vtkm::Dot(v1,v2);
}

VTKM_EXEC_CONT
inline vec3 cross(const vec3 &v1, const vec3 &v2) {
//    return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
//                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
//                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
  return vtkm::Cross(v1,v2);
}

VTKM_EXEC_CONT
inline vec3 unit_vector(vec3 v) {
    //return v / v.length();
  return v * vtkm::RMagnitude(v);
}

#endif
