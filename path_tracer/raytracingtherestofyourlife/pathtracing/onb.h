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

#ifndef ONBH
#define ONBH
#include "vec3.h"

class onb
{
    public:
        VTKM_EXEC_CONT
        onb() {}
        VTKM_EXEC_CONT
        inline vec3 operator[](int i) const { return axis[i]; }
        VTKM_EXEC_CONT
        vec3 u() const       { return axis[0]; }
        VTKM_EXEC_CONT
        vec3 v() const       { return axis[1]; }
        VTKM_EXEC_CONT
        vec3 w() const       { return axis[2]; }
        VTKM_EXEC_CONT
        vec3 local(float a, float b, float c) const { return a*u() + b*v() + c*w(); }
        VTKM_EXEC_CONT
        vec3 local(const vec3& a) const { return a[0]*u() + a[1]*v() + a[2]*w(); }
        VTKM_EXEC_CONT
        void build_from_w(const vec3&n)
        {
          axis[2] = unit_vector(n);
          vec3 a;
          if (fabs(w()[0]) > 0.9)
              a = vec3(0, 1, 0);
          else
              a = vec3(1, 0, 0);
          axis[1] = unit_vector( cross( w(), a ) );
          axis[0] = cross(w(), v());

        }

        vec3 axis[3];
};


#endif
