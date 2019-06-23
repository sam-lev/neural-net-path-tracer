//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
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
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//
//=============================================================================

#ifndef xorShiftWang_h
#define xorShiftWang_h

#include <vtkm/Types.h>

namespace xorshiftWang
{
VTKM_EXEC_CONT inline vtkm::UInt32 getWang32(vtkm::UInt32 &seed)
{ //wangshift
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}


///* The state array must be initialized to not be all zero in the first four words */
//uint32_t xorwow(vtkm::Vec<vtkm::UInt32, 5> state)
//{
//  /* Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" */
//  vtkm::UInt32 s, t = state[3];
//  state[3] = state[2];
//  state[2] = state[1];
//  state[1] = s = state[0];
//  t ^= t >> 2;
//  t ^= t << 1;
//  state[0] = t ^= s ^ (s << 4);
//  return t + (state[4] += 362437);

//}
VTKM_EXEC_CONT inline vtkm::Float32 getRandF(vtkm::UInt32 &seed)
{ //xorShift128. modifies randState!
  vtkm::UInt32 t = getWang32(seed);
  return vtkm::Float32(t) / 4294967295.f;
}

}

#endif //vtk_m_xorShift_h
