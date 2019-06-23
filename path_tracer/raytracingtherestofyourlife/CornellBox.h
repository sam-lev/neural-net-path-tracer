#ifndef CORNELLBOX_H
#define CORNELLBOX_H
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/DataSet.h>
#include "pathtracing/vec3.h"
#include <vtkm/cont/CoordinateSystem.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>

class CornellBox
{
public:
  vtkm::cont::ArrayHandle<vec3> tex;
  vtkm::cont::ArrayHandle<vtkm::Id> matIdx[2];
  vtkm::cont::ArrayHandle<vtkm::Id> texIdx[2];
  vtkm::cont::ArrayHandle<int> matType, texType;
  vtkm::cont::CoordinateSystem coord;
  vtkm::cont::ArrayHandle<vtkm::Float32> field;

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Id,5>> QuadIds;
  vtkm::cont::ArrayHandle<vtkm::Id> SphereIds;

  vtkm::cont::ArrayHandle<vtkm::Float32> SphereRadii;
  vtkm::cont::ArrayHandle<vtkm::Id> ShapeOffset;

  void invert(vtkm::Vec<vec3,4> &pts);
  void extract();

  vtkm::cont::DataSet buildDataSet();

  vtkm::cont::DataSet ds;

  void saveDS(std::string fname){
    vtkm::io::writer::VTKDataSetWriter writer1(fname);
    writer1.WriteDataSet(ds);
  }
};

#endif
