#include "CornellBox.h"

#include <vtkm/Transform3D.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include "vtkm/cont/DataSetBuilderExplicit.h"
#include "pathtracing/SphereExtractor.h"
#include "pathtracing/QuadExtractor.h"

void CornellBox::invert(vtkm::Vec<vec3,4> &pts)
{
  using vec4 = vtkm::Vec<vtkm::Float32, 4>;
  auto vec3ToVec4 = [&](vec3 in) mutable ->vec4{vec4 ret; ret[0] = in[0]; ret[1] = in[1]; ret[2] = in[2]; ret[3] = 1.f; return ret;};
  auto vec4ToVec3 = [&](vec4 in) mutable ->vec3{vec3 ret; ret[0] = in[0]; ret[1] = in[1]; ret[2] = in[2]; return ret;};

  vec3 offset(265,0,295);
  vtkm::Float32 angle = -15;
  auto translationMatrix = vtkm::Transform3DTranslate(offset[0], offset[1], offset[2]);
  auto rotationMatrix = vtkm::Transform3DRotate(angle, 0.f,1.f,0.f);
  rotationMatrix = vtkm::MatrixTranspose(rotationMatrix);
  auto mat = MatrixMultiply(translationMatrix,rotationMatrix);


  auto pt = vec3ToVec4(pts[0]);
  pts[0] = vec4ToVec3(MatrixMultiply(mat, pt));

  pt = vec3ToVec4(pts[1]);
  pts[1] = vec4ToVec3(MatrixMultiply(mat, pt));

  pt = vec3ToVec4(pts[2]);
  pts[2] = vec4ToVec3(MatrixMultiply(mat, pt));

  pt = vec3ToVec4(pts[3]);
  pts[3] = vec4ToVec3(MatrixMultiply(mat, pt));
}
vtkm::cont::DataSet CornellBox::buildDataSet()
{
  tex.Allocate(4);
  tex.GetPortalControl().Set(0, vec3(0.65, 0.05, 0.05));
  tex.GetPortalControl().Set(1, vec3(0.73, 0.73, 0.73));
  tex.GetPortalControl().Set(2, vec3(0.12, 0.45, 0.15));
  tex.GetPortalControl().Set(3, vec3(15, 15, 15));

  matType.Allocate(5);
  matType.GetPortalControl().Set(0, 0); //lambertian
  matType.GetPortalControl().Set(1, 0); //lambertian
  matType.GetPortalControl().Set(2, 0); //lambertian
  matType.GetPortalControl().Set(3, 1); //light
  matType.GetPortalControl().Set(4, 2); //dielectric

  texType.Allocate(5);
  texType.GetPortalControl().Set(0, 0); //red
  texType.GetPortalControl().Set(1, 1); //white
  texType.GetPortalControl().Set(2, 2); //green
  texType.GetPortalControl().Set(3, 3); //super bright
  texType.GetPortalControl().Set(4, 0); //dielectric


  vtkm::cont::ArrayHandle<vtkm::UInt8> shapes;
  vtkm::cont::ArrayHandle<vtkm::IdComponent> numindices;
  vtkm::cont::ArrayHandle<vtkm::Id> conn;
  vtkm::cont::ArrayHandle<vec3> pts1;

  field.Allocate(12*4+1);
  pts1.Allocate(12 * 4 + 1);
  numindices.Allocate(13);
  shapes.Allocate(13);
  conn.Allocate(12*4+1);
  QuadIds.Allocate(13);
  vtkm::cont::ArrayCopy(vtkm::cont::ArrayHandleCounting<vtkm::Id>(0,1, 12*4 + 1), conn);


  matIdx[0].Allocate(12);
  texIdx[0].Allocate(12);
  matIdx[1].Allocate(1);
  texIdx[1].Allocate(1);
  int cell_cnt = 0;
  int pt_idx = 0;
  auto close = [&](){  pt_idx += 4; cell_cnt++; };

  //yz_rect //green
  matIdx[0].GetPortalControl().Set(cell_cnt, 2);
  texIdx[0].GetPortalControl().Set(cell_cnt, 2);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, vec3(555,0,0));
  pts1.GetPortalControl().Set(pt_idx+1, vec3(555,555,0));
  pts1.GetPortalControl().Set(pt_idx+2, vec3(555,555,555));
  pts1.GetPortalControl().Set(pt_idx+3, vec3(555,0,555));
  field.GetPortalControl().Set(pt_idx, 0);
  field.GetPortalControl().Set(pt_idx+1, 0);
  field.GetPortalControl().Set(pt_idx+2, 0);
  field.GetPortalControl().Set(pt_idx+3, 0);
  close();

  matIdx[0].GetPortalControl().Set(cell_cnt, 0); //red
  texIdx[0].GetPortalControl().Set(cell_cnt, 0);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, vec3(0,0,0));
  pts1.GetPortalControl().Set(pt_idx+1, vec3(0,555,0));
  pts1.GetPortalControl().Set(pt_idx+2, vec3(0,555,555));
  pts1.GetPortalControl().Set(pt_idx+3, vec3(0,0,555));
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  //xz_rect
  matIdx[0].GetPortalControl().Set(cell_cnt, 3);
  texIdx[0].GetPortalControl().Set(cell_cnt, 3); //light
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, vec3(213,554,227));
  pts1.GetPortalControl().Set(pt_idx+1, vec3(343,554,227));
  pts1.GetPortalControl().Set(pt_idx+2, vec3(343,554,332));
  pts1.GetPortalControl().Set(pt_idx+3, vec3(213,554,332));
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  matIdx[0].GetPortalControl().Set(cell_cnt, 1); //white
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, vec3(0,555,0));
  pts1.GetPortalControl().Set(pt_idx+1, vec3(555,555,0));
  pts1.GetPortalControl().Set(pt_idx+2, vec3(555,555,555));
  pts1.GetPortalControl().Set(pt_idx+3, vec3(0,555,555));
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  matIdx[0].GetPortalControl().Set(cell_cnt, 1); //white
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, vec3(0,0,0));
  pts1.GetPortalControl().Set(pt_idx+1, vec3(555,0,0));
  pts1.GetPortalControl().Set(pt_idx+2, vec3(555,0,555));
  pts1.GetPortalControl().Set(pt_idx+3, vec3(0,0,555));
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  //xy_rect
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1); //white
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, vec3(0,0,555));
  pts1.GetPortalControl().Set(pt_idx+1, vec3(555,0,555));
  pts1.GetPortalControl().Set(pt_idx+2, vec3(555,555,555));
  pts1.GetPortalControl().Set(pt_idx+3, vec3(0,555,555));
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();


//  //small box
  //xy
  vtkm::Vec<vec3,4> pts;
  pts[0] = vec3(0,0,165);
  pts[1] = vec3(165,0,165);
  pts[2] = vec3(165,330,165);
  pts[3] = vec3(0,330,165);
  invert(pts);
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, pts[0]);
  pts1.GetPortalControl().Set(pt_idx+1, pts[1]);
  pts1.GetPortalControl().Set(pt_idx+2, pts[2]);
  pts1.GetPortalControl().Set(pt_idx+3, pts[3]);
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  pts[0] = vec3(0,0,0);
  pts[1] = vec3(165,0,0);
  pts[2] = vec3(165,330,0);
  pts[3] = vec3(0,330,0);
  invert(pts);
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, pts[0]);
  pts1.GetPortalControl().Set(pt_idx+1, pts[1]);
  pts1.GetPortalControl().Set(pt_idx+2, pts[2]);
  pts1.GetPortalControl().Set(pt_idx+3, pts[3]);
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  //yz
  pts[0] = vec3(165,0,0);
  pts[1] = vec3(165,330,0);
  pts[2] = vec3(165,330,165);
  pts[3] = vec3(165, 0, 165);
  invert(pts);
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, pts[0]);
  pts1.GetPortalControl().Set(pt_idx+1, pts[1]);
  pts1.GetPortalControl().Set(pt_idx+2, pts[2]);
  pts1.GetPortalControl().Set(pt_idx+3, pts[3]);
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  pts[0] = vec3(0,0,0);
  pts[1] = vec3(0,330,0);
  pts[2] = vec3(0,330,165);
  pts[3] = vec3(0, 0, 165);
  invert(pts);
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, pts[0]);
  pts1.GetPortalControl().Set(pt_idx+1, pts[1]);
  pts1.GetPortalControl().Set(pt_idx+2, pts[2]);
  pts1.GetPortalControl().Set(pt_idx+3, pts[3]);
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();


  //xz_rect
  pts[0] = vec3(0,333,0);
  pts[1] = vec3(165,330,0);
  pts[2] = vec3(165,330,165);
  pts[3] = vec3(0, 330, 165);
  invert(pts);
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, pts[0]);
  pts1.GetPortalControl().Set(pt_idx+1, pts[1]);
  pts1.GetPortalControl().Set(pt_idx+2, pts[2]);
  pts1.GetPortalControl().Set(pt_idx+3, pts[3]);
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  pts[0] = vec3(0,0,0);
  pts[1] = vec3(165,0,0);
  pts[2] = vec3(165,0,165);
  pts[3] = vec3(0, 0, 165);
  invert(pts);
  matIdx[0].GetPortalControl().Set(cell_cnt, 1);
  texIdx[0].GetPortalControl().Set(cell_cnt, 1);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_QUAD);
  numindices.GetPortalControl().Set(cell_cnt, 4);
  pts1.GetPortalControl().Set(pt_idx, pts[0]);
  pts1.GetPortalControl().Set(pt_idx+1, pts[1]);
  pts1.GetPortalControl().Set(pt_idx+2, pts[2]);
  pts1.GetPortalControl().Set(pt_idx+3, pts[3]);
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+1, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+2, float(cell_cnt)/13.0);
  field.GetPortalControl().Set(pt_idx+3, float(cell_cnt)/13.0);
  close();

  //    //sphere

  matIdx[1].GetPortalControl().Set(0, 4);
  texIdx[1].GetPortalControl().Set(0, 0);
  shapes.GetPortalControl().Set(cell_cnt, vtkm::CELL_SHAPE_VERTEX);
  numindices.GetPortalControl().Set(cell_cnt, 1);
  pts1.GetPortalControl().Set(pt_idx, vec3(190,90,190));
  field.GetPortalControl().Set(pt_idx, float(cell_cnt)/13.0);
  cell_cnt++;
  pt_idx++;

  coord.SetData( pts1);

  vtkm::cont::DataSetBuilderExplicit dsb;
  auto arr = coord.GetData().Cast<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3>>>();
  ds = dsb.Create(arr,shapes,numindices,conn, "coords", "cells");


  vtkm::cont::Field pfield(
    "point_var",
    vtkm::cont::Field::Association::POINTS,
    field);

  ds.AddField(pfield);
  return ds;
}

void CornellBox::extract()
{

  vtkm::rendering::raytracing::SphereExtractor sphereExtractor;
  sphereExtractor.ExtractCells(ds.GetCellSet(0), 90);
  SphereIds = sphereExtractor.GetPointIds();
  SphereRadii = sphereExtractor.GetRadii();
  ShapeOffset = ds.GetCellSet(0).Cast<vtkm::cont::CellSetExplicit<>>().GetIndexOffsetArray(vtkm::TopologyElementTagPoint(), vtkm::TopologyElementTagCell());
  for (int i=0; i<SphereIds.GetNumberOfValues(); i++){
    std::cout << SphereIds.GetPortalConstControl().Get(i) << std::endl;
    std::cout << SphereRadii.GetPortalConstControl().Get(i) << std::endl;

  }
  vtkm::rendering::raytracing::QuadExtractor quadExtractor;
  quadExtractor.ExtractCells(ds.GetCellSet(0));
  QuadIds = quadExtractor.GetQuadIds();

}
