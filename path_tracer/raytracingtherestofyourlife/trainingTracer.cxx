//==================================================================================================
// Written in 2019 by Mark Kim (and barely anything from a kid named sam)
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================


#include <iostream>
#include <limits>
#include <vector>
#include <tuple>
#include <sstream>
#include <vtkm/cont/ArrayHandleConstant.h>
#include "pathtracing/Camera.h"
#include <vtkm/cont/internal/DeviceAdapterAlgorithmGeneral.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>
#include <omp.h>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/worklet/Invoker.h>
#include <vtkm/rendering/raytracing/BoundingVolumeHierarchy.h>
#include <vtkm/cont/ArrayHandleExtractComponent.h>

#include <raytracing/RayTracerNormals.h>
#include <raytracing/RayTracerAlbedo.h>
#include "MapperPathTracer.h"

#include <fstream>
#include "CornellBox.h"

#include <vtkm/rendering/MapperRayTracer.h>
#include "MapperQuad.h"
#include "MapperQuadNormals.h"
#include "MapperQuadAlbedo.h"
#include <vtkm/rendering/Scene.h>
#include "View3D.h"

#include <algorithm>

// to allow python binding
#include "pybind11/include/pybind11/pybind11.h"
#include <pybind11/include/pybind11/stl.h> //for conversion to python type

// adios for i/o with python
#include <adios2.h>

//for groups of vectors to pass to python
#include <array>
#include <vector>

// opaque types to avoid saving buffers to memory and allow
// call by reference from within python
//PYBIND11_MAKE_OPAQUE(std::vector<int, std::allocator<int>>) // to prevent copying vectors
//// not sure if needed
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int, std::allocator<int>>, std::allocator<std::vector<int, std::allocator<int>>>>) // to prevent copying vectors
////PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>)
////PYBIND11_MAKE_OPAQUE(std::array<std::vector<std::vector<int>>,4>)
using ColorBufferType = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4>>;
using DepthBufferType = vtkm::cont::ArrayHandle<vtkm::Float32>;

PYBIND11_MAKE_OPAQUE(ColorBufferType)
PYBIND11_MAKE_OPAQUE(DepthBufferType)

namespace py = pybind11;


using BufferHandle = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,4>>;
using ArrayType = vtkm::cont::ArrayHandle<vec3>; //vtkm::Vec<vtkm::Float32, 3>;= vec3
using vtkNestVec = vtkm::cont::ArrayHandleCompositeVector<
                                            vtkm::cont::ArrayHandle<vtkm::Float32>,
                                            vtkm::cont::ArrayHandle<vtkm::Float32>,
                                            vtkm::cont::ArrayHandle<vtkm::Float32>>;


template<typename VecType>
inline VecType de_nan(const VecType& c) {
    auto temp = c;
    if (!(temp[0] == temp[0])) temp[0] = 0;
    if (!(temp[1] == temp[1])) temp[1] = 0;
    if (!(temp[2] == temp[2])) temp[2] = 0;
    return temp;
}


const auto
parse(int argc, char **argv){
  int x = 128;
  int y = 128;
  int s = 10;
  int depth = 5;

  bool hemi = false;
  bool direct = false;
  for (int i=1; i<argc; i++){
    if (!strcmp(argv[i], "-x")){
      if (i+1 < argc){
        x = atoi(argv[i+1]);
        i += 1;
      }

    }
    else if (!strcmp(argv[i], "-y")){
      if (i+1 < argc){
        y = atoi(argv[i+1]);
        i += 1;
      }
    }
    else if (!strcmp(argv[i], "-samplecount")){
      if (i+1 < argc){
        s = atoi(argv[i+1]);
        i += 1;
      }
    }
    else if (!strcmp(argv[i], "-raydepth")){
      if (i+1 < argc){
        depth = atoi(argv[i+1]);
        i += 1;
      }
    }
    else if (!strcmp(argv[i], "-hemisphere"))
    {
      hemi = true;
    }
    else if(!strcmp(argv[i], "-direct"))
      direct = true;
  }

  return std::make_tuple(x,y, s, depth, hemi, direct);
}
std::vector<double> norm_color_range( std::vector<double> color_vals){
    std::vector<double> normalized_colors = color_vals;
    for(int i=0; i<color_vals.size(); i++)
        normalized_colors[i] = color_vals[i]/255.0;
    return normalized_colors;
}
void runRay(int nx, int ny, int samplecount, int depthcount,
              vtkm::rendering::Canvas &canvas, vtkm::rendering::Camera &cam)
{
  CornellBox cb;
  path::rendering::MapperQuad mapper;
  auto ds = cb.buildDataSet();
  vtkm::rendering::Scene scene;


  std::vector<double> colors ={113, 31 ,30,
                               84, 23, 23,
                              132 ,128 ,126,
                               48, 93 ,53,
                              69 ,63 ,59,
                              54 ,47 ,43,
                              109,104, 102,
                              251, 251, 251,
                              1, 1, 1,
                              35, 66 ,38,
                              91 ,86, 84,
                              25, 32, 21};
  colors = norm_color_range(colors);


  std::vector<double> red = {1, 0, 0};
  std::vector<double> white = {1, 1, 1};
  std::vector<double> green = {0, 0, 1};

  std::vector<double> blue = {0, 0, 1};
  //0,1,1 0,1,0 1,1,0
  std::vector<double> lamb1 = {0, 1, 1};
  std::vector<double> lamb2 = {0, 1, 0};
  std::vector<double> lamb3 = {1, 1, 0};

  std::vector<double> c1 = {0.65, 0.05, 0.05}; //red
  std::vector<double> c2 = {0.73, 0.73, 0.73}; //white
  std::vector<double> c3 = {0.12, 0.45, 0.15}; //green
  std::vector<double> fill1 = {15, 15, 15};

  std::vector<double> pallet;
  int num_quads = 12;
  int num_colors = 3;
  pallet.reserve(num_quads*num_colors);


  pallet.insert(pallet.end(), c3.begin(), c3.end()); //green
  pallet.insert(pallet.end(), c1.begin(), c1.end()); //red
  pallet.insert(pallet.end(), fill1.begin(), fill1.end()); //light
  for (int i=0; i<num_quads - 3; i++)
    pallet.insert(pallet.end(), c2.begin(), c2.end()); //white


  std::vector<double> alpha(num_quads);
  for (int i=0; i<alpha.size(); i++)
      alpha[i] = 1.0;

  vtkm::cont::ColorTable ct_12_quad("pallet_color_table",
                            vtkm::cont::ColorSpace::RGB,
                            vtkm::Vec<double,3>(0,0,0),
                            pallet, alpha);

   scene.AddActor(vtkm::rendering::Actor(
                      ds.GetCellSet(),
                      ds.GetCoordinateSystem(),
                      ds.GetField("point_var"),
                      ct_12_quad));

  vtkm::rendering::Color background(0,0,0, 1.0f);
  vtkm::rendering::Color foreground(1,1,1, 1.0f);
  vtkm::rendering::View3D view(scene, mapper, canvas, cam, background, foreground);

  view.Initialize();
  view.Paint();

}
void runNorms(int nx, int ny, int samplecount, int depthcount,
              vtkm::rendering::Canvas &canvas, vtkm::rendering::Camera &cam)
{
  CornellBox cb;
  path::rendering::MapperQuadNormals mapper;
  auto ds = cb.buildDataSet();
  vtkm::rendering::Scene scene;

  scene.AddActor(vtkm::rendering::Actor(
                   ds.GetCellSet(),
                   ds.GetCoordinateSystem(),
                   ds.GetField("point_var"),
                   vtkm::cont::ColorTable{vtkm::cont::ColorTable::Preset::COOL_TO_WARM_EXTENDED}));
  vtkm::rendering::Color background(0,0,0, 1.0f);
  vtkm::rendering::Color foreground(1,1,1, 1.0f);
  vtkm::rendering::View3D view(scene, mapper, canvas, cam, background, foreground);

  view.Initialize();
  view.Paint();

}
void runAlbedo(int nx, int ny, int samplecount, int depthcount,
              vtkm::rendering::Canvas &canvas, vtkm::rendering::Camera &cam)
{
  CornellBox cb;
  path::rendering::MapperQuadAlbedo mapper;
  auto ds = cb.buildDataSet();
  vtkm::rendering::Scene scene;

  std::vector<double> c1 = {0.65, 0.05, 0.05}; //red
  std::vector<double> c2 = {0.73, 0.73, 0.73}; //white
  std::vector<double> c3 = {0.12, 0.45, 0.15}; //green
  std::vector<double> fill1 = {15, 15, 15};

  std::vector<double> pallet;
  int num_quads = 12;
  int num_colors = 3;
  pallet.reserve(num_quads*num_colors);
  pallet.insert(pallet.end(), c3.begin(), c3.end()); //green
  pallet.insert(pallet.end(), c1.begin(), c1.end()); //red
  pallet.insert(pallet.end(), fill1.begin(), fill1.end()); //light
  for (int i=0; i<num_quads - 3; i++)
    pallet.insert(pallet.end(), c2.begin(), c2.end()); //white
  std::vector<double> alpha(num_quads); //alpha
  for (int i=0; i<alpha.size(); i++)
      alpha[i] = 1.0;

  vtkm::cont::ColorTable ct_12_quad("pallet_color_table",
                            vtkm::cont::ColorSpace::RGB,
                            vtkm::Vec<double,3>(0,0,0),
                            pallet, alpha);

  scene.AddActor(vtkm::rendering::Actor(
                   ds.GetCellSet(),
                   ds.GetCoordinateSystem(),
                   ds.GetField("point_var"),
                   ct_12_quad));
  vtkm::rendering::Color background(0,0,0, 1.0f);
  vtkm::rendering::Color foreground(1,1,1, 1.0f);
  vtkm::rendering::View3D view(scene, mapper, canvas, cam, background, foreground);

  view.Initialize();
  view.Paint();

}
void runPath(int nx, int ny, int samplecount, int depthcount,
              vtkm::rendering::Canvas &canvas, vtkm::rendering::Camera &cam)
{
  using MyAlgos = details::PathAlgorithms<vtkm::cont::DeviceAdapterAlgorithm<VTKM_DEFAULT_DEVICE_ADAPTER_TAG>, VTKM_DEFAULT_DEVICE_ADAPTER_TAG>;
  using StorageTag = vtkm::cont::StorageTagBasic;
  using Device = VTKM_DEFAULT_DEVICE_ADAPTER_TAG;

  CornellBox cb;

  auto ds = cb.buildDataSet();

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 2>> uvs;
  uvs.Allocate(nx*ny);

  vtkm::rendering::MapperPathTracer mapper(samplecount,
                                           depthcount,
                                           cb.matIdx,
                                           cb.texIdx,
                                           cb.matType,
                                           cb.texType,
                                           cb.tex);

  mapper.SetCanvas(&canvas);



  vtkm::cont::Field field;
  vtkm::cont::ColorTable ct;
  vtkm::Range sr;
  mapper.RenderCells(ds.GetCellSet(0),
                     cb.coord,
                     field,
                     ct,
                     cam,
                     sr);


} //in sample count save atten
vtkm::cont::ArrayHandleCompositeVector<
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>> runPathAlbedo(int nx, int ny, int samplecount, int depthcount,
              vtkm::rendering::Canvas &canvas, vtkm::rendering::Camera &cam)
{
    using MyAlgos = details::PathAlgorithms<vtkm::cont::DeviceAdapterAlgorithm<VTKM_DEFAULT_DEVICE_ADAPTER_TAG>, VTKM_DEFAULT_DEVICE_ADAPTER_TAG>;
    using StorageTag = vtkm::cont::StorageTagBasic;
    using Device = VTKM_DEFAULT_DEVICE_ADAPTER_TAG;

    CornellBox cb;

    auto ds = cb.buildDataSet();

    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 2>> uvs;
    uvs.Allocate(nx*ny);

    vtkm::rendering::MapperPathTracer mapper(samplecount,
                                             depthcount,
                                             cb.matIdx,
                                             cb.texIdx,
                                             cb.matType,
                                             cb.texType,
                                             cb.tex);

    mapper.SetCanvas(&canvas);



    vtkm::cont::Field field;
    vtkm::cont::ColorTable ct;
    vtkm::Range sr;
    mapper.RenderCells(ds.GetCellSet(0),
                       cb.coord,
                       field,
                       ct,
                       cam,
                       sr);


    using StorageTag = vtkm::cont::StorageTagBasic;
    using vec3CompositeType = vtkm::cont::ArrayHandleCompositeVector<
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>,
      vtkm::cont::ArrayHandle<vtkm::Float32>>;
    vec3CompositeType attenuation = mapper.get_attenuation();
    return attenuation;
}




// first called
template<typename ArrayType>
std::vector<std::vector<int>> passMatrix(std::vector<std::vector<int>> buffer,
          int nx, int ny, int samplecount,
          ArrayType &cols)
{

    std::vector<int> clmn_order = {2,1,0};
    //fs << "P3\n" << nx << " "  << ny << " 255" << std::endl;
    //for (int i=0; i<cols.GetNumberOfValues(); i++){    //reversed for image orientation
    for (int i=cols.GetNumberOfValues()-1; i>=0 ; i--){
      auto col = cols.GetPortalConstControl().Get(i);
      if (col != col)
        col = 0.0f;
      passMatrix(buffer, samplecount, col);
    }

    return buffer;

}

// return the coordinate in the image dimensions indexing
// given linear values and dimensions of file
void CoordinateFromLinearIndex(int idx, int dim_x, int dim_y, double& x, double& y, double& z){
  x =  idx % (dim_x);
  idx /= (dim_x);
  y = idx % (dim_y);
  idx /= (dim_y);
  z = idx;
}
// %% save to file func :

template<typename ValueType>
void save(std::fstream &fs,
          int samplecount,
          ValueType &col);

template<>
void save(std::fstream &fs,
          int samplecount,
          vtkm::Vec<vtkm::Float32,4> &col)
{
  col = de_nan(col);
  col = col / float(samplecount);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  int ir = int(255.99*col[0]);
  int ig = int(255.99*col[1]);
  int ib = int(255.99*col[2]);
  fs << ir << " " << ig << " " << ib << std::endl;

}
template<typename ValueType>
void save_diff(std::fstream &fs,
          int samplecount,
          ValueType &col1,
               ValueType &col2);

template<>
void save_diff(std::fstream &fs,
          int samplecount,
          vtkm::Vec<vtkm::Float32,4> &col1,
          vtkm::Vec<vtkm::Float32,4> &col2)
{
  col1 = de_nan(col1);
  col1 = col1 / float(samplecount);
  col1[0] = sqrt(col1[0]);
  col1[1] = sqrt(col1[1]);
  col1[2] = sqrt(col1[2]);
  col2 = de_nan(col2);
  col2 = col2 / float(samplecount);
  col2[0] = sqrt(col2[0]);
  col2[1] = sqrt(col2[1]);
  col2[2] = sqrt(col2[2]);
  int ir = std::max(int(255.99*col1[0] - 255.99*col2[0]), 0);
  int ig = std::max(int(255.99*col1[1] -  255.99*col2[1]),0);
  int ib = std::max(int(255.99*col1[2] -  255.99*col2[2]),0);
  fs << ir << " " << ig << " " << ib << std::endl;

}
template<>
void save(std::fstream &fs,
          int samplecount,
          vtkm::Float32 &col)
{
  col = sqrt(col);
  int ir = int(255.99*col);
  int ig = int(255.99*col);
  int ib = int(255.99*col);
  fs << ir << " " << ig << " " << ib << std::endl;

}

template<typename ArrayType>
void save(std::string fn,
          int nx, int ny, int samplecount,
          ArrayType &cols)
{
  std::fstream fs;
  fs.open(fn.c_str(), std::fstream::out);
  if (fs.is_open()){
    std::vector<int> clmn_order = {2,1,0};
    fs << "P3\n" << nx << " "  << ny << " 255" << std::endl;
    //for (int i=0; i<cols.GetNumberOfValues(); i++){    //reversed for image orientation
    for (int i=cols.GetNumberOfValues()-1; i>=0 ; i--){
      auto col = cols.GetPortalConstControl().Get(i);
      if (col != col)
        col = 0.0f;
      save(fs, samplecount, col);
    }
    fs.close();
  }
  else
    std::cout << "Couldn't save pnm." << std::endl;
//  std::vector<std::uint8_t> PngBuffer;
}

template<typename vtkNestVec>
void save(std::string fn,
          int nx, int ny, int samplecount,
          vtkm::cont::ArrayHandleCompositeVector<
                vtkm::cont::ArrayHandle<vtkm::Float32>,
                vtkm::cont::ArrayHandle<vtkm::Float32>,
                vtkm::cont::ArrayHandle<vtkm::Float32>> &albedos)
{
  std::fstream fs;
  fs.open(fn.c_str(), std::fstream::out);
  if (fs.is_open()){
    fs << "P3\n" << nx << " "  << ny << " 255" << std::endl;
    for (int i=0; i<3; i++){
      auto alb_r = albedos.GetPortalConstControl().Get(i)[0];//.GetPortalConstControl().Get(i);
      auto alb_b = albedos.GetPortalConstControl().Get(i)[1];//.GetPortalConstControl().Get(i);
      auto alb_g = albedos.GetPortalConstControl().Get(i)[2];//.GetPortalConstControl().Get(i);
      vtkm::Float32 sum_alb = alb_r + alb_b + alb_g;
      if (sum_alb != sum_alb)
          sum_alb = 0.0f;
      save(fs, samplecount, sum_alb);
    }
    fs.close();
  }
  else
    std::cout << "Couldn't save pnm." << std::endl;
//  std::vector<std::uint8_t> PngBuffer;
}

template<typename ArrayType>
void save_buffer_diff(std::string fn,
          int nx, int ny, int samplecount,
          ArrayType &albedo, ArrayType &direct)
{
  std::fstream fs;
  fs.open(fn.c_str(), std::fstream::out);
  if (fs.is_open()){
    fs << "P3\n" << nx << " "  << ny << " 255" << std::endl;
    for (int i=0; i<albedo.GetNumberOfValues(); i++){
      auto albedo_col =albedo.GetPortalConstControl().Get(i);
      auto direct_col = direct.GetPortalConstControl().Get(i) ;
      if (albedo_col != albedo_col)
          albedo_col = 0.0f;
      if (direct_col != direct_col)
          direct_col = 0.0f;
      save_diff(fs, samplecount, albedo_col, direct_col);
    }
    fs.close();
  }
  else
    std::cout << "Couldn't save pnm." << std::endl;
//  std::vector<std::uint8_t> PngBuffer;
}


//
// Adios outfile
//
template<typename ValueType>
void saveADIOS(std::fstream &fs,
          int samplecount,
          ValueType &col);


template<typename ArrayType>
void saveADIOS(std::string fn,
               int nx, int ny, int samplecount,
               ArrayType &cols)
{
  adios2::ADIOS adios(adios2::DebugON);
  adios2::IO bpIO = adios.DeclareIO("BPFile_N2N");

  adios2::Variable<vtkm::Float32> bpOut = bpIO.DefineVariable<vtkm::Float32>(
        "pnms", {}, {}, {static_cast<std::size_t>(nx*ny)}, adios2::ConstantDims);

  adios2::Engine writer = bpIO.Open(fn, adios2::Mode::Write);

  auto *ptr = cols.GetStorage().GetArray();
  writer.Put<vtkm::Float32>(bpOut, ptr );
  writer.Close();

}
template<>
void saveADIOS(std::fstream &fs,
          int samplecount,
          vtkm::Float32 &col,
             int nx, int ny,   int linear_coordinate)
{
  col = sqrt(col);
  int ir = int(255.99*col);
  int ig = int(255.99*col);
  int ib = int(255.99*col);

  adios2::ADIOS adios(adios2::DebugON);
  adios2::IO bpIO = adios.DeclareIO("BPFile_N2N");

  int x;
  int y;
  int z;
  CoordinateFromLinearIndex(linear_coordinate, nx, ny, &x, &y, &z);
  adios2::Variable<int> bpOut = bpIO.DefineVariable<int>(
        "pnms", {1,1}, {x,y}, {nx,ny}, adios2::ConstantDims);
   //type -> //shape //start //count //dim
  //note: count specifies amount to write in total dim
  adios2::Engine writer = bpIO.Open(fn, adios2::Mode::Write);
  writer.Put<vtkm::Float32>(bpOut, ir );
  writer.Close();

  CoordinateFromLinearIndex(linear_coordinate, nx, ny, &x, &y, &z);
  adios2::Variable<int> bpOut = bpIO.DefineVariable<int>(
        "pnms", {1,1}, {x,y}, {nx,ny}, adios2::ConstantDims);
   //type -> //shape //start //count //dim
  //note: count specifies amount to write in total dim
  adios2::Engine writer = bpIO.Open(fn, adios2::Mode::Write);
  writer.Put<vtkm::Float32>(bpOut, ig );
  writer.close();

  CoordinateFromLinearIndex(linear_coordinate, nx, ny, &x, &y, &z);
  adios2::Variable<int> bpOut = bpIO.DefineVariable<int>(
        "pnms", {1,1}, {x,y}, {nx,ny}, adios2::ConstantDims);
   //type -> //shape //start //count //dim
  //note: count specifies amount to write in total dim
  adios2::Engine writer = bpIO.Open(fn, adios2::Mode::Write);
  writer.Put<vtkm::Float32>(bpOut, ib );
  write.close();

}
template<>
void saveADIOS(std::fstream &fs,
          int samplecount,
          vtkm::Vec<vtkm::Float32,4> &col,
               int nx, int ny, int linear_coordinate)
{
  col = de_nan(col);
  col = col / float(samplecount);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  int ir = int(255.99*col[0]);
  int ig = int(255.99*col[1]);
  int ib = int(255.99*col[2]);
  fs << ir << " " << ig << " " << ib << std::endl;

  adios2::ADIOS adios(adios2::DebugON);
  adios2::IO bpIO = adios.DeclareIO("BPFile_N2N");

  adios2::Variable<int> bpOut = bpIO.DefineVariable<int>(
        "pnms", {nx,ny}, {0,0}, {nx,ny}, adios2::ConstantDims);
   //type -> //shape(global) //start(current offset) //count(current dim) //dim
  //note: count specifies amount to write in total dim
  adios2::Engine writer = bpIO.Open(fn, adios2::Mode::Write);

  writer.Put<vtkm::Float32>(bpOut, ir );
  writer.Put<vtkm::Float32>(bpOut, ig );
  writer.Put<vtkm::Float32>(bpOut, ib );
  writer.Close();

}

template<typename vtkNestVec>
void saveADIOS(std::string fn,
          int samplecount,
          vtkm::cont::ArrayHandleCompositeVector<
                vtkm::cont::ArrayHandle<vtkm::Float32>,
                vtkm::cont::ArrayHandle<vtkm::Float32>,
                vtkm::cont::ArrayHandle<vtkm::Float32>> &albedos,
               int nx, int ny, int linear_coordinate)
{
  std::fstream fs;
  fs.open(fn.c_str(), std::fstream::out);
  if (fs.is_open()){
    fs << "P3\n" << nx << " "  << ny << " 255" << std::endl;
    for (int i=0; i<3; i++){
      auto alb_r = albedos.GetPortalConstControl().Get(i)[0];//.GetPortalConstControl().Get(i);
      auto alb_b = albedos.GetPortalConstControl().Get(i)[1];//.GetPortalConstControl().Get(i);
      auto alb_g = albedos.GetPortalConstControl().Get(i)[2];//.GetPortalConstControl().Get(i);
      vtkm::Float32 sum_alb = alb_r + alb_b + alb_g;
      if (sum_alb != sum_alb)
          sum_alb = 0.0f;
      saveADIOS(fs, samplecount, sum_alb, nx, ny, linear_coordinate + i);
    }
    fs.close();
  }
  else
    std::cout << "Couldn't save pnm." << std::endl;
//  std::vector<std::uint8_t> PngBuffer;
}
template<typename ArrayType>
void saveADIOS(std::string fn,
          int samplecount,
          ArrayType &cols,
               int nx, int ny)
{
  std::fstream fs;
  fs.open(fn.c_str(), std::fstream::out);
  if (fs.is_open()){
    std::vector<int> clmn_order = {2,1,0};
    fs << "P3\n" << nx << " "  << ny << " 255" << std::endl;
    //for (int i=0; i<cols.GetNumberOfValues(); i++){    //reversed for image orientation
    for (int i=cols.GetNumberOfValues()-1; i>=0 ; i--){
      auto col = cols.GetPortalConstControl().Get(i);
      if (col != col)
        col = 0.0f;
      saveADIOS(fs, samplecount, col, nx, ny, i);
    }
    fs.close();
  }
  else
    std::cout << "Couldn't save pnm." << std::endl;
//  std::vector<std::uint8_t> PngBuffer;
}
void generateHemisphere(int nx, int ny, int samplecount, int depthcount, bool direct, bool save_image)
{
  vtkm::rendering::CanvasRayTracer canvas(nx,ny);
  vtkm::rendering::Camera cam;
  cam.SetPosition(vec3(278,278,-800));
  cam.SetFieldOfView(40.f);
  cam.SetViewUp(vec3(0,1,0));
  cam.SetLookAt(vec3(278,278,278));

  int numPhi = 30;
  int numTheta = 30; //for 1080 upside down both 50

  float rTheta = (2.0*M_PI)/float(numTheta);
  float rPhi = (M_PI/2.0)/float(numPhi);

  float r = -1078;
  float phi_start = -1*M_PI*0.25;//M_PI*0.25;  //original angles for 1080 upsidedown
  float phi_stop = M_PI/2.0;
  float theta_start = M_PI/8.0; //0
  float theta_stop = 7.0*M_PI/8.0; //2*M_PI
  for (float phi=phi_start; phi<phi_stop; phi += rPhi){
    //std::cout << "Phi: " << phi << std::endl;
    for (float theta=theta_start; theta<theta_stop; theta+=rTheta){
      auto x = r * cos(theta) * sin(phi);
      auto y = r * sin(theta) * sin(phi);
      auto z = r * cos(phi);
      //std::cout << " x " << x << " y " <<y <<" z " << z << std::endl;
      //std::cout << "PHI " << phi << " THETA " << theta << " R " << r << std::endl;
      if(x!=x) x=0;
      if(y!=y) y=0;
      if(z!=z) z=0;
      vec3 pos(x+278, y+278, z+278 );
      cam.SetPosition(pos);
      std::stringstream sstr;
      if (direct){

          //
          // TODO: Once faster, use passMatrix to feed pytorch hemisphere in else statements
          //
        sstr << "direct-" << phi << "-" << theta << ".pnm";
        runRay(nx,ny,samplecount, depthcount, canvas, cam);
        if(save_image)
            save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());

        sstr.str("");
        sstr << "depth-" << phi << "-" << theta << ".pnm";
        if(save_image)
            save(sstr.str(), nx, ny, samplecount, canvas.GetDepthBuffer());

        sstr.str("");
        runNorms(nx,ny,samplecount,depthcount, canvas, cam);
        sstr << "normals-" << phi << "-" << theta << ".pnm";
        if(save_image)
            save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());

        sstr.str("");
        runAlbedo(nx,ny,samplecount,depthcount, canvas, cam);
        sstr << "albedo-" << phi << "-" << theta << ".pnm";
        if(save_image)
            save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
      }
      else{
        sstr << "output-" << phi << "-" << theta << ".pnm";
        runPath(nx,ny, samplecount, depthcount, canvas, cam);
      }
      if(save_image)
        save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
    }
  }
}

std::array<BufferHandle, 4> renderFromOrientation(
        const std::string &buffer_type,
        int nx, int ny, int samplecount, int depthcount,
        float theta, float phi,
        bool save_image)
{
      vtkm::rendering::CanvasRayTracer canvas(nx,ny);
      vtkm::rendering::Camera cam;
      cam.SetClippingRange(500.f, 2000.f);
      cam.SetPosition(vec3(278,278,-800));
      cam.SetFieldOfView(40.f);
      cam.SetViewUp(vec3(0,1,0));
      cam.SetLookAt(vec3(278,278,278));

      int numPhi = 30;
      int numTheta = 30; //for 1080 upside down both 50

      //float rTheta = (2.0*M_PI)/float(numTheta);
      //float rPhi = (M_PI/2.0)/float(numPhi);

      float r = -1078;

      auto x = r * cos(theta) * sin(phi);
      auto y = r * sin(theta) * sin(phi);
      auto z = r * cos(phi);
      if(x!=x) x=0;
      if(y!=y) y=0;
      if(z!=z) z=0;

      // position with angle
      vec3 pos(x+278, y+278, z+278 );
      cam.SetPosition(pos);
      std::stringstream sstr;


      ColorBufferType direct_buffer;
      //DepthBufferType depth_buffer; //DepthBufferType
      ColorBufferType path_trace;
      ColorBufferType normal_buffer;
      ColorBufferType albedo_buffer;

      runRay(nx,ny,samplecount, depthcount, canvas, cam);
      if(buffer_type == "direct" || buffer_type == "dataPack"){
          sstr.str("");
          sstr << "direct-" << phi << "-" << theta << ".pnm";
          //runRay(nx,ny,samplecount, depthcount, canvas, cam);
          if(save_image)
              save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
          else
              direct_buffer = canvas.GetColorBuffer();

      }

      if(buffer_type == "path" || buffer_type == "dataPack"){
        sstr.str("");
        sstr << "output-" << phi << "-" << theta << ".pnm";
        runPath(nx,ny, samplecount, depthcount, canvas, cam);
        if(save_image)
          save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
        else
          path_trace = canvas.GetColorBuffer();
      }


      if(buffer_type == "normal" || buffer_type == "dataPack"){
          sstr.str("");
          runNorms(nx,ny,samplecount,depthcount, canvas, cam);
          sstr << "normals-" << phi << "-" << theta << ".pnm";
          if(save_image)
              save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
          else
              normal_buffer = canvas.GetColorBuffer();
      }


      if(buffer_type == "albedo" || buffer_type == "dataPack"){
        sstr.str("");
        runAlbedo(nx,ny,samplecount,depthcount, canvas, cam);
        sstr << "albedo-" << phi << "-" << theta << ".pnm";
        if(save_image)
          save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
        else
          albedo_buffer = canvas.GetColorBuffer();
      }


      std::array<BufferHandle, 4> conditionalImagePack;
      conditionalImagePack[0] = path_trace;
      conditionalImagePack[1] = direct_buffer;// depth_buffer;
      conditionalImagePack[2] = normal_buffer;
      conditionalImagePack[3] = albedo_buffer;


      return conditionalImagePack;
}

DepthBufferType renderDepthBuffer(
        int nx, int ny, int samplecount, int depthcount,
        float theta, float phi,
        bool save_image)
{
      vtkm::rendering::CanvasRayTracer canvas(nx,ny);
      vtkm::rendering::Camera cam;
      cam.SetClippingRange(500.f, 2000.f);
      cam.SetPosition(vec3(278,278,-800));
      cam.SetFieldOfView(40.f);
      cam.SetViewUp(vec3(0,1,0));
      cam.SetLookAt(vec3(278,278,278));

      int numPhi = 30;
      int numTheta = 30; //for 1080 upside down both 50

      //float rTheta = (2.0*M_PI)/float(numTheta);
      //float rPhi = (M_PI/2.0)/float(numPhi);

      float r = -1078;

      auto x = r * cos(theta) * sin(phi);
      auto y = r * sin(theta) * sin(phi);
      auto z = r * cos(phi);
      if(x!=x) x=0;
      if(y!=y) y=0;
      if(z!=z) z=0;

      // position with angle
      vec3 pos(x+278, y+278, z+278 );
      cam.SetPosition(pos);
      std::stringstream sstr;


      ColorBufferType direct_buffer;
      DepthBufferType depth_buffer; //DepthBufferType
      ColorBufferType path_trace;
      ColorBufferType normal_buffer;
      ColorBufferType albedo_buffer;

      runRay(nx,ny,samplecount, depthcount, canvas, cam);

      sstr.str("");
      sstr << "depth-" << phi << "-" << theta << ".pnm";
      if(save_image)
          save(sstr.str(), nx, ny, samplecount, canvas.GetDepthBuffer());
      else
          depth_buffer = canvas.GetDepthBuffer();



      return depth_buffer;
}



std::array<ColorBufferType, 4> traceFromOrientation(
                                 int nx, int ny, int samplecount, int depthcount,
                                 float theta, float phi,
                                 bool save_image)
  {
    vtkm::rendering::CanvasRayTracer canvas(nx,ny);
    vtkm::rendering::Camera cam;
    cam.SetPosition(vec3(278,278,-800));
    cam.SetFieldOfView(40.f);
    cam.SetViewUp(vec3(0,1,0));
    cam.SetLookAt(vec3(278,278,278));

    int numPhi = 30;
    int numTheta = 30; //for 1080 upside down both 50

    //float rTheta = (2.0*M_PI)/float(numTheta);
    //float rPhi = (M_PI/2.0)/float(numPhi);

    float r = -1078;

    auto x = r * cos(theta) * sin(phi);
    auto y = r * sin(theta) * sin(phi);
    auto z = r * cos(phi);
    //std::cout << " x " << x << " y " <<y <<" z " << z << std::endl;
    //std::cout << "PHI " << phi << " THETA " << theta << " R " << r << std::endl;
    if(x!=x) x=0;
    if(y!=y) y=0;
    if(z!=z) z=0;
    vec3 pos(x+278, y+278, z+278 );
    cam.SetPosition(pos);
    std::stringstream sstr;
    sstr.str("");
    sstr << "output-" << phi << "-" << theta << ".pnm";
    runPath(nx,ny, samplecount, depthcount, canvas, cam);
    //std::vector<std::vector<int>> pathTracedImage;
    ColorBufferType pathTracedImage;
    pathTracedImage = canvas.GetColorBuffer();

    std::array<ColorBufferType, 4> conditionalImagePack;
    conditionalImagePack[0] = pathTracedImage;

    if(save_image)
        save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());

    return conditionalImagePack; //error here


}

ColorBufferType pathTraceImage(int nx, int ny,
                   int samplecount, int depthcount,
                   int hemisphere,  int buffers){//char *argv[]) {

  //const auto tup = parse(argc, argv);
  //const int nx = std::get<0>(tup);
  //const int ny = std::get<1>(tup);
  //const int samplecount = std::get<2>(tup);
  //const int depthcount = std::get<3>(tup);
  bool hemi = hemisphere != 0;//std::get<4>(tup);
  //bool conditionals = buffers != 0;//std::get<5>(tup);

  if (hemi){
      std::string temp = "under construction";
      ColorBufferType pathTracedImage;
      //return generateHemisphere(nx,ny, samplecount, depthcount, conditionals, false);


  }
  else
  {
    vtkm::rendering::CanvasRayTracer canvas(nx,ny);
    vtkm::rendering::Camera cam;
    cam.SetClippingRange(500.0f, 2000.f);
    cam.SetPosition(vec3(278,278,-800));
    cam.SetFieldOfView(40.);
    cam.SetViewUp(vec3(0,1,0));
    cam.SetLookAt(vec3(278,278,278));

    runPath(nx,ny, samplecount, depthcount, canvas, cam);
    //std::stringstream sstr;
    //sstr << "output.pnm";
    ColorBufferType pathTracedImage;//(4, std::vector<int>(nx* ny));
    pathTracedImage = canvas.GetColorBuffer();

    return pathTracedImage;
  }
}


int saveTracedImage(int nx, int ny,
                   int samplecount, int depthcount,
                   int hemisphere,  int buffers){//char *argv[]) {


  bool hemi = hemisphere != 0;//std::get<4>(tup);
  bool conditionals = buffers != 0;//std::get<5>(tup);

  if (hemi)
    generateHemisphere(nx,ny, samplecount, depthcount, conditionals, true);
  else
  {
    vtkm::rendering::CanvasRayTracer canvas(nx,ny);
    vtkm::rendering::Camera cam;
    cam.SetClippingRange(500.0f, 2000.f);
    cam.SetPosition(vec3(278,278,-800));
    cam.SetFieldOfView(40.);
    cam.SetViewUp(vec3(0,1,0));
    cam.SetLookAt(vec3(278,278,278));

    runPath(nx,ny, samplecount, depthcount, canvas, cam);
    std::stringstream sstr;
    sstr << "output.pnm";
    save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());

  }

  return 1;
}

int saveRenderedBuffers(int nx, int ny,
                   int samplecount, int depthcount,
                   int hemisphere,  int buffers){//char *argv[]) {

  bool hemi = hemisphere != 0;//std::get<4>(tup);
  bool direct = buffers != 0;//std::get<5>(tup);

  if (hemi)
    generateHemisphere(nx,ny, samplecount, depthcount, direct, true);
  else
  {
    vtkm::rendering::CanvasRayTracer canvas(nx,ny);
    vtkm::rendering::Camera cam;
    cam.SetClippingRange(500.0f, 2000.f);
    cam.SetPosition(vec3(278,278,-800));
    cam.SetFieldOfView(40.);
    cam.SetViewUp(vec3(0,1,0));
    cam.SetLookAt(vec3(278,278,278));
    if (direct){
      runRay(nx,ny,samplecount,depthcount, canvas, cam);
      std::stringstream sstr;
      sstr << "direct.pnm";
      save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
      sstr.str("");
      sstr << "depth.pnm";
      save(sstr.str(), nx, ny, samplecount, canvas.GetDepthBuffer());
      sstr.str("");
      runNorms(nx,ny,samplecount,depthcount, canvas, cam);
      sstr << "normals.pnm";
      save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());
      sstr.str("");
      runAlbedo(nx,ny,samplecount,depthcount, canvas, cam);//
      sstr << "albedo.pnm";//
      save(sstr.str(), nx, ny, samplecount, canvas.GetColorBuffer());//
    }
  }
  return 1;
}


std::array<ColorBufferType, 4>  renderBuffers(int nx, int ny,          //std::array<std::vector< std::vector<int> >,4>
                   int samplecount, int depthcount,
                   int hemisphere,  int buffers){//char *argv[]) {


  bool hemi = hemisphere != 0;//std::get<4>(tup);
  bool direct = buffers != 0;//std::get<5>(tup);

  if (hemi){

    std::array<ColorBufferType ,4> conditionalImagePack;

    ColorBufferType direct_buffer;
    ColorBufferType depth_buffer;
    ColorBufferType normal_buffer;
    ColorBufferType albedo_buffer;

    // need to collect all orientations and feed to python
    generateHemisphere(nx,ny, samplecount, depthcount, direct, false);

    conditionalImagePack[0] = direct_buffer;
    conditionalImagePack[1] = depth_buffer;
    conditionalImagePack[2] = normal_buffer;
    conditionalImagePack[3] = albedo_buffer;

    return conditionalImagePack;
  }
  else
  {
    vtkm::rendering::CanvasRayTracer canvas(nx,ny);
    vtkm::rendering::Camera cam;
    cam.SetClippingRange(500.0f, 2000.f);
    cam.SetPosition(vec3(278,278,-800));
    cam.SetFieldOfView(40.);
    cam.SetViewUp(vec3(0,1,0));
    cam.SetLookAt(vec3(278,278,278));
    //if (direct){

      std::array<ColorBufferType,4> conditionalImagePack;

      ColorBufferType direct_buffer;

      runRay(nx,ny,samplecount,depthcount, canvas, cam);
      direct_buffer = canvas.GetColorBuffer();

//      ColorBufferType depth_buffer;

//      //sstr << "depth.pnm";
//      depth_buffer = canvas.GetDepthBuffer();
      //sstr.str("");

      ColorBufferType normal_buffer;// = reserveImageMatrix(nx, ny);

      runNorms(nx,ny,samplecount,depthcount, canvas, cam);
      normal_buffer = canvas.GetColorBuffer();

      ColorBufferType albedo_buffer;// = reserveImageMatrix(nx, ny);
      runAlbedo(nx,ny,samplecount,depthcount, canvas, cam);
      //sstr << "albedo.pnm";
      albedo_buffer = canvas.GetColorBuffer();

      conditionalImagePack[0] = direct_buffer;
      conditionalImagePack[1] = normal_buffer; //////////////////////////////depth_buffer;
      conditionalImagePack[2] = normal_buffer;
      conditionalImagePack[3] = albedo_buffer;

      return conditionalImagePack;
  }
}

//TODO
//write to adios in C++ and read from python adios
// fix direct non hemisphere
// convert buffer to array type applicable to python (getportanlcontrol)
// save above with adios in c++
// define opaque datatypes for images / buffers and to contain images / buffers?

 using namespace pybind11::literals;

PYBIND11_MODULE(trainingTracer, m){ //called at import, module example, m denotes variable type py::module
  m.doc() = "path tracer and buffer rendering implementation in vtk-m";// optional module docstring
  // return as matrix defs
  m.def("pathTraceImage", &pathTraceImage, "Render Path Traced Image and return as %%%%% type matrix");
  m.def("renderBuffers", &renderBuffers, "Render Path Traced Image and return as %%%% type matrix");
  // store to file defs
  m.def("saveTracedImage", &saveTracedImage, "Render Path Traced Image and save to file");//, py::arg("i"), py::arg("j")); //method generating bianry code exposing add() to python, arg allows keyword arguments for function calls from within python.}
  m.def("saveRenderedBuffers", &saveRenderedBuffers, "Render Conditional Image Buffers and save to file");
  //m.def("renderFromOrientation", &renderFromOrientation, "Render Conditional Image Buffers or path traced image from given camera angle perspective");
  m.def("renderFromOrientation", &renderFromOrientation, "Render Conditional Image Buffers or path traced image from given camera angle perspective",
       "buffer_type"_a, "nx"_a, "ny"_a, "samplecount"_a, "depthcount"_a, "theta"_a, "phi"_a,"save_image"_a);
  m.def("traceFromOrientation", &traceFromOrientation, "path trace Image Buffers or path traced image from given camera angle perspective",
       "nx"_a, "ny"_a, "samplecount"_a, "depthcount"_a, "theta"_a, "phi"_a,"save_image"_a);
  m.def("renderDepthBuffer", &renderDepthBuffer, "Render Conditional Image Buffers or path traced image from given camera angle perspective",
        "nx"_a, "ny"_a, "samplecount"_a, "depthcount"_a, "theta"_a, "phi"_a,"save_image"_a);

    //to pass in place so nothing stored
    py::class_<ColorBufferType>(m, "ColorBuffer")
        .def(py::init<>())
        //.def("clear", &std::vector<std::vector<int>>::clear)
        //.def("pop_back", &std::vector<std::vector<int>>::pop_back)
        .def("__len__", [](ColorBufferType &cBuff) {return  cBuff.GetNumberOfValues(); })
        .def("__getitem__", [](ColorBufferType &cBuff, ssize_t i) {
            if(i >= cBuff.GetNumberOfValues())
                throw py::index_error();
            std::vector<float> c =  {cBuff.GetPortalConstControl().Get(i)[0],
                        cBuff.GetPortalConstControl().Get(i)[1],
                        cBuff.GetPortalConstControl().Get(i)[2]};
            return c;
          });
    py::class_<DepthBufferType>(m, "DepthBuffer")
        .def(py::init<>())
        //.def("clear", &std::vector<std::vector<int>>::clear)
        //.def("pop_back", &std::vector<std::vector<int>>::pop_back)
        .def("__len__", [](DepthBufferType &dBuff) {return dBuff.GetNumberOfValues(); })
        .def("__getitem__", [](DepthBufferType &dBuff, ssize_t i) {
            if(i >= dBuff.GetNumberOfValues())
                throw py::index_error();
            return dBuff.GetPortalConstControl().Get(i);
          });
  }


PYBIND11_MODULE(trainingTracer_cuda, m){ //called at import, module example, m denotes variable type py::module
  m.doc() = "path tracer and buffer rendering implementation in vtk-m with cuda";// optional module docstring
  // return as matrix defs
  m.def("pathTraceImage", &pathTraceImage, "Render Path Traced Image and return as %%%%% type matrix");
  m.def("renderBuffers", &renderBuffers, "Render Path Traced Image and return as %%%% type matrix");
  // store to file defs
  m.def("saveTracedImage", &saveTracedImage, "Render Path Traced Image and save to file");//, py::arg("i"), py::arg("j")); //method generating bianry code exposing add() to python, arg allows keyword arguments for function calls from within python.}
  m.def("saveRenderedBuffers", &saveRenderedBuffers, "Render Conditional Image Buffers and save to file");
  m.def("renderFromOrientation", &renderFromOrientation, "Render Conditional Image Buffers or path traced image from given camera angle perspective",
       "buffer_type"_a, "nx"_a, "ny"_a, "samplecount"_a, "depthcount"_a, "theta"_a, "phi"_a,"save_image"_a);
  m.def("traceFromOrientation", &traceFromOrientation, "Path Trace Image Buffers or path traced image from given camera angle perspective",
       "nx"_a, "ny"_a, "samplecount"_a, "depthcount"_a, "theta"_a, "phi"_a,"save_image"_a);
  m.def("renderDepthBuffer", &renderDepthBuffer, "Render Depth Buffer Image from given camera angle perspective",
        "nx"_a, "ny"_a, "samplecount"_a, "depthcount"_a, "theta"_a, "phi"_a,"save_image"_a);
  py::class_<ColorBufferType>(m, "ColorBuffer")
      .def(py::init<>())
      //.def("clear", &std::vector<std::vector<int>>::clear)
      //.def("pop_back", &std::vector<std::vector<int>>::pop_back)
      .def("__len__", [](ColorBufferType &cBuff) {return  cBuff.GetNumberOfValues(); })
      .def("__getitem__", [](ColorBufferType &cBuff, ssize_t i) {
          if(i >= cBuff.GetNumberOfValues())
              throw py::index_error();
          std::vector<float> c = {cBuff.GetPortalConstControl().Get(i)[0],
                      cBuff.GetPortalConstControl().Get(i)[1],
                      cBuff.GetPortalConstControl().Get(i)[2]};
          return c;
        });
  py::class_<DepthBufferType>(m, "DepthBuffer")
      .def(py::init<>())
      //.def("clear", &std::vector<std::vector<int>>::clear)
      //.def("pop_back", &std::vector<std::vector<int>>::pop_back)
      .def("__len__", [](DepthBufferType &dBuff) {return dBuff.GetNumberOfValues(); })
      .def("__getitem__", [](DepthBufferType &dBuff, ssize_t i) {
          if(i >= dBuff.GetNumberOfValues())
              throw py::index_error();
          return dBuff.GetPortalConstControl().Get(i);
        });
}
