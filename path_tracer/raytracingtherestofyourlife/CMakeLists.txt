cmake_minimum_required(VERSION 3.8)

project(PathTracing VERSION 0.1 LANGUAGES CXX)

find_package(VTKm REQUIRED
  OPTIONAL_COMPONENTS CUDA)

#VTKm requires OpenMP because pthreads is required in ColorTable(!!)
  find_package(OpenMP REQUIRED)

set (PathTracingSrc 
main.cc 
CornellBox.cpp 
pathtracing/Camera.cxx 
pathtracing/PathTracer.cxx
pathtracing/SphereExtractor.cxx 
pathtracing/QuadExtractor.cxx 
pathtracing/QuadIntersector.cxx
pathtracing/SphereIntersector.cxx 
pathtracing/WhichGenerateDir.cxx 
CanvasRayTracer.cxx
MapperPathTracer.cxx 
MapperQuad.cxx
raytracing/OrigQuadIntersector.cxx
raytracing/ChannelBuffer.cxx
raytracing/RayTracerNormals.cxx
raytracing/RayTracerAlbedo.cxx
MapperQuadNormals.cxx
MapperQuadAlbedo.cxx
View3D.cxx )
set (PathTracingHdr
pathtracing/Camera.h

pathtracing/CosineGenerateDir.h
pathtracing/PathAlgorithms.h
pathtracing/PathTracer.h
pathtracing/Pdf.h
pathtracing/QuadExtractor.h
pathtracing/QuadGenerateDir.h
pathtracing/SphereExtractor.h
pathtracing/SphereIntersector.h
pathtracing/SphereGenerateDir.h
pathtracing/SpherePdf.h
pathtracing/WhichGenerateDir.h
pathtracing/Worklets.h 
pathtracing/QuadIntersector.h
pathtracing/AABBSurface.h 
pathtracing/BVHTraverser.h
pathtracing/EmitWorklet.h 
pathtracing/ScatterWorklet.h 
pathtracing/SurfaceWorklets.h 
pathtracing/Surface.h 
pathtracing/PdfWorklet.h 
pathtracing/Record.h 
pathtracing/onb.h
MapperPathTracer.h
MapperQuad.h
raytracing/ChannelBuffer.h
raytracing/ChannelBufferOperations.h
raytracing/Ray.h
raytracing/RayTracerNormals.h
raytracing/RayTracerAlbedo.h
MapperQuadNormals.h
MapperQuadAlbedo.h
CornellBox.h 
pathtracing/vec3.h 
pathtracing/wangXor.h 
pathtracing/GenerateDir.h 
pathtracing/QuadPdf.h 
pathtracing/SpherePdf.h
View3D.h
pathtracing/RayOperations.h)

add_executable(CornellBox ${PathTracingSrc} ${PathTracingHdr})


set_target_properties(CornellBox PROPERTIES
  CXX_STANDARD 17)

target_include_directories(CornellBox
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  )
target_link_libraries(CornellBox PUBLIC vtkm_cont vtkm_rendering OpenMP::OpenMP_CXX)


if(VTKm_ENABLE_CUDA)
  vtkm_compile_as_cuda(cudaSource ${PathTracingSrc})
  add_executable(CornellBox_CUDA ${cudaSource} )
  target_compile_definitions(CornellBox_CUDA PRIVATE "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_CUDA")
  target_include_directories(CornellBox_CUDA
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    )

  set_target_properties(CornellBox_CUDA PROPERTIES
    CUDA_STANDARD 14)
  target_link_libraries(CornellBox_CUDA PUBLIC vtkm_cont vtkm_rendering)

endif()
