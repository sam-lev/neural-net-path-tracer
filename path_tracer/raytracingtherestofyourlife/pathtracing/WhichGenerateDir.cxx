#include "WhichGenerateDir.h"
#include "PdfWorklet.h"
#include <vtkm/worklet/Invoker.h>


void WhichGenerateDir::apply(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays)
{
  vtkm::worklet::Invoker Invoke;

  WorketletGenerateDir genDir(3);
  Invoke(genDir, seeds, whichPdf);
}


