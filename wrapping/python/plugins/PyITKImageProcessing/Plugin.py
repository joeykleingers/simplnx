from PyITKImageProcessing.ITKGrayscaleFillholeImage import ITKGrayscaleFillholeImage
import simplnx as sx

class PyITKImageProcessingPlugin:
  def __init__(self) -> None:
    pass

  def id(self) -> sx.Uuid:
    return sx.Uuid('3efe91e4-a962-465a-8d5a-bd7e5269b0ce')

  def name(self) -> str:
    return 'PyITKImageProcessingPlugin'

  def description(self) -> str:
    return 'Python filters that wrap the ITK Software library. ITK is located at https://github.com/InsightSoftwareConsortium/ITK'

  def vendor(self) -> str:
    return 'BlueQuartz Software'

  def get_filters(self):
    return [ITKGrayscaleFillholeImage]
