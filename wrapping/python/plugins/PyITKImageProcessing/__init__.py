from PyITKImageProcessingPlugin.Plugin import PyITKImageProcessingPlugin
from PyITKImageProcessingPlugin.ITKGrayscaleFillholeImage import ITKGrayscaleFillholeImage

def get_plugin():
  return PyITKImageProcessingPlugin()

__all__ = ['PyITKImageProcessingPlugin', 'ITKGrayscaleFillholeImage', 'get_plugin']
