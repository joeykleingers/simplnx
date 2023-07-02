# Find Feature Phases Binary

## Group (Subgroup)

Generic (Misc)

## Description

This **Filter** assigns an **Ensemble** Id number to binary data. The *true* **Cells** will be **Ensemble** 1, and *false* **Cells** will be **Ensemble** 0. This **Filter** is generally useful when the **Cell Ensembles** were not known ahead of time. For example, if an image is segmented into precipitates and non-precipitates, this **Filter** will assign the precipitates to **Ensemble** 1, and the non-precipitates to **Ensemble** 0.

## Parameters

None

## Required Geometry

Image

## Required Objects

| Kind | Default Name | Type | Component Dimensions | Description |
|------|--------------|------|----------------------|-------------|
| **Cell Attribute Array** | FeatureIds | int32_t | (1) | Specifies to which **Feature** each **Cell** belongs |
| **Cell Attribute Array** | Mask | bool | (1) | Specifies if the **Cell** is to be counted in the algorithm |
| **Attribute Matrix** | Cell Data Attribute Matrix | Attribute Matrix | N/A | The *Cell Data* **Attribute Matrix** of the **Image Geometry** where the *Binary Phases Array* will be created |

## Created Objects

| Kind | Default Name | Type | Component Dimensions | Description |
|------|--------------|------|----------------------|-------------|
| **Feature Attribute Array** | Binary Feature Phases Array Name | int32_t | (1) | Specifies to which **Ensemble** each **Feature** belongs |

## Example Pipelines

## License & Copyright

Please see the description file distributed with this **Plugin**

## DREAM.3D Mailing Lists

If you need more help with a **Filter**, please consider asking your question on the [DREAM.3D Users Google group!](https://groups.google.com/forum/?hl=en#!forum/dream3d-users)