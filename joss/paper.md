---
title: 'HSI-Wizard: A magical open-source Python package for medical hyperspectral imaging applications'
tags:
  - Python
  - hyperspectral imaging
  - DataCubes
authors:
  - name: Felix Wühler
    orcid: 0000-0002-4821-8549
    equal-contrib: false
    affiliation: 1
  - name: Matthias Rädle
    orcid: 0000-0002-3382-7608
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: CeMOS Research and Transfer Center, University of Applied Science Mannheim, Germany
   index: 1
date: 18 November 2024
bibliography: paper.bib
---

# Summary

`hsi-wizard` is an open-source Python package for processing, analyzing, and visualizing hyperspectral datasets. While its methods are adaptable to other fields, the core emphasis is on analytics for medical applications. Hyperspectral data, in this context, refers to images where each pixel contains multiple pieces of information, typically intensity values at different wavelengths. The number of spectral bands per pixel (n) can range from just a few to over 2000. These datasets originate from various sources, including scanners, imaging devices, and tomographic slides. Since hyperspectral data can come with different resolutions and pixel offsets, hsi-wizard enables seamless integration of multiple data sources. `hsi-wizard` provides an intuitive environment for hyperspectral analysis. With the `DataCube`, a standardized class for representing hyperspectral data is presented. This ensures uniform and consistent processing of hyperspectral data. In addition, `hsi-wizard` provides tools to manipulate, explore, and analyze `DataCubes`.  

A key feature of the `DataCube` class is a protocol functionality that records the methods used from the `hsi-wizard` library. The protocol serves as a reusable template for other datasets, enabling researchers to reproduce results with minimal programming effort. Another feature of the `hsi-wizard` is a method for merging `DataCubes`. This makes it easy to combine different modalities into one large dataset. 

The `hsi-wizard` simplifies the workflow for processing, analyzing, and visualizing hyperspectral data. It is designed to cater to both beginners and experts. Students can use the straightforward methods for educational purposes, while researchers can leverage advanced functions for professional studies, with the advantage that the results are reproducible and transparently documented.

# Statement of Need

Hyperspectral imaging (HSI) enables detailed analysis of the electromagnetic spectrum across multiple wavelengths for each pixel in an image. Originally developed for NASA applications, HSI is now used in a wide range of fields [@Bhargava]. This versatility has resulted in diverse methods for acquiring hyperspectral data, including different measurement techniques (reflection, transmission, fluorescence), wavelength ranges (ultraviolet, visible, infrared), and scanning methods (e.g., point scanning, line scanning, and Fourier transform infrared imaging (FTIR)) [@Guolan].

To streamline the analysis of these diverse datasets and eliminate the need for developing custom methods for each, `hsi-wizard` standardizes their representation through the `DataCube` class. It enables uniform analysis, manipulation, and visualization through well-defined functions. Furthermore, the standardization is essential for enabling data fusion across different scanning processes. Different scans from various scanners result in unique datasets with differing resolutions, information and aspect ratios. Therefore, the first step in merging these scans is to use a well-defined data representation combined with methods that can address these challenges. `hsi-wizard` provides this capability, making it easier to combine datasets for further analysis.

In addition, `hsi-wizard` allows users to log and save the manipulations of the `DataCube`. These logs can be reused for similar `DataCubes`, ensuring consistent analysis across different measurements with minimal effort. This feature reduces the need for programming expertise and enhances reproducibility, thereby supporting the goal of increased transparency in research [@Burke] [@Knottnerus].

Besides the supplied functionalities, `hsi-wizard` is also extensible, allowing users to integrate additional methods and customize the package for diverse applications.

# Future Work

The development of `hsi-wizard` is ongoing. Ideas, feedback, and contributions are happily welcomed. All versions of `hsi-wizard` are available on the Python Package Index (PyPI). 

# Acknowledgements

`hsi-wizard` is the result of several published articles like [@vanmarwick], [@heintz], [@kummel], [@nachtmann], and [@manser]. The work on `hsi-wizard` was funded by CeMOS Research & Transfer Center and the University of Applied Science Mannheim. 

# References

