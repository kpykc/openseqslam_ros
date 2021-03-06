This is a rosified fork of [c++ implementation by Subokita](https://github.com/subokita/OpenSeqSLAM).

This is a C++ + OpenCV port of Niko Sünderhauf's OpenSeqSLAM, whose original code can be found here:
http://www.tu-chemnitz.de/etit/proaut/mitarbeiter/niko.html


OpenSeqSLAM
===========

Copyright 2013, Niko Sünderhauf
Chemnitz University of Technology
niko@etit.tu-chemnitz.de

OpenSeqSLAM is an open source Matlab implementation of the original SeqSLAM
algorithm published by Milford and Wyeth at ICRA12 [1]. SeqSLAM performs place
recognition by matching sequences of images. 

Quick start guide: 
 - Download the Nordland dataset:
     cd datasets/norland; ./getDataset.bash; 
 - start Matlab and run demo.m from within the matlab directory


[1] Michael Milford and Gordon F. Wyeth (2012). SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights. In Proc. of IEEE Intl. Conf. on Robotics and Automation (ICRA)
