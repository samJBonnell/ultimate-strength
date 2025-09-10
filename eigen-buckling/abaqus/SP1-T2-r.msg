1

   Abaqus 2020                                  Date 08-Sep-2025   Time 17:30:14
   For use by UNIVERSITY OF BRITISH COLUMBIA under license from Dassault Systemes or its subsidiary.

                                                                                 
 Incremental                                                                     
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


          Incremental                                                                     

     AUTOMATIC LOAD CONTROL WITH -
          A SUGGESTED INITIAL INCREMENT IN ARC LENGTH           5.000E-02
          AND A TOTAL ARC LENGTH SCALE FACTOR OF                 1.00    
          THE MINIMUM ARC LENGTH INCREMENT ALLOWED IS           1.000E-05
          THE MAXIMUM ARC LENGTH INCREMENT ALLOWED IS           1.000E+36

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

     MODIFIED RIKS METHOD IS USED - AMPLITUDE REFERENCES ARE IGNORED

     THE STEP ENDS WHEN LOAD PROPORTIONALITY FACTOR EXCEEDS    11.5    

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                      50

          LARGE DISPLACEMENT THEORY WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     2.000E-02

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       55.8       TIME AVG. FORCE        55.8    
 LARGEST RESIDUAL FORCE              27.1       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.206E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.206E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.100E-02   TIME AVG. MOMENT      1.100E-02
 LARGEST RESIDUAL MOMENT            2.691E-04   AT NODE      11327   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.264E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.264E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       47.0       TIME AVG. FORCE        47.0    
 LARGEST RESIDUAL FORCE            -1.346E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.216E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.774E-06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.877E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.110E-02   TIME AVG. MOMENT      1.110E-02
 LARGEST RESIDUAL MOMENT           -3.620E-07   AT NODE      11328   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.275E-04   AT NODE      11476   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.096E-06   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     1 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         4.2061E-02
     TOTAL ARC LENGTH   5.000E-02, INCREMENT OF ARC LENGTH   5.000E-02


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       187.       TIME AVG. FORCE        117.    
 LARGEST RESIDUAL FORCE              27.6       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.222E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.222E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     2.240E-02   TIME AVG. MOMENT      1.675E-02
 LARGEST RESIDUAL MOMENT            1.871E-04   AT NODE      11327   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.282E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.282E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       178.       TIME AVG. FORCE        113.    
 LARGEST RESIDUAL FORCE            -5.453E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.226E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.311E-06   AT NODE        302   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.572E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.244E-02   TIME AVG. MOMENT      1.677E-02
 LARGEST RESIDUAL MOMENT            9.024E-08   AT NODE      11474   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.286E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.728E-07   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    7.500E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     2 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         7.2416E-02
     TOTAL ARC LENGTH   0.100    , INCREMENT OF ARC LENGTH   5.000E-02


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       234.       TIME AVG. FORCE        153.    
 LARGEST RESIDUAL FORCE              63.1       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.843E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.843E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     3.930E-02   TIME AVG. MOMENT      2.428E-02
 LARGEST RESIDUAL MOMENT           -3.123E-04   AT NODE      11475   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.934E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.934E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       223.       TIME AVG. FORCE        149.    
 LARGEST RESIDUAL FORCE            -1.071E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.848E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.197E-06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    5.425E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     3.933E-02   TIME AVG. MOMENT      2.429E-02
 LARGEST RESIDUAL MOMENT           -1.829E-07   AT NODE      22499   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.939E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.957E-07   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.113    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     3 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1047    
     TOTAL ARC LENGTH   0.175    , INCREMENT OF ARC LENGTH   7.500E-02


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       274.       TIME AVG. FORCE        181.    
 LARGEST RESIDUAL FORCE              144.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.775E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.775E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.472E-02   TIME AVG. MOMENT      3.440E-02
 LARGEST RESIDUAL MOMENT            4.761E-04   AT NODE      11327   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.913E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.913E-04   AT NODE      11326   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       264.       TIME AVG. FORCE        178.    
 LARGEST RESIDUAL FORCE            -1.564E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.778E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.858E-06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.203E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     6.473E-02   TIME AVG. MOMENT      3.440E-02
 LARGEST RESIDUAL MOMENT           -3.574E-07   AT NODE      22349   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.917E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.524E-07   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.169    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     4 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1364    
     TOTAL ARC LENGTH   0.288    , INCREMENT OF ARC LENGTH   0.113    


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       310.       TIME AVG. FORCE        205.    
 LARGEST RESIDUAL FORCE              325.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.169E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.169E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.103       TIME AVG. MOMENT      4.810E-02
 LARGEST RESIDUAL MOMENT           -6.618E-04   AT NODE      11475   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.379E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.379E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       300.       TIME AVG. FORCE        203.    
 LARGEST RESIDUAL FORCE            -1.706E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.170E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.064E-06   AT NODE        302   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.103       TIME AVG. MOMENT      4.810E-02
 LARGEST RESIDUAL MOMENT            5.962E-07   AT NODE        453   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.384E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.637E-07   AT NODE         50   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.253    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     5 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1653    
     TOTAL ARC LENGTH   0.456    , INCREMENT OF ARC LENGTH   0.169    


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       340.       TIME AVG. FORCE        225.    
 LARGEST RESIDUAL FORCE              735.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.254E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.254E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.160       TIME AVG. MOMENT      6.678E-02
 LARGEST RESIDUAL MOMENT            8.351E-04   AT NODE      11327   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.580E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.580E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       331.       TIME AVG. FORCE        224.    
 LARGEST RESIDUAL FORCE            -1.256E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.251E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.728E-06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.160       TIME AVG. MOMENT      6.679E-02
 LARGEST RESIDUAL MOMENT            8.633E-07   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.587E-04   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.872E-07   AT NODE        112   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.380    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     6 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1904    
     TOTAL ARC LENGTH   0.709    , INCREMENT OF ARC LENGTH   0.253    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       340.       TIME AVG. FORCE        241.    
 LARGEST RESIDUAL FORCE             1.664E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.369E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.369E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.246       TIME AVG. MOMENT      9.244E-02
 LARGEST RESIDUAL MOMENT            9.304E-04   AT NODE      11327   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.891E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.891E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       342.       TIME AVG. FORCE        241.    
 LARGEST RESIDUAL FORCE             3.455E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.355E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.875E-06   AT NODE        302   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.246       TIME AVG. MOMENT      9.245E-02
 LARGEST RESIDUAL MOMENT           -1.085E-06   AT NODE      22499   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.908E-04   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.611E-06   AT NODE       7324   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.570    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     7 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2121    
     TOTAL ARC LENGTH    1.09    , INCREMENT OF ARC LENGTH   0.380    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       322.       TIME AVG. FORCE        251.    
 LARGEST RESIDUAL FORCE             3.780E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.401E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.401E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.376       TIME AVG. MOMENT      0.128    
 LARGEST RESIDUAL MOMENT           -1.039E-03   AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.489E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.489E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       321.       TIME AVG. FORCE        251.    
 LARGEST RESIDUAL FORCE             9.777E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.397E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.385E-06   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.376       TIME AVG. MOMENT      0.128    
 LARGEST RESIDUAL MOMENT           -1.769E-06   AT NODE      11423   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.494E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.127E-06   AT NODE       7324   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.854    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     8 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2327    
     TOTAL ARC LENGTH    1.66    , INCREMENT OF ARC LENGTH   0.570    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       334.       TIME AVG. FORCE        260.    
 LARGEST RESIDUAL FORCE             8.676E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.087E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.087E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.571       TIME AVG. MOMENT      0.177    
 LARGEST RESIDUAL MOMENT           -2.628E-03   AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.250E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.250E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       337.       TIME AVG. FORCE        260.    
 LARGEST RESIDUAL FORCE             0.935       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.074E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.343E-05   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.572       TIME AVG. MOMENT      0.177    
 LARGEST RESIDUAL MOMENT            9.241E-06   AT NODE      22553   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.264E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.456E-05   AT NODE       7324   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.720E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.28    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     9 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2568    
     TOTAL ARC LENGTH    2.51    , INCREMENT OF ARC LENGTH   0.854    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       371.       TIME AVG. FORCE        271.    
 LARGEST RESIDUAL FORCE             2.033E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.087E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.087E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.868       TIME AVG. MOMENT      0.246    
 LARGEST RESIDUAL MOMENT            7.924E-03   AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.422E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.422E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       382.       TIME AVG. FORCE        272.    
 LARGEST RESIDUAL FORCE              8.36       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.049E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.875E-05   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.871       TIME AVG. MOMENT      0.247    
 LARGEST RESIDUAL MOMENT            6.036E-05   AT NODE      22598   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.463E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.209E-05   AT NODE      22767   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       382.       TIME AVG. FORCE        272.    
 LARGEST RESIDUAL FORCE             7.292E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.048E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.082E-07   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.871       TIME AVG. MOMENT      0.247    
 LARGEST RESIDUAL MOMENT           -6.486E-08   AT NODE      15383   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.464E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.810E-06   AT NODE       6191   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.92    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    10 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2932    
     TOTAL ARC LENGTH    3.79    , INCREMENT OF ARC LENGTH    1.28    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       458.       TIME AVG. FORCE        289.    
 LARGEST RESIDUAL FORCE             4.943E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.506E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.506E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.32       TIME AVG. MOMENT      0.345    
 LARGEST RESIDUAL MOMENT            2.628E-02   AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.264E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.264E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       476.       TIME AVG. FORCE        291.    
 LARGEST RESIDUAL FORCE              64.5       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.406E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.913E-05   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.33       TIME AVG. MOMENT      0.345    
 LARGEST RESIDUAL MOMENT           -4.155E-04   AT NODE       6039   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.362E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.098E-04   AT NODE      22769   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       475.       TIME AVG. FORCE        291.    
 LARGEST RESIDUAL FORCE             0.288       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.402E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.798E-06   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.33       TIME AVG. MOMENT      0.345    
 LARGEST RESIDUAL MOMENT           -6.165E-07   AT NODE       6965   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.367E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.173E-05   AT NODE      17214   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.88    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    11 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.3580    
     TOTAL ARC LENGTH    5.72    , INCREMENT OF ARC LENGTH    1.92    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       618.       TIME AVG. FORCE        318.    
 LARGEST RESIDUAL FORCE             1.272E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.435E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.435E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.05       TIME AVG. MOMENT      0.487    
 LARGEST RESIDUAL MOMENT           -9.103E-02   AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.201E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.201E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       656.       TIME AVG. FORCE        321.    
 LARGEST RESIDUAL FORCE              425.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.226E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.091E-04   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.07       TIME AVG. MOMENT      0.489    
 LARGEST RESIDUAL MOMENT            2.738E-03   AT NODE      17518   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.360E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.798E-04   AT NODE         30   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       654.       TIME AVG. FORCE        321.    
 LARGEST RESIDUAL FORCE              6.84       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.206E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.028E-05   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.07       TIME AVG. MOMENT      0.489    
 LARGEST RESIDUAL MOMENT            4.858E-06   AT NODE      17363   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.376E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.252E-05   AT NODE       4833   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       654.       TIME AVG. FORCE        321.    
 LARGEST RESIDUAL FORCE            -6.355E-05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.206E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.004E-08   AT NODE      18498   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      2.07       TIME AVG. MOMENT      0.489    
 LARGEST RESIDUAL MOMENT           -2.057E-08   AT NODE        126   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.376E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.182E-07   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     4.32    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    12 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.4769    
     TOTAL ARC LENGTH    8.60    , INCREMENT OF ARC LENGTH    2.88    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       956.       TIME AVG. FORCE        370.    
 LARGEST RESIDUAL FORCE             3.498E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.965E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.965E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.24       TIME AVG. MOMENT      0.700    
 LARGEST RESIDUAL MOMENT            0.312       AT NODE        219   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.280E-02   AT NODE      22741   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.280E-02   AT NODE      22741   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       987.       TIME AVG. FORCE        372.    
 LARGEST RESIDUAL FORCE             2.353E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.647E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.203E-04   AT NODE      19052   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.28       TIME AVG. MOMENT      0.704    
 LARGEST RESIDUAL MOMENT            1.827E-02   AT NODE      22535   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE      22698   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.470E-03   AT NODE      22675   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       977.       TIME AVG. FORCE        372.    
 LARGEST RESIDUAL FORCE              126.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.591E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.446E-05   AT NODE      19304   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.30       TIME AVG. MOMENT      0.705    
 LARGEST RESIDUAL MOMENT           -1.365E-04   AT NODE        170   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.350E-02   AT NODE      22756   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.121E-04   AT NODE      22774   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -2.331E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       977.       TIME AVG. FORCE        372.    
 LARGEST RESIDUAL FORCE             5.337E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.591E-03   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.307E-07   AT NODE       4002   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      3.30       TIME AVG. MOMENT      0.705    
 LARGEST RESIDUAL MOMENT            8.764E-07   AT NODE      22671   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.350E-02   AT NODE      22756   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.036E-06   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     6.49    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    13 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.6858    
     TOTAL ARC LENGTH    12.9    , INCREMENT OF ARC LENGTH    4.32    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.681E+03   TIME AVG. FORCE        465.    
 LARGEST RESIDUAL FORCE             1.019E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.241E-02   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.241E-02   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      5.31       TIME AVG. MOMENT       1.03    
 LARGEST RESIDUAL MOMENT             1.34       AT NODE        191   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.191E-02   AT NODE      22689   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.191E-02   AT NODE      22689   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.519E+03   TIME AVG. FORCE        453.    
 LARGEST RESIDUAL FORCE             6.744E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.220E-02   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.050E-04   AT NODE       2850   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      5.38       TIME AVG. MOMENT       1.04    
 LARGEST RESIDUAL MOMENT           -0.107       AT NODE        178   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.333E-02   AT NODE      22685   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.835E-03   AT NODE        133   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.481E+03   TIME AVG. FORCE        451.    
 LARGEST RESIDUAL FORCE             1.346E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.214E-02   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.409E-04   AT NODE       2850   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      5.42       TIME AVG. MOMENT       1.04    
 LARGEST RESIDUAL MOMENT            3.030E-03   AT NODE        288   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.420E-02   AT NODE        119   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.156E-03   AT NODE        131   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    3.285E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.481E+03   TIME AVG. FORCE        451.    
 LARGEST RESIDUAL FORCE             0.590       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.214E-02   AT NODE      11401   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.803E-06   AT NODE      19045   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      5.42       TIME AVG. MOMENT       1.04    
 LARGEST RESIDUAL MOMENT           -2.400E-05   AT NODE      22786   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.419E-02   AT NODE      22769   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.198E-05   AT NODE      19027   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     9.73    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    14 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.031    
     TOTAL ARC LENGTH    19.4    , INCREMENT OF ARC LENGTH    6.49    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.261E+03   TIME AVG. FORCE        638.    
 LARGEST RESIDUAL FORCE             2.871E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.803E-02   AT NODE       9136   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.803E-02   AT NODE       9136   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.94       TIME AVG. MOMENT       1.57    
 LARGEST RESIDUAL MOMENT             7.16       AT NODE        180   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.963E-02   AT NODE         29   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.963E-02   AT NODE         29   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.407E+03   TIME AVG. FORCE        581.    
 LARGEST RESIDUAL FORCE             9.902E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.810E-02   AT NODE       8230   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.128E-04   AT NODE      21002   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.99       TIME AVG. MOMENT       1.57    
 LARGEST RESIDUAL MOMENT            0.476       AT NODE      22633   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.103E-02   AT NODE         25   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.513E-03   AT NODE      22662   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.346E+03   TIME AVG. FORCE        577.    
 LARGEST RESIDUAL FORCE             5.216E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.814E-02   AT NODE       7928   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.507E-04   AT NODE      19404   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.06       TIME AVG. MOMENT       1.58    
 LARGEST RESIDUAL MOMENT            2.383E-02   AT NODE      22640   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.262E-02   AT NODE         23   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.582E-03   AT NODE       2114   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.349E+03   TIME AVG. FORCE        577.    
 LARGEST RESIDUAL FORCE              3.07       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.814E-02   AT NODE       7928   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.770E-05   AT NODE      19706   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.06       TIME AVG. MOMENT       1.58    
 LARGEST RESIDUAL MOMENT            3.231E-04   AT NODE      22661   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.263E-02   AT NODE         23   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.536E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.349E+03   TIME AVG. FORCE        577.    
 LARGEST RESIDUAL FORCE            -2.676E-04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.814E-02   AT NODE       7928   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.679E-07   AT NODE      19857   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      9.06       TIME AVG. MOMENT       1.58    
 LARGEST RESIDUAL MOMENT           -5.867E-07   AT NODE       4434   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.263E-02   AT NODE         23   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.039E-06   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    15 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.601    
     TOTAL ARC LENGTH    29.1    , INCREMENT OF ARC LENGTH    9.73    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.330E+03   TIME AVG. FORCE        812.    
 LARGEST RESIDUAL FORCE             3.199E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.823E-02   AT NODE      15780   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.823E-02   AT NODE      15780   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.9       TIME AVG. MOMENT       2.29    
 LARGEST RESIDUAL MOMENT             11.7       AT NODE      22636   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.547E-02   AT NODE      22783   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.547E-02   AT NODE      22783   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.452E+03   TIME AVG. FORCE        757.    
 LARGEST RESIDUAL FORCE             5.329E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.821E-02   AT NODE      16082   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.942E-04   AT NODE       1351   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      13.0       TIME AVG. MOMENT       2.29    
 LARGEST RESIDUAL MOMENT           -0.576       AT NODE      21441   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.681E-02   AT NODE      22785   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.287E-03   AT NODE        143   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.406E+03   TIME AVG. FORCE        754.    
 LARGEST RESIDUAL FORCE             1.251E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.831E-02   AT NODE       6418   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.762E-04   AT NODE       2945   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      13.0       TIME AVG. MOMENT       2.29    
 LARGEST RESIDUAL MOMENT            1.486E-02   AT NODE      21987   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.761E-02   AT NODE         17   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.380E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.409E+03   TIME AVG. FORCE        754.    
 LARGEST RESIDUAL FORCE              2.84       AT NODE      22726   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.831E-02   AT NODE      16384   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.463E-05   AT NODE       2643   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      13.0       TIME AVG. MOMENT       2.29    
 LARGEST RESIDUAL MOMENT           -6.283E-04   AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.764E-02   AT NODE         17   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.543E-04   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    16 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.215    
     TOTAL ARC LENGTH    38.9    , INCREMENT OF ARC LENGTH    9.73    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.511E+03   TIME AVG. FORCE       1.034E+03
 LARGEST RESIDUAL FORCE             3.349E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.821E-02   AT NODE       5965   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.821E-02   AT NODE       5965   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      17.2       TIME AVG. MOMENT       3.17    
 LARGEST RESIDUAL MOMENT            -19.6       AT NODE      22510   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.973E-02   AT NODE      22664   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.973E-02   AT NODE      22664   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.614E+03   TIME AVG. FORCE        981.    
 LARGEST RESIDUAL FORCE             2.545E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.805E-02   AT NODE       5814   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.014E-04   AT NODE        149   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      17.2       TIME AVG. MOMENT       3.17    
 LARGEST RESIDUAL MOMENT            0.592       AT NODE      21595   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.113E-02   AT NODE      22663   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.032E-03   AT NODE      22795   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.581E+03   TIME AVG. FORCE        979.    
 LARGEST RESIDUAL FORCE              404.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.815E-02   AT NODE      17139   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.005E-04   AT NODE       2492   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      17.3       TIME AVG. MOMENT       3.18    
 LARGEST RESIDUAL MOMENT            1.183E-02   AT NODE        692   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.162E-02   AT NODE         13   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.109E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -6.219E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.584E+03   TIME AVG. FORCE        980.    
 LARGEST RESIDUAL FORCE             -3.51       AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.815E-02   AT NODE      17139   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.684E-05   AT NODE       2190   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      17.3       TIME AVG. MOMENT       3.18    
 LARGEST RESIDUAL MOMENT            6.765E-04   AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.164E-02   AT NODE         13   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.964E-04   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     14.6    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    17 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.870    
     TOTAL ARC LENGTH    48.6    , INCREMENT OF ARC LENGTH    9.73    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.626E+03   TIME AVG. FORCE       1.460E+03
 LARGEST RESIDUAL FORCE             7.783E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.695E-02   AT NODE      17441   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.695E-02   AT NODE      17441   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.0       TIME AVG. MOMENT       4.33    
 LARGEST RESIDUAL MOMENT             65.4       AT NODE        294   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.017E-02   AT NODE        141   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.017E-02   AT NODE        141   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.714E+03   TIME AVG. FORCE       1.298E+03
 LARGEST RESIDUAL FORCE            -3.672E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.654E-02   AT NODE       5210   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.845E-04   AT NODE      22800   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.0       TIME AVG. MOMENT       4.33    
 LARGEST RESIDUAL MOMENT            -2.38       AT NODE        905   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.247E-02   AT NODE        142   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.356E-03   AT NODE        605   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.632E+03   TIME AVG. FORCE       1.294E+03
 LARGEST RESIDUAL FORCE             2.824E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.686E-02   AT NODE      17743   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.158E-03   AT NODE      20595   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.2       TIME AVG. MOMENT       4.34    
 LARGEST RESIDUAL MOMENT            9.064E-02   AT NODE        553   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.397E-02   AT NODE        142   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.053E-03   AT NODE      22709   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.644E+03   TIME AVG. FORCE       1.294E+03
 LARGEST RESIDUAL FORCE              333.       AT NODE      22726   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.691E-02   AT NODE      17743   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.153E-04   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.2       TIME AVG. MOMENT       4.34    
 LARGEST RESIDUAL MOMENT            2.761E-02   AT NODE        234   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.401E-02   AT NODE        142   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.892E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      6.643E+03   TIME AVG. FORCE       1.294E+03
 LARGEST RESIDUAL FORCE             -6.50       AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.690E-02   AT NODE      17743   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.397E-05   AT NODE      21236   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.2       TIME AVG. MOMENT       4.34    
 LARGEST RESIDUAL MOMENT           -1.970E-03   AT NODE      19847   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.402E-02   AT NODE         10   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.451E-04   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      6.643E+03   TIME AVG. FORCE       1.294E+03
 LARGEST RESIDUAL FORCE             1.148E-02   AT NODE      22726   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.690E-02   AT NODE      17743   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.215E-06   AT NODE       1888   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      24.2       TIME AVG. MOMENT       4.34    
 LARGEST RESIDUAL MOMENT           -5.612E-05   AT NODE       2936   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.402E-02   AT NODE         10   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.292E-05   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    18 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.928    
     TOTAL ARC LENGTH    63.2    , INCREMENT OF ARC LENGTH    14.6    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.208E+04   TIME AVG. FORCE       1.862E+03
 LARGEST RESIDUAL FORCE             8.147E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.651E-02   AT NODE       4606   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.651E-02   AT NODE       4606   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.5       TIME AVG. MOMENT       5.77    
 LARGEST RESIDUAL MOMENT             101.       AT NODE        296   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.732E-02   AT NODE      22793   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.732E-02   AT NODE      22793   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.017E+03   TIME AVG. FORCE       1.701E+03
 LARGEST RESIDUAL FORCE            -1.069E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.604E-02   AT NODE      18347   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.646E-04   AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.6       TIME AVG. MOMENT       5.78    
 LARGEST RESIDUAL MOMENT             2.17       AT NODE      22348   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.975E-02   AT NODE      22794   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.023E-03   AT NODE        454   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.936E+03   TIME AVG. FORCE       1.696E+03
 LARGEST RESIDUAL FORCE            -2.745E+03   AT NODE         50   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.643E-02   AT NODE      18347   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.459E-03   AT NODE       1711   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.7       TIME AVG. MOMENT       5.78    
 LARGEST RESIDUAL MOMENT            0.263       AT NODE        202   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.054E-02   AT NODE        144   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.136E-02   AT NODE         50   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.959E+03   TIME AVG. FORCE       1.698E+03
 LARGEST RESIDUAL FORCE             -343.       AT NODE         66   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.648E-02   AT NODE      18347   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.497E-04   AT NODE       1577   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.7       TIME AVG. MOMENT       5.78    
 LARGEST RESIDUAL MOMENT           -5.738E-02   AT NODE        215   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.056E-02   AT NODE      22658   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.059E-03   AT NODE       1417   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.957E+03   TIME AVG. FORCE       1.698E+03
 LARGEST RESIDUAL FORCE              6.05       AT NODE      22752   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.647E-02   AT NODE      18347   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.939E-05   AT NODE       1409   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      31.7       TIME AVG. MOMENT       5.78    
 LARGEST RESIDUAL MOMENT           -3.716E-03   AT NODE       2306   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.054E-02   AT NODE      22658   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.371E-04   AT NODE         50   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    19 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.059    
     TOTAL ARC LENGTH    77.8    , INCREMENT OF ARC LENGTH    14.6    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.493E+04   TIME AVG. FORCE       2.359E+03
 LARGEST RESIDUAL FORCE             8.357E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.674E-02   AT NODE       2794   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.674E-02   AT NODE       2794   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      42.1       TIME AVG. MOMENT       7.60    
 LARGEST RESIDUAL MOMENT             379.       AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.159       AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.159       AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.692E+04   TIME AVG. FORCE       2.459E+03
 LARGEST RESIDUAL FORCE            -1.724E+06   AT NODE        108   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.702E-02   AT NODE       5092   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.951E-02   AT NODE       1467   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      55.5       TIME AVG. MOMENT       8.27    
 LARGEST RESIDUAL MOMENT           -1.978E+03   AT NODE        259   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.289       AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.292       AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.075E+04   TIME AVG. FORCE       2.150E+03
 LARGEST RESIDUAL FORCE             1.039E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.688E-02   AT NODE       5694   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.716E-03   AT NODE       2977   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      52.5       TIME AVG. MOMENT       8.12    
 LARGEST RESIDUAL MOMENT            -278.       AT NODE        259   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.277       AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.780E-02   AT NODE       1920   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.052E+04   TIME AVG. FORCE       2.139E+03
 LARGEST RESIDUAL FORCE             5.373E+04   AT NODE      22703   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.761E-02   AT NODE       5996   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.720E-03   AT NODE      19587   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      50.8       TIME AVG. MOMENT       8.03    
 LARGEST RESIDUAL MOMENT             21.9       AT NODE       1162   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.237       AT NODE      22694   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.490E-02   AT NODE       1009   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.056E+04   TIME AVG. FORCE       2.141E+03
 LARGEST RESIDUAL FORCE             4.408E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.849E-02   AT NODE       6448   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.209E-03   AT NODE       3274   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      51.3       TIME AVG. MOMENT       8.06    
 LARGEST RESIDUAL MOMENT            -10.7       AT NODE        853   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.251       AT NODE        107   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.479E-02   AT NODE         99   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      1.066E+04   TIME AVG. FORCE       2.146E+03
 LARGEST RESIDUAL FORCE             5.449E+04   AT NODE      22703   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.240E-02   AT NODE      20883   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.018E-03   AT NODE      19280   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      52.1       TIME AVG. MOMENT       8.10    
 LARGEST RESIDUAL MOMENT             14.9       AT NODE      22598   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.267       AT NODE      22695   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.067E-02   AT NODE      22703   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    20 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.595E+03   TIME AVG. FORCE       2.092E+03
 LARGEST RESIDUAL FORCE             5.297E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.737E-03   AT NODE       2643   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.737E-03   AT NODE       2643   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.1       TIME AVG. MOMENT       7.20    
 LARGEST RESIDUAL MOMENT            -8.82       AT NODE      22505   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.068E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.068E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.640E+03   TIME AVG. FORCE       2.095E+03
 LARGEST RESIDUAL FORCE            -1.675E+03   AT NODE         39   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.439E-03   AT NODE       2643   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.947E-04   AT NODE       1452   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.9       TIME AVG. MOMENT       7.19    
 LARGEST RESIDUAL MOMENT           -0.431       AT NODE      22593   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.233E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.926E-03   AT NODE         39   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.717E+03   TIME AVG. FORCE       2.099E+03
 LARGEST RESIDUAL FORCE            -9.571E+03   AT NODE        108   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.732E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.342E-03   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.6       TIME AVG. MOMENT       7.23    
 LARGEST RESIDUAL MOMENT             1.06       AT NODE        414   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.463E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.122E-02   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.777E+03   TIME AVG. FORCE       2.102E+03
 LARGEST RESIDUAL FORCE            -2.870E+03   AT NODE         59   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.590E-03   AT NODE       5186   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.484E-03   AT NODE       1721   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.9       TIME AVG. MOMENT       7.24    
 LARGEST RESIDUAL MOMENT            0.398       AT NODE        195   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.638E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.160E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.001E+04   TIME AVG. FORCE       2.113E+03
 LARGEST RESIDUAL FORCE            -1.258E+05   AT NODE         60   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.094E-02   AT NODE       2326   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.024E-02   AT NODE       1721   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.9       TIME AVG. MOMENT       7.24    
 LARGEST RESIDUAL MOMENT             40.9       AT NODE        211   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.304E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.768E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    20 STARTS. ATTEMPT NUMBER  3

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.020E+03   TIME AVG. FORCE       2.064E+03
 LARGEST RESIDUAL FORCE             2.147E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.222E-04   AT NODE       2643   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.222E-04   AT NODE       2643   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.9       TIME AVG. MOMENT       7.09    
 LARGEST RESIDUAL MOMENT            6.005E-02   AT NODE      22499   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.561E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.561E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.020E+03   TIME AVG. FORCE       2.064E+03
 LARGEST RESIDUAL FORCE             0.768       AT NODE      22726   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.315E-04   AT NODE       2492   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.888E-05   AT NODE       1435   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    6.754E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      31.9       TIME AVG. MOMENT       7.09    
 LARGEST RESIDUAL MOMENT           -1.172E-03   AT NODE       2148   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.751E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.905E-04   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.020E+03   TIME AVG. FORCE       2.064E+03
 LARGEST RESIDUAL FORCE            -4.656E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.316E-04   AT NODE       2492   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.194E-07   AT NODE      21401   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      31.9       TIME AVG. MOMENT       7.09    
 LARGEST RESIDUAL MOMENT            1.248E-05   AT NODE       2233   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.754E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.126E-06   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    20 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.077    
     TOTAL ARC LENGTH    78.7    , INCREMENT OF ARC LENGTH   0.912    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.080E+03   TIME AVG. FORCE       2.398E+03
 LARGEST RESIDUAL FORCE             2.095E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.467E-04   AT NODE       2341   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.467E-04   AT NODE       2341   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.0       TIME AVG. MOMENT       8.27    
 LARGEST RESIDUAL MOMENT           -6.198E-02   AT NODE        453   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.994E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.994E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.082E+03   TIME AVG. FORCE       2.398E+03
 LARGEST RESIDUAL FORCE             -1.18       AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.618E-04   AT NODE       2190   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.280E-05   AT NODE       1435   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.279E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      32.0       TIME AVG. MOMENT       8.27    
 LARGEST RESIDUAL MOMENT            1.310E-03   AT NODE      20571   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.230E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.356E-04   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.082E+03   TIME AVG. FORCE       2.398E+03
 LARGEST RESIDUAL FORCE            -5.181E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.620E-04   AT NODE       2190   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.470E-07   AT NODE       1620   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      32.0       TIME AVG. MOMENT       8.27    
 LARGEST RESIDUAL MOMENT           -1.424E-05   AT NODE       2147   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.232E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.292E-06   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.37    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    21 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.095    
     TOTAL ARC LENGTH    79.6    , INCREMENT OF ARC LENGTH   0.912    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.158E+03   TIME AVG. FORCE       2.705E+03
 LARGEST RESIDUAL FORCE             4.649E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.246E-04   AT NODE       2039   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.246E-04   AT NODE       2039   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.2       TIME AVG. MOMENT       9.36    
 LARGEST RESIDUAL MOMENT            9.459E-02   AT NODE      22349   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.299E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.299E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.163E+03   TIME AVG. FORCE       2.705E+03
 LARGEST RESIDUAL FORCE             -2.81       AT NODE         59   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.435E-04   AT NODE      20763   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.108E-05   AT NODE      21199   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.483E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      32.2       TIME AVG. MOMENT       9.36    
 LARGEST RESIDUAL MOMENT            2.474E-03   AT NODE      20319   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.660E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.636E-04   AT NODE      22709   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.163E+03   TIME AVG. FORCE       2.705E+03
 LARGEST RESIDUAL FORCE            -1.685E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.451E-04   AT NODE      20763   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.426E-06   AT NODE      21401   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      32.2       TIME AVG. MOMENT       9.36    
 LARGEST RESIDUAL MOMENT           -1.470E-04   AT NODE       2148   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.680E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.242E-05   AT NODE      22760   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.05    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    22 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.122    
     TOTAL ARC LENGTH    81.0    , INCREMENT OF ARC LENGTH    1.37    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.280E+03   TIME AVG. FORCE       2.991E+03
 LARGEST RESIDUAL FORCE             1.027E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.135E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.135E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.6       TIME AVG. MOMENT       10.4    
 LARGEST RESIDUAL MOMENT            0.200       AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.935E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.935E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.285E+03   TIME AVG. FORCE       2.991E+03
 LARGEST RESIDUAL FORCE              6.90       AT NODE      22709   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.083E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.651E-05   AT NODE      21200   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -4.466E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      32.6       TIME AVG. MOMENT       10.4    
 LARGEST RESIDUAL MOMENT           -5.425E-03   AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.683E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.689E-04   AT NODE      22709   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.285E+03   TIME AVG. FORCE       2.991E+03
 LARGEST RESIDUAL FORCE            -1.747E-02   AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.080E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.883E-06   AT NODE      21367   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      32.6       TIME AVG. MOMENT       10.4    
 LARGEST RESIDUAL MOMENT           -1.023E-04   AT NODE      20318   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.654E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.900E-05   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     3.08    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    23 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.162    
     TOTAL ARC LENGTH    83.0    , INCREMENT OF ARC LENGTH    2.05    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.451E+03   TIME AVG. FORCE       3.261E+03
 LARGEST RESIDUAL FORCE             2.277E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.541E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.541E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.2       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT           -0.470       AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.261E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.261E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.468E+03   TIME AVG. FORCE       3.261E+03
 LARGEST RESIDUAL FORCE             -53.1       AT NODE         91   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.459E-03   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.955E-04   AT NODE       1752   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.2       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT            1.068E-02   AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.251E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.580E-03   AT NODE         91   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.471E+03   TIME AVG. FORCE       3.261E+03
 LARGEST RESIDUAL FORCE             -3.12       AT NODE         44   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.473E-03   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.489E-05   AT NODE       1554   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.632E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.2       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT           -2.661E-03   AT NODE       2300   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.275E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.821E-04   AT NODE         44   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -9.517E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     4.62    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    24 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.222    
     TOTAL ARC LENGTH    86.1    , INCREMENT OF ARC LENGTH    3.08    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.729E+03   TIME AVG. FORCE       3.520E+03
 LARGEST RESIDUAL FORCE             4.649E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.197E-03   AT NODE      21216   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.197E-03   AT NODE      21216   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.5       TIME AVG. MOMENT       12.3    
 LARGEST RESIDUAL MOMENT             1.75       AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.092E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.092E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.805E+03   TIME AVG. FORCE       3.523E+03
 LARGEST RESIDUAL FORCE            -1.009E+04   AT NODE         60   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.585E-03   AT NODE       1570   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.872E-03   AT NODE       1872   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      35.2       TIME AVG. MOMENT       12.3    
 LARGEST RESIDUAL MOMENT            -1.04       AT NODE        493   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.143E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.166E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.801E+03   TIME AVG. FORCE       3.523E+03
 LARGEST RESIDUAL FORCE            -1.041E+04   AT NODE         42   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.464E-03   AT NODE       1721   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.059E-03   AT NODE       1855   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      36.0       TIME AVG. MOMENT       12.3    
 LARGEST RESIDUAL MOMENT            -1.36       AT NODE        642   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.231E-02   AT NODE         43   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.192E-02   AT NODE         42   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.709E+03   TIME AVG. FORCE       3.519E+03
 LARGEST RESIDUAL FORCE            -2.422E+04   AT NODE         41   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.103E-02   AT NODE       1704   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.877E-03   AT NODE       2004   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      37.3       TIME AVG. MOMENT       12.4    
 LARGEST RESIDUAL MOMENT            -3.26       AT NODE        641   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.341E-02   AT NODE         42   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.332E-02   AT NODE         40   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    25 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.582E+03   TIME AVG. FORCE       3.514E+03
 LARGEST RESIDUAL FORCE             6.626E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.264E-04   AT NODE      21216   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.264E-04   AT NODE      21216   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       12.2    
 LARGEST RESIDUAL MOMENT           -0.189       AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.867E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.867E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.606E+03   TIME AVG. FORCE       3.515E+03
 LARGEST RESIDUAL FORCE             -295.       AT NODE         60   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.691E-04   AT NODE       1435   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.733E-04   AT NODE       1721   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.8       TIME AVG. MOMENT       12.2    
 LARGEST RESIDUAL MOMENT           -3.403E-02   AT NODE        493   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.252E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.722E-03   AT NODE         61   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.617E+03   TIME AVG. FORCE       3.516E+03
 LARGEST RESIDUAL FORCE             -559.       AT NODE        108   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.448E-03   AT NODE       1467   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.527E-04   AT NODE       1769   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.9       TIME AVG. MOMENT       12.2    
 LARGEST RESIDUAL MOMENT            6.996E-02   AT NODE        565   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.301E-02   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.100E-03   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.606E+03   TIME AVG. FORCE       3.515E+03
 LARGEST RESIDUAL FORCE              256.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.552E-03   AT NODE      21248   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.609E-04   AT NODE      19419   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.9       TIME AVG. MOMENT       12.2    
 LARGEST RESIDUAL MOMENT           -2.367E-02   AT NODE      22499   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.358E-02   AT NODE      22758   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.072E-04   AT NODE      22687   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      9.604E+03   TIME AVG. FORCE       3.515E+03
 LARGEST RESIDUAL FORCE             0.236       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.544E-03   AT NODE      21248   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.096E-05   AT NODE      17010   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      33.8       TIME AVG. MOMENT       12.2    
 LARGEST RESIDUAL MOMENT            1.373E-03   AT NODE        453   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.350E-02   AT NODE      22758   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.024E-05   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    25 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.238    
     TOTAL ARC LENGTH    87.3    , INCREMENT OF ARC LENGTH    1.15    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.625E+03   TIME AVG. FORCE       3.750E+03
 LARGEST RESIDUAL FORCE             3.841E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.757E-03   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.757E-03   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.2       TIME AVG. MOMENT       13.1    
 LARGEST RESIDUAL MOMENT           -0.462       AT NODE        492   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.311E-02   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.311E-02   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.541E+03   TIME AVG. FORCE       3.747E+03
 LARGEST RESIDUAL FORCE             1.608E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.050E-03   AT NODE       3081   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.833E-04   AT NODE      21232   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.1    
 LARGEST RESIDUAL MOMENT           -0.150       AT NODE      22284   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.886E-03   AT NODE         40   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.595E-03   AT NODE      22758   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.586E+03   TIME AVG. FORCE       3.749E+03
 LARGEST RESIDUAL FORCE             5.652E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.607E-04   AT NODE       5529   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.314E-04   AT NODE       2508   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.1    
 LARGEST RESIDUAL MOMENT           -0.133       AT NODE        490   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.434E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.980E-03   AT NODE         42   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 11 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.607E+03   TIME AVG. FORCE       3.749E+03
 LARGEST RESIDUAL FORCE              892.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.441E-04   AT NODE      21081   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.989E-04   AT NODE      19691   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.8       TIME AVG. MOMENT       13.1    
 LARGEST RESIDUAL MOMENT            4.276E-02   AT NODE      22499   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.950E-03   AT NODE      22758   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.706E-03   AT NODE      22761   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      9.612E+03   TIME AVG. FORCE       3.750E+03
 LARGEST RESIDUAL FORCE              45.1       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.737E-04   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.990E-05   AT NODE       4443   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.8       TIME AVG. MOMENT       13.1    
 LARGEST RESIDUAL MOMENT           -9.321E-03   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.078E-03   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.634E-04   AT NODE        118   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      9.612E+03   TIME AVG. FORCE       3.750E+03
 LARGEST RESIDUAL FORCE             1.384E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.732E-04   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.120E-07   AT NODE       5987   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      33.8       TIME AVG. MOMENT       13.1    
 LARGEST RESIDUAL MOMENT           -9.078E-05   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.073E-03   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.553E-06   AT NODE       1427   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    26 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.241    
     TOTAL ARC LENGTH    88.4    , INCREMENT OF ARC LENGTH    1.15    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.563E+03   TIME AVG. FORCE       3.965E+03
 LARGEST RESIDUAL FORCE            -4.774E+03   AT NODE        109   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.937E-03   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.937E-03   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.5       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT            0.665       AT NODE        566   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.495E-02   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.495E-02   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.620E+03   TIME AVG. FORCE       3.967E+03
 LARGEST RESIDUAL FORCE              653.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.650E-03   AT NODE       1920   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.268E-04   AT NODE       1602   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT            7.526E-02   AT NODE        243   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.243E-02   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.799E-03   AT NODE      22742   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.671E+03   TIME AVG. FORCE       3.969E+03
 LARGEST RESIDUAL FORCE             3.034E+03   AT NODE      22758   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.129E-04   AT NODE       4323   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.410E-03   AT NODE      21248   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.1       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT            0.360       AT NODE      22460   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.267E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.190E-02   AT NODE      22758   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.700E+03   TIME AVG. FORCE       3.970E+03
 LARGEST RESIDUAL FORCE              790.       AT NODE      22710   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.723E-04   AT NODE      16859   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.367E-04   AT NODE      20898   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.2       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT           -0.105       AT NODE      22085   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.204E-03   AT NODE      22742   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.083E-03   AT NODE      22710   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      9.580E+03   TIME AVG. FORCE       3.966E+03
 LARGEST RESIDUAL FORCE            -2.507E+04   AT NODE         43   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.070E-03   AT NODE       2023   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.581E-03   AT NODE      20898   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT             3.88       AT NODE        195   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.924E-02   AT NODE         43   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.427E-02   AT NODE         43   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    27 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.616E+03   TIME AVG. FORCE       3.967E+03
 LARGEST RESIDUAL FORCE             -299.       AT NODE        109   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.841E-04   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.841E-04   AT NODE       1904   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT            4.068E-02   AT NODE        566   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.735E-03   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.735E-03   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.600E+03   TIME AVG. FORCE       3.966E+03
 LARGEST RESIDUAL FORCE              47.4       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.793E-04   AT NODE      21097   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.026E-04   AT NODE      21398   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT            1.165E-02   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.691E-03   AT NODE      22758   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.013E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.601E+03   TIME AVG. FORCE       3.966E+03
 LARGEST RESIDUAL FORCE             0.752       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.867E-04   AT NODE      21033   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.581E-06   AT NODE       1855   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.204E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.7       TIME AVG. MOMENT       13.8    
 LARGEST RESIDUAL MOMENT            7.543E-04   AT NODE      22349   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.745E-03   AT NODE      22694   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.544E-05   AT NODE         42   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -3.590E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    27 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.244    
     TOTAL ARC LENGTH    88.7    , INCREMENT OF ARC LENGTH   0.289    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.578E+03   TIME AVG. FORCE       4.167E+03
 LARGEST RESIDUAL FORCE             -354.       AT NODE         44   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.846E-04   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.846E-04   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.5       TIME AVG. MOMENT       14.5    
 LARGEST RESIDUAL MOMENT            4.755E-02   AT NODE        518   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.119E-03   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.119E-03   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.575E+03   TIME AVG. FORCE       4.167E+03
 LARGEST RESIDUAL FORCE              61.4       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.968E-04   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.041E-04   AT NODE       2325   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.5       TIME AVG. MOMENT       14.5    
 LARGEST RESIDUAL MOMENT            1.157E-02   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.536E-03   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.109E-04   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.575E+03   TIME AVG. FORCE       4.167E+03
 LARGEST RESIDUAL FORCE            -0.497       AT NODE        108   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.781E-04   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.876E-05   AT NODE       1769   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.519E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.5       TIME AVG. MOMENT       14.5    
 LARGEST RESIDUAL MOMENT           -1.008E-03   AT NODE       2300   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.384E-03   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.523E-04   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.326E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.433    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    28 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.238    
     TOTAL ARC LENGTH    89.0    , INCREMENT OF ARC LENGTH   0.289    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.535E+03   TIME AVG. FORCE       4.352E+03
 LARGEST RESIDUAL FORCE              639.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.677E-04   AT NODE       1467   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.677E-04   AT NODE       1467   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.3       TIME AVG. MOMENT       15.2    
 LARGEST RESIDUAL MOMENT           -4.136E-02   AT NODE      22349   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.692E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.692E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.539E+03   TIME AVG. FORCE       4.352E+03
 LARGEST RESIDUAL FORCE             -13.0       AT NODE         60   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.320E-04   AT NODE       1435   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.023E-04   AT NODE       1872   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.074E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.3       TIME AVG. MOMENT       15.2    
 LARGEST RESIDUAL MOMENT            4.953E-03   AT NODE      20420   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.447E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.805E-04   AT NODE         61   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.539E+03   TIME AVG. FORCE       4.352E+03
 LARGEST RESIDUAL FORCE             0.444       AT NODE      22758   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.253E-04   AT NODE       1435   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.729E-05   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    5.927E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.3       TIME AVG. MOMENT       15.2    
 LARGEST RESIDUAL MOMENT           -9.794E-04   AT NODE       2300   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.354E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.441E-04   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -2.849E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.649    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    29 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.227    
     TOTAL ARC LENGTH    89.4    , INCREMENT OF ARC LENGTH   0.433    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.485E+03   TIME AVG. FORCE       4.523E+03
 LARGEST RESIDUAL FORCE             1.955E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.713E-04   AT NODE       1586   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.713E-04   AT NODE       1586   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.1       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT           -6.768E-02   AT NODE      22499   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.631E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.631E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.487E+03   TIME AVG. FORCE       4.523E+03
 LARGEST RESIDUAL FORCE             -9.37       AT NODE         91   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.667E-04   AT NODE       1586   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.384E-05   AT NODE       1753   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.020E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.1       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT            4.260E-03   AT NODE       2382   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.383E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.650E-04   AT NODE         91   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.486E+03   TIME AVG. FORCE       4.523E+03
 LARGEST RESIDUAL FORCE            -0.156       AT NODE         44   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.619E-04   AT NODE      21216   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.005E-05   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.675E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      33.1       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT            5.849E-04   AT NODE       2382   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.318E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.549E-05   AT NODE        108   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    1.174E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.974    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    30 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.208    
     TOTAL ARC LENGTH    90.1    , INCREMENT OF ARC LENGTH   0.649    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.409E+03   TIME AVG. FORCE       4.681E+03
 LARGEST RESIDUAL FORCE             4.833E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.895E-04   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.895E-04   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.8       TIME AVG. MOMENT       16.3    
 LARGEST RESIDUAL MOMENT           -0.105       AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.199E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.199E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.417E+03   TIME AVG. FORCE       4.681E+03
 LARGEST RESIDUAL FORCE             -8.21       AT NODE         91   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.974E-04   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.694E-05   AT NODE       1752   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.307E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      32.8       TIME AVG. MOMENT       16.3    
 LARGEST RESIDUAL MOMENT            3.550E-03   AT NODE       2381   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.041E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.221E-04   AT NODE         91   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.418E+03   TIME AVG. FORCE       4.681E+03
 LARGEST RESIDUAL FORCE            -0.108       AT NODE         43   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.926E-04   AT NODE      21065   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.136E-06   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.070E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      32.8       TIME AVG. MOMENT       16.3    
 LARGEST RESIDUAL MOMENT            4.863E-04   AT NODE       2231   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.979E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.126E-05   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    9.761E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.46    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    31 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.180    
     TOTAL ARC LENGTH    91.1    , INCREMENT OF ARC LENGTH   0.974    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.294E+03   TIME AVG. FORCE       4.825E+03
 LARGEST RESIDUAL FORCE             1.132E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.061E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.061E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.4       TIME AVG. MOMENT       16.8    
 LARGEST RESIDUAL MOMENT           -0.228       AT NODE      22505   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.907E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.907E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.302E+03   TIME AVG. FORCE       4.825E+03
 LARGEST RESIDUAL FORCE              10.4       AT NODE      22742   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.109E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.572E-05   AT NODE       1752   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -7.854E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      32.4       TIME AVG. MOMENT       16.8    
 LARGEST RESIDUAL MOMENT           -3.281E-03   AT NODE        147   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.049E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.975E-04   AT NODE         92   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.302E+03   TIME AVG. FORCE       4.825E+03
 LARGEST RESIDUAL FORCE             3.443E-02   AT NODE      22694   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.108E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.469E-06   AT NODE       1618   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      32.4       TIME AVG. MOMENT       16.8    
 LARGEST RESIDUAL MOMENT           -2.709E-04   AT NODE       2149   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.027E-03   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.026E-05   AT NODE        109   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.19    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    32 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.137    
     TOTAL ARC LENGTH    92.5    , INCREMENT OF ARC LENGTH    1.46    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.138E+03   TIME AVG. FORCE       4.956E+03
 LARGEST RESIDUAL FORCE             2.596E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.768E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.768E-03   AT NODE      20914   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.9       TIME AVG. MOMENT       17.3    
 LARGEST RESIDUAL MOMENT           -0.493       AT NODE      22505   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.417E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.417E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.150E+03   TIME AVG. FORCE       4.956E+03
 LARGEST RESIDUAL FORCE             -85.8       AT NODE         59   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.761E-03   AT NODE      20763   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.277E-04   AT NODE      21233   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.9       TIME AVG. MOMENT       17.3    
 LARGEST RESIDUAL MOMENT           -1.589E-02   AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.328E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.008E-03   AT NODE      22743   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.153E+03   TIME AVG. FORCE       4.957E+03
 LARGEST RESIDUAL FORCE             -4.08       AT NODE        110   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.739E-03   AT NODE      20763   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.503E-05   AT NODE      21401   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.139E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      31.9       TIME AVG. MOMENT       17.3    
 LARGEST RESIDUAL MOMENT            2.811E-03   AT NODE      20569   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.303E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.395E-04   AT NODE      22760   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    7.774E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     3.29    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    33 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.073    
     TOTAL ARC LENGTH    94.7    , INCREMENT OF ARC LENGTH    2.19    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.938E+03   TIME AVG. FORCE       5.074E+03
 LARGEST RESIDUAL FORCE             6.034E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.367E-03   AT NODE       2341   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.367E-03   AT NODE       2341   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.2       TIME AVG. MOMENT       17.7    
 LARGEST RESIDUAL MOMENT            -1.08       AT NODE      22505   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.541E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.541E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.957E+03   TIME AVG. FORCE       5.074E+03
 LARGEST RESIDUAL FORCE             -195.       AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.240E-03   AT NODE       3096   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.056E-04   AT NODE       1435   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.2       TIME AVG. MOMENT       17.7    
 LARGEST RESIDUAL MOMENT           -6.595E-02   AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.240E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.017E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.960E+03   TIME AVG. FORCE       5.074E+03
 LARGEST RESIDUAL FORCE              15.0       AT NODE      22762   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.212E-03   AT NODE       3247   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.440E-05   AT NODE       1471   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    6.488E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      31.2       TIME AVG. MOMENT       17.7    
 LARGEST RESIDUAL MOMENT            5.232E-03   AT NODE      20719   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.174E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.441E-04   AT NODE        112   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -6.697E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     4.93    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    34 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.975    
     TOTAL ARC LENGTH    98.0    , INCREMENT OF ARC LENGTH    3.29    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.638E+03   TIME AVG. FORCE       5.176E+03
 LARGEST RESIDUAL FORCE             1.367E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.302E-03   AT NODE      19102   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.302E-03   AT NODE      19102   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       18.0    
 LARGEST RESIDUAL MOMENT            -2.35       AT NODE      22505   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.467E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.467E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.672E+03   TIME AVG. FORCE       5.177E+03
 LARGEST RESIDUAL FORCE             -38.5       AT NODE         65   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.326E-03   AT NODE      19102   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.499E-04   AT NODE      21356   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       18.0    
 LARGEST RESIDUAL MOMENT            3.295E-02   AT NODE      22655   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.415E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.627E-03   AT NODE      21349   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.670E+03   TIME AVG. FORCE       5.177E+03
 LARGEST RESIDUAL FORCE             -16.5       AT NODE        103   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.324E-03   AT NODE      19102   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.163E-05   AT NODE      21394   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       18.0    
 LARGEST RESIDUAL MOMENT           -4.626E-03   AT NODE       2662   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.377E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.826E-04   AT NODE      22753   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    1.239E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.670E+03   TIME AVG. FORCE       5.177E+03
 LARGEST RESIDUAL FORCE            -6.540E-02   AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.325E-03   AT NODE      19102   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.729E-06   AT NODE      21074   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       18.0    
 LARGEST RESIDUAL MOMENT            3.178E-04   AT NODE       2661   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.380E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.535E-05   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     7.40    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    35 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.830    
     TOTAL ARC LENGTH    103.    , INCREMENT OF ARC LENGTH    4.93    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.213E+03   TIME AVG. FORCE       5.261E+03
 LARGEST RESIDUAL FORCE             3.061E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.039E-03   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.039E-03   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      28.9       TIME AVG. MOMENT       18.3    
 LARGEST RESIDUAL MOMENT             5.01       AT NODE        296   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.151E-02   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.151E-02   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.242E+03   TIME AVG. FORCE       5.262E+03
 LARGEST RESIDUAL FORCE              207.       AT NODE      22701   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.067E-03   AT NODE      18951   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.250E-04   AT NODE       1460   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      28.8       TIME AVG. MOMENT       18.3    
 LARGEST RESIDUAL MOMENT            0.110       AT NODE      22581   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.965E-02   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.128E-03   AT NODE        101   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.520E+03   TIME AVG. FORCE       5.270E+03
 LARGEST RESIDUAL FORCE             1.169E+05   AT NODE      22717   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -7.961E-03   AT NODE       1613   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.527E-03   AT NODE      21056   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.5       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT            -39.9       AT NODE      22566   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.024E-02   AT NODE        103   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.450E-02   AT NODE      22716   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.230E+03   TIME AVG. FORCE       5.262E+03
 LARGEST RESIDUAL FORCE             6.300E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.351E-02   AT NODE       1764   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.742E-03   AT NODE       2066   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.0       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT             8.88       AT NODE        254   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.840E-02   AT NODE        103   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.049E-02   AT NODE      21998   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.235E+04   TIME AVG. FORCE       5.654E+03
 LARGEST RESIDUAL FORCE            -1.714E+06   AT NODE         47   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.762E-02   AT NODE       2011   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.457E-02   AT NODE       2010   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      47.6       TIME AVG. MOMENT       18.9    
 LARGEST RESIDUAL MOMENT           -2.008E+03   AT NODE        198   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.379       AT NODE         48   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.286       AT NODE         47   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    36 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.503E+03   TIME AVG. FORCE       5.269E+03
 LARGEST RESIDUAL FORCE             4.321E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.890E-03   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.890E-03   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.7       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT            0.729       AT NODE        297   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.073E-03   AT NODE         68   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.073E-03   AT NODE         68   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.528E+03   TIME AVG. FORCE       5.270E+03
 LARGEST RESIDUAL FORCE             -750.       AT NODE         67   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.298E-03   AT NODE      20303   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.176E-04   AT NODE      21207   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.8       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT           -0.114       AT NODE      22411   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.396E-02   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.072E-03   AT NODE      21385   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.814E+03   TIME AVG. FORCE       5.278E+03
 LARGEST RESIDUAL FORCE            -1.626E+05   AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.829E-03   AT NODE       1577   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.089E-02   AT NODE       1745   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.1       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT             63.1       AT NODE        218   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.430E-02   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.826E-02   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.187E+03   TIME AVG. FORCE       5.261E+03
 LARGEST RESIDUAL FORCE             1.944E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.123E-02   AT NODE       1764   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.559E-03   AT NODE      20736   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.7       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT             12.7       AT NODE        237   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.439E-02   AT NODE         49   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.400E-02   AT NODE      21847   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    36 STARTS. ATTEMPT NUMBER  3

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.660E+03   TIME AVG. FORCE       5.274E+03
 LARGEST RESIDUAL FORCE              374.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.771E-04   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.771E-04   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT           -2.216E-02   AT NODE      22349   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.570E-04   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.570E-04   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.661E+03   TIME AVG. FORCE       5.274E+03
 LARGEST RESIDUAL FORCE             6.560E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.774E-04   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.072E-06   AT NODE      21226   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT           -9.583E-05   AT NODE      22796   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.654E-04   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.062E-06   AT NODE      21385   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -2.733E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.694    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    36 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.822    
     TOTAL ARC LENGTH    103.    , INCREMENT OF ARC LENGTH   0.462    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.643E+03   TIME AVG. FORCE       5.365E+03
 LARGEST RESIDUAL FORCE              832.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.666E-04   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.666E-04   AT NODE      19253   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.1       TIME AVG. MOMENT       18.7    
 LARGEST RESIDUAL MOMENT            3.293E-02   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.162E-03   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.162E-03   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.644E+03   TIME AVG. FORCE       5.365E+03
 LARGEST RESIDUAL FORCE             0.102       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.674E-04   AT NODE      19404   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.975E-06   AT NODE       1746   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -3.657E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      30.1       TIME AVG. MOMENT       18.7    
 LARGEST RESIDUAL MOMENT            2.049E-04   AT NODE          6   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.185E-03   AT NODE      22734   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.459E-05   AT NODE       1417   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.644E+03   TIME AVG. FORCE       5.365E+03
 LARGEST RESIDUAL FORCE            -6.725E-04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.674E-04   AT NODE       3398   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.265E-08   AT NODE       1888   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      30.1       TIME AVG. MOMENT       18.7    
 LARGEST RESIDUAL MOMENT           -1.077E-06   AT NODE      22494   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.185E-03   AT NODE         68   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.591E-08   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.04    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    37 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.810    
     TOTAL ARC LENGTH    104.    , INCREMENT OF ARC LENGTH   0.694    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.614E+03   TIME AVG. FORCE       5.450E+03
 LARGEST RESIDUAL FORCE             1.883E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.028E-04   AT NODE      19404   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.028E-04   AT NODE      19404   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.0       TIME AVG. MOMENT       19.0    
 LARGEST RESIDUAL MOMENT            4.887E-02   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.818E-03   AT NODE      22734   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.818E-03   AT NODE      22734   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.617E+03   TIME AVG. FORCE       5.450E+03
 LARGEST RESIDUAL FORCE             0.167       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.060E-04   AT NODE       3398   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.685E-06   AT NODE       1746   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -8.602E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      30.0       TIME AVG. MOMENT       19.0    
 LARGEST RESIDUAL MOMENT           -4.606E-04   AT NODE        147   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.895E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.885E-05   AT NODE         86   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.617E+03   TIME AVG. FORCE       5.450E+03
 LARGEST RESIDUAL FORCE            -1.284E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.060E-04   AT NODE      19404   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.113E-07   AT NODE       1410   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      30.0       TIME AVG. MOMENT       19.0    
 LARGEST RESIDUAL MOMENT            5.694E-06   AT NODE       2375   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.894E-03   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.050E-06   AT NODE         51   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.56    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    38 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.793    
     TOTAL ARC LENGTH    105.    , INCREMENT OF ARC LENGTH    1.04    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.565E+03   TIME AVG. FORCE       5.530E+03
 LARGEST RESIDUAL FORCE             4.246E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.161E-04   AT NODE      19557   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.161E-04   AT NODE      19557   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.8       TIME AVG. MOMENT       19.3    
 LARGEST RESIDUAL MOMENT            7.879E-02   AT NODE      22645   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.980E-03   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.980E-03   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.567E+03   TIME AVG. FORCE       5.530E+03
 LARGEST RESIDUAL FORCE             -2.50       AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.379E-04   AT NODE      19852   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.260E-05   AT NODE      21056   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.505E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      29.8       TIME AVG. MOMENT       19.3    
 LARGEST RESIDUAL MOMENT            1.905E-03   AT NODE      20140   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.322E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.419E-04   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.567E+03   TIME AVG. FORCE       5.530E+03
 LARGEST RESIDUAL FORCE            -9.398E-03   AT NODE         50   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -6.372E-04   AT NODE       2940   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.174E-06   AT NODE      21393   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      29.8       TIME AVG. MOMENT       19.3    
 LARGEST RESIDUAL MOMENT            1.127E-04   AT NODE      20178   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.307E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.117E-05   AT NODE      22752   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.34    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    39 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.768    
     TOTAL ARC LENGTH    107.    , INCREMENT OF ARC LENGTH    1.56    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.490E+03   TIME AVG. FORCE       5.604E+03
 LARGEST RESIDUAL FORCE             9.481E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.009E-03   AT NODE      20166   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.009E-03   AT NODE      20166   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.6       TIME AVG. MOMENT       19.5    
 LARGEST RESIDUAL MOMENT           -0.166       AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.611E-03   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.611E-03   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.517E+03   TIME AVG. FORCE       5.605E+03
 LARGEST RESIDUAL FORCE             -500.       AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.546E-03   AT NODE       2047   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.074E-04   AT NODE       1728   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.7       TIME AVG. MOMENT       19.5    
 LARGEST RESIDUAL MOMENT           -6.954E-02   AT NODE        365   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.043E-02   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.819E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.491E+03   TIME AVG. FORCE       5.604E+03
 LARGEST RESIDUAL FORCE             -243.       AT NODE         67   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.210E-03   AT NODE      20303   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.011E-04   AT NODE       1577   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.6       TIME AVG. MOMENT       19.5    
 LARGEST RESIDUAL MOMENT            3.533E-02   AT NODE        391   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.073E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.356E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.511E+03   TIME AVG. FORCE       5.605E+03
 LARGEST RESIDUAL FORCE             -95.5       AT NODE         67   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.434E-03   AT NODE       2182   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.629E-04   AT NODE       1745   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.7       TIME AVG. MOMENT       19.5    
 LARGEST RESIDUAL MOMENT           -1.251E-02   AT NODE        365   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.179E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.106E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.504E+03   TIME AVG. FORCE       5.605E+03
 LARGEST RESIDUAL FORCE             -14.1       AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.347E-03   AT NODE      20453   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.856E-05   AT NODE       1595   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      29.6       TIME AVG. MOMENT       19.5    
 LARGEST RESIDUAL MOMENT           -4.546E-03   AT NODE      20141   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.370E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.086E-04   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      8.505E+03   TIME AVG. FORCE       5.605E+03
 LARGEST RESIDUAL FORCE             6.706E-02   AT NODE      22717   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.354E-03   AT NODE       2333   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.380E-06   AT NODE       1745   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      29.6       TIME AVG. MOMENT       19.5    
 LARGEST RESIDUAL MOMENT           -2.554E-04   AT NODE      20026   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.426E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.578E-05   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    40 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.727    
     TOTAL ARC LENGTH    109.    , INCREMENT OF ARC LENGTH    2.34    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.453E+03   TIME AVG. FORCE       5.674E+03
 LARGEST RESIDUAL FORCE             8.800E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.471E-03   AT NODE       2182   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.471E-03   AT NODE       2182   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.5       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT            0.228       AT NODE        392   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.474E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.474E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.423E+03   TIME AVG. FORCE       5.673E+03
 LARGEST RESIDUAL FORCE             -710.       AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -9.421E-04   AT NODE      19102   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.330E-04   AT NODE       1745   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.4       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT            9.019E-02   AT NODE        391   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.757E-03   AT NODE      22716   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.724E-03   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 7 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.510E+03   TIME AVG. FORCE       5.676E+03
 LARGEST RESIDUAL FORCE             6.197E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -9.695E-04   AT NODE      20906   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.573E-04   AT NODE      17139   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.7       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT           -8.468E-02   AT NODE        158   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.053E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.563E-03   AT NODE      22700   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.447E+03   TIME AVG. FORCE       5.674E+03
 LARGEST RESIDUAL FORCE             6.020E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.421E-03   AT NODE       2333   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.086E-04   AT NODE       4455   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.5       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT           -0.104       AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.660E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.018E-03   AT NODE      22702   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.452E+03   TIME AVG. FORCE       5.674E+03
 LARGEST RESIDUAL FORCE             0.944       AT NODE      22735   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.447E-03   AT NODE      20453   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.815E-05   AT NODE      20906   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -4.416E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      29.5       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT           -1.745E-03   AT NODE        453   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.869E-03   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.091E-04   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -4.309E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    41 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.688    
     TOTAL ARC LENGTH    111.    , INCREMENT OF ARC LENGTH    2.34    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.391E+03   TIME AVG. FORCE       5.739E+03
 LARGEST RESIDUAL FORCE             9.229E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.273E-03   AT NODE      20302   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.273E-03   AT NODE      20302   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.3       TIME AVG. MOMENT       20.0    
 LARGEST RESIDUAL MOMENT            0.153       AT NODE      22644   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.302E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.302E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.389E+03   TIME AVG. FORCE       5.739E+03
 LARGEST RESIDUAL FORCE              21.5       AT NODE      22735   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.173E-03   AT NODE       2635   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.257E-04   AT NODE      21073   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.935E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      29.3       TIME AVG. MOMENT       20.0    
 LARGEST RESIDUAL MOMENT            6.067E-03   AT NODE      22565   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.309E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.926E-04   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.392E+03   TIME AVG. FORCE       5.739E+03
 LARGEST RESIDUAL FORCE              2.07       AT NODE      22717   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.208E-03   AT NODE       2484   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.194E-05   AT NODE      21056   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -4.023E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      29.3       TIME AVG. MOMENT       20.0    
 LARGEST RESIDUAL MOMENT            1.400E-03   AT NODE      19990   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.618E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.084E-04   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.392E+03   TIME AVG. FORCE       5.739E+03
 LARGEST RESIDUAL FORCE            -4.035E-03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.207E-03   AT NODE       2484   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.614E-07   AT NODE      21056   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      29.3       TIME AVG. MOMENT       20.0    
 LARGEST RESIDUAL MOMENT            3.132E-05   AT NODE      20026   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.611E-03   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.579E-06   AT NODE      22735   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    42 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.651    
     TOTAL ARC LENGTH    114.    , INCREMENT OF ARC LENGTH    2.34    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.326E+03   TIME AVG. FORCE       5.799E+03
 LARGEST RESIDUAL FORCE             9.385E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.165E-03   AT NODE       2635   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.165E-03   AT NODE       2635   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.1       TIME AVG. MOMENT       20.2    
 LARGEST RESIDUAL MOMENT           -0.154       AT NODE        158   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.108E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.108E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.331E+03   TIME AVG. FORCE       5.799E+03
 LARGEST RESIDUAL FORCE              3.68       AT NODE      22718   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.129E-03   AT NODE       2786   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.167E-05   AT NODE      21057   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.024E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      29.1       TIME AVG. MOMENT       20.2    
 LARGEST RESIDUAL MOMENT            2.133E-03   AT NODE      22585   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.708E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.090E-04   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.330E+03   TIME AVG. FORCE       5.799E+03
 LARGEST RESIDUAL FORCE             0.125       AT NODE      22735   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.137E-03   AT NODE       2786   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.062E-05   AT NODE      20923   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      29.1       TIME AVG. MOMENT       20.2    
 LARGEST RESIDUAL MOMENT            3.306E-04   AT NODE      19990   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.783E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.572E-05   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.174E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     3.51    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    43 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.613    
     TOTAL ARC LENGTH    116.    , INCREMENT OF ARC LENGTH    2.34    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.216E+03   TIME AVG. FORCE       5.854E+03
 LARGEST RESIDUAL FORCE             2.125E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.676E-03   AT NODE      20000   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.676E-03   AT NODE      20000   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      28.8       TIME AVG. MOMENT       20.4    
 LARGEST RESIDUAL MOMENT           -0.335       AT NODE      22506   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.278E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.278E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.220E+03   TIME AVG. FORCE       5.854E+03
 LARGEST RESIDUAL FORCE             -7.45       AT NODE         68   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.631E-03   AT NODE       2953   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.450E-05   AT NODE      21057   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.612E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      28.8       TIME AVG. MOMENT       20.4    
 LARGEST RESIDUAL MOMENT            5.496E-03   AT NODE      22565   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.710E-03   AT NODE         86   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.806E-04   AT NODE      22718   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.220E+03   TIME AVG. FORCE       5.854E+03
 LARGEST RESIDUAL FORCE             0.579       AT NODE      22717   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.648E-03   AT NODE       2937   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.341E-05   AT NODE      20923   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.818E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      28.8       TIME AVG. MOMENT       20.4    
 LARGEST RESIDUAL MOMENT            6.919E-04   AT NODE      19989   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.872E-03   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.627E-04   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -2.048E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     5.27    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    44 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.557    
     TOTAL ARC LENGTH    120.    , INCREMENT OF ARC LENGTH    3.51    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.055E+03   TIME AVG. FORCE       5.903E+03
 LARGEST RESIDUAL FORCE             4.789E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.447E-03   AT NODE       2937   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.447E-03   AT NODE       2937   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      28.3       TIME AVG. MOMENT       20.6    
 LARGEST RESIDUAL MOMENT           -0.730       AT NODE        158   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.134E-02   AT NODE         65   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.134E-02   AT NODE         65   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.062E+03   TIME AVG. FORCE       5.903E+03
 LARGEST RESIDUAL FORCE             -18.7       AT NODE         67   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.383E-03   AT NODE       3088   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.216E-04   AT NODE       1729   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -4.759E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      28.3       TIME AVG. MOMENT       20.6    
 LARGEST RESIDUAL MOMENT            1.590E-02   AT NODE      22585   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.048E-02   AT NODE         65   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.205E-04   AT NODE         67   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.064E+03   TIME AVG. FORCE       5.903E+03
 LARGEST RESIDUAL FORCE             -3.45       AT NODE         85   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.420E-03   AT NODE       3088   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.846E-05   AT NODE       1879   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.077E-05
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      28.3       TIME AVG. MOMENT       20.6    
 LARGEST RESIDUAL MOMENT            1.647E-03   AT NODE      19838   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.086E-02   AT NODE         65   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.967E-04   AT NODE         85   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    4.108E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     7.90    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    45 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.474    
     TOTAL ARC LENGTH    125.    , INCREMENT OF ARC LENGTH    5.27    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.807E+03   TIME AVG. FORCE       5.945E+03
 LARGEST RESIDUAL FORCE             1.073E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.628E-03   AT NODE       3088   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.628E-03   AT NODE       3088   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.6       TIME AVG. MOMENT       20.7    
 LARGEST RESIDUAL MOMENT             1.58       AT NODE        296   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.583E-02   AT NODE         64   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.583E-02   AT NODE         64   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.820E+03   TIME AVG. FORCE       5.945E+03
 LARGEST RESIDUAL FORCE              203.       AT NODE      22735   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.396E-03   AT NODE       3555   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.416E-04   AT NODE      20905   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.5       TIME AVG. MOMENT       20.7    
 LARGEST RESIDUAL MOMENT            9.141E-02   AT NODE      22565   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.327E-02   AT NODE         61   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.024E-03   AT NODE      22717   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.842E+03   TIME AVG. FORCE       5.945E+03
 LARGEST RESIDUAL FORCE              538.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.387E-03   AT NODE      19697   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.128E-04   AT NODE      16384   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.7       TIME AVG. MOMENT       20.7    
 LARGEST RESIDUAL MOMENT           -1.840E-02   AT NODE        453   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.446E-02   AT NODE         64   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.728E-03   AT NODE      21047   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.843E+03   TIME AVG. FORCE       5.945E+03
 LARGEST RESIDUAL FORCE             0.720       AT NODE      22736   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.370E-03   AT NODE       3239   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.673E-05   AT NODE       1897   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      27.6       TIME AVG. MOMENT       20.7    
 LARGEST RESIDUAL MOMENT            6.573E-04   AT NODE       2925   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.429E-02   AT NODE         64   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.806E-04   AT NODE         86   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    6.453E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     11.8    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    46 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.358    
     TOTAL ARC LENGTH    133.    , INCREMENT OF ARC LENGTH    7.90    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.478E+03   TIME AVG. FORCE       5.978E+03
 LARGEST RESIDUAL FORCE             2.391E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.461E-03   AT NODE       3238   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.461E-03   AT NODE       3238   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      26.7       TIME AVG. MOMENT       20.9    
 LARGEST RESIDUAL MOMENT             3.38       AT NODE      22644   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.263E-02   AT NODE         63   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.263E-02   AT NODE         63   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.505E+03   TIME AVG. FORCE       5.979E+03
 LARGEST RESIDUAL FORCE             -10.7       AT NODE         94   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.633E-03   AT NODE       3237   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.029E-04   AT NODE       2809   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.280E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      26.7       TIME AVG. MOMENT       20.9    
 LARGEST RESIDUAL MOMENT            4.775E-02   AT NODE        245   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.332E-02   AT NODE         62   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.331E-04   AT NODE       2818   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.507E+03   TIME AVG. FORCE       5.979E+03
 LARGEST RESIDUAL FORCE              46.4       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.450E-03   AT NODE       3256   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.309E-04   AT NODE       2179   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      26.6       TIME AVG. MOMENT       20.9    
 LARGEST RESIDUAL MOMENT           -4.615E-03   AT NODE        453   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.203E-02   AT NODE         90   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.379E-03   AT NODE         65   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    1.333E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.508E+03   TIME AVG. FORCE       5.979E+03
 LARGEST RESIDUAL FORCE            -1.008E-02   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.450E-03   AT NODE       3256   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.002E-06   AT NODE       1712   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      26.6       TIME AVG. MOMENT       20.9    
 LARGEST RESIDUAL MOMENT           -5.537E-05   AT NODE         61   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.203E-02   AT NODE         90   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.717E-06   AT NODE       1418   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     17.8    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    47 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.177    
     TOTAL ARC LENGTH    145.    , INCREMENT OF ARC LENGTH    11.8    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.019E+03   TIME AVG. FORCE       6.000E+03
 LARGEST RESIDUAL FORCE             5.227E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -8.398E-03   AT NODE       3388   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.398E-03   AT NODE       3388   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.3       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT             6.96       AT NODE      22644   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.331E-02   AT NODE         61   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.331E-02   AT NODE         61   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.993E+03   TIME AVG. FORCE       6.000E+03
 LARGEST RESIDUAL FORCE             -92.6       AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -8.551E-03   AT NODE       3387   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.063E-04   AT NODE       2925   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.3       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT            0.189       AT NODE      22575   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.387E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.181E-03   AT NODE       2177   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.000E+03   TIME AVG. FORCE       6.000E+03
 LARGEST RESIDUAL FORCE             -53.1       AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -8.711E-03   AT NODE       3387   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.997E-04   AT NODE       2492   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.3       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT           -3.955E-03   AT NODE       3537   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.390E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.547E-03   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -3.233E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.999E+03   TIME AVG. FORCE       6.000E+03
 LARGEST RESIDUAL FORCE             7.164E-02   AT NODE      22729   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -8.705E-03   AT NODE       3387   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.557E-06   AT NODE       2044   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      25.3       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT           -1.706E-04   AT NODE       3266   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.388E-02   AT NODE         60   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.640E-05   AT NODE         79   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     26.7    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    48 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.909    
     TOTAL ARC LENGTH    162.    , INCREMENT OF ARC LENGTH    17.8    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.483E+03   TIME AVG. FORCE       6.010E+03
 LARGEST RESIDUAL FORCE             1.089E+06   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.350E-02   AT NODE       3386   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.350E-02   AT NODE       3386   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.7       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT             13.3       AT NODE        295   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.192E-02   AT NODE         59   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.192E-02   AT NODE         59   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.676E+03   TIME AVG. FORCE       6.014E+03
 LARGEST RESIDUAL FORCE             1.824E+05   AT NODE      22726   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.924E-02   AT NODE      20008   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.776E-02   AT NODE      20310   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      26.0       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT            -54.7       AT NODE      20471   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.130       AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.028E-02   AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.059E+03   TIME AVG. FORCE       6.001E+03
 LARGEST RESIDUAL FORCE             6.294E+04   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.161E-02   AT NODE      19706   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.883E-03   AT NODE       4455   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.7       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT             6.43       AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.138       AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.243E-02   AT NODE      16686   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.146E+03   TIME AVG. FORCE       6.003E+03
 LARGEST RESIDUAL FORCE            -6.599E+04   AT NODE         76   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.538E-02   AT NODE       3549   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.605E-02   AT NODE       4455   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.7       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT            -15.9       AT NODE        227   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.192       AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.390E-02   AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      6.124E+03   TIME AVG. FORCE       6.002E+03
 LARGEST RESIDUAL FORCE             5.199E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.563E-02   AT NODE       3700   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.746E-03   AT NODE      16837   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.6       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT             2.03       AT NODE       5361   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.192       AT NODE         76   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.620E-02   AT NODE       4757   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      6.692E+03   TIME AVG. FORCE       6.014E+03
 LARGEST RESIDUAL FORCE             1.339E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.799E-02   AT NODE      18800   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.594E-02   AT NODE      17290   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.2       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT            -19.0       AT NODE      15478   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.224       AT NODE      22726   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.912E-02   AT NODE      15478   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    49 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.702E+03   TIME AVG. FORCE       6.014E+03
 LARGEST RESIDUAL FORCE             1.539E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.063E-03   AT NODE       3386   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.063E-03   AT NODE       3386   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.6       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT             1.90       AT NODE        295   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.949E-02   AT NODE         59   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.949E-02   AT NODE         59   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.721E+03   TIME AVG. FORCE       6.015E+03
 LARGEST RESIDUAL FORCE             -87.6       AT NODE         65   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.895E-03   AT NODE       3688   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.366E-04   AT NODE       2180   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.5       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT            9.043E-02   AT NODE      22562   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.843E-02   AT NODE         57   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.978E-03   AT NODE         65   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.692E+03   TIME AVG. FORCE       6.014E+03
 LARGEST RESIDUAL FORCE             1.678E+03   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.839E-03   AT NODE       3537   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.001E-03   AT NODE      19869   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      24.5       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT            4.756E-02   AT NODE      21840   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.240E-02   AT NODE         58   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.357E-03   AT NODE      22740   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.695E+03   TIME AVG. FORCE       6.014E+03
 LARGEST RESIDUAL FORCE             0.204       AT NODE      22736   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.838E-03   AT NODE       3537   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.418E-05   AT NODE       1878   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      24.5       TIME AVG. MOMENT       21.0    
 LARGEST RESIDUAL MOMENT           -8.945E-04   AT NODE      22707   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.238E-02   AT NODE         58   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.571E-05   AT NODE         66   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    10.00    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    49 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.750    
     TOTAL ARC LENGTH    169.    , INCREMENT OF ARC LENGTH    6.67    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.306E+03   TIME AVG. FORCE       6.020E+03
 LARGEST RESIDUAL FORCE             3.127E+05   AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -8.138E-03   AT NODE       3536   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.138E-03   AT NODE       3536   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.8       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT             3.70       AT NODE      22643   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.107E-02   AT NODE         58   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.107E-02   AT NODE         58   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.318E+03   TIME AVG. FORCE       6.020E+03
 LARGEST RESIDUAL FORCE              265.       AT NODE        151   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -8.130E-03   AT NODE       3535   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.374E-04   AT NODE       2341   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.7       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT           -0.193       AT NODE        215   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.100E-02   AT NODE         56   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.607E-03   AT NODE       2323   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.319E+03   TIME AVG. FORCE       6.020E+03
 LARGEST RESIDUAL FORCE              617.       AT NODE      22741   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -9.359E-03   AT NODE       3535   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.294E-03   AT NODE       3084   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.8       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT            6.654E-02   AT NODE       1154   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.584E-02   AT NODE         57   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.213E-03   AT NODE         91   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    136656     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.44E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.325E+03   TIME AVG. FORCE       6.020E+03
 LARGEST RESIDUAL FORCE             -2.42       AT NODE         87   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -9.325E-03   AT NODE       3535   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.901E-05   AT NODE      20450   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      23.8       TIME AVG. MOMENT       21.1    
 LARGEST RESIDUAL MOMENT           -1.956E-03   AT NODE        241   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.566E-02   AT NODE         57   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.270E-04   AT NODE      22715   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     15.0    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    50 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.543    
     TOTAL ARC LENGTH    179.    , INCREMENT OF ARC LENGTH   10.00    


     THE NUMBER OF INCS ON THE *STEP CARD HAS BEEN COMPLETED - THE STEP ENDS



     ANALYSIS SUMMARY:
     TOTAL OF         50  INCREMENTS
                       7  CUTBACKS IN AUTOMATIC INCREMENTATION
                     204  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     204  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     204  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       0  WARNING MESSAGES DURING USER INPUT PROCESSING
                     141  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                     141  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   139.60    
       SYSTEM TIME (SEC)    =   3.9000    
       TOTAL CPU TIME (SEC) =   143.50    
       WALLCLOCK TIME (SEC) =        255
