1

   Abaqus 2020                                  Date 09-Sep-2025   Time 15:32:29
   For use by UNIVERSITY OF BRITISH COLUMBIA under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC LOAD CONTROL WITH -
          A SUGGESTED INITIAL INCREMENT IN ARC LENGTH           1.000E-02
          AND A TOTAL ARC LENGTH SCALE FACTOR OF                 30.0    
          THE MINIMUM ARC LENGTH INCREMENT ALLOWED IS           1.000E-08
          THE MAXIMUM ARC LENGTH INCREMENT ALLOWED IS           1.000E+36

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

     MODIFIED RIKS METHOD IS USED - AMPLITUDE REFERENCES ARE IGNORED

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

     CHARACTERISTIC ELEMENT LENGTH     2.026E-02

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.324       TIME AVG. FORCE       0.324    
 LARGEST RESIDUAL FORCE             1.007E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.942E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.942E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    5.865E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     7.399E-05   TIME AVG. MOMENT      7.399E-05
 LARGEST RESIDUAL MOMENT           -7.883E-08   AT NODE       2148   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.467E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.467E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.324       TIME AVG. FORCE       0.324    
 LARGEST RESIDUAL FORCE             4.137E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.942E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.989E-13   AT NODE       9546   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.399E-05   TIME AVG. MOMENT      7.399E-05
 LARGEST RESIDUAL MOMENT           -5.193E-09   AT NODE      27500   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.467E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.692E-12   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     1 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.3333E-04
     TOTAL ARC LENGTH   1.000E-02, INCREMENT OF ARC LENGTH   1.000E-02


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.05       TIME AVG. FORCE       0.688    
 LARGEST RESIDUAL FORCE             1.186E-04   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.942E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.942E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    6.979E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.863E-04   TIME AVG. MOMENT      1.301E-04
 LARGEST RESIDUAL MOMENT            7.762E-08   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.467E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.467E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.05       TIME AVG. FORCE       0.688    
 LARGEST RESIDUAL FORCE            -6.008E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.942E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.898E-13   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.863E-04   TIME AVG. MOMENT      1.301E-04
 LARGEST RESIDUAL MOMENT            4.191E-09   AT NODE      26760   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.467E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.643E-12   AT NODE        824   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.500E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     2 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         6.6667E-04
     TOTAL ARC LENGTH   2.000E-02, INCREMENT OF ARC LENGTH   1.000E-02


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.79       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE            -2.311E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.913E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.913E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.360E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     3.206E-04   TIME AVG. MOMENT      1.936E-04
 LARGEST RESIDUAL MOMENT           -1.743E-07   AT NODE       2148   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.200E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.200E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.79       TIME AVG. FORCE        1.05    
 LARGEST RESIDUAL FORCE             7.060E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.913E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.999E-12   AT NODE       9546   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     3.207E-04   TIME AVG. MOMENT      1.936E-04
 LARGEST RESIDUAL MOMENT            6.528E-09   AT NODE       1227   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.200E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.043E-11   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    2.250E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     3 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.1667E-03
     TOTAL ARC LENGTH   3.500E-02, INCREMENT OF ARC LENGTH   1.500E-02


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       2.86       TIME AVG. FORCE        1.50    
 LARGEST RESIDUAL FORCE             5.043E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.369E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.369E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.967E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     5.202E-04   TIME AVG. MOMENT      2.753E-04
 LARGEST RESIDUAL MOMENT            3.909E-07   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.301E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.301E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       2.86       TIME AVG. FORCE        1.50    
 LARGEST RESIDUAL FORCE            -7.459E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.369E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.494E-12   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.202E-04   TIME AVG. MOMENT      2.753E-04
 LARGEST RESIDUAL MOMENT            4.467E-09   AT NODE      24679   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.301E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.347E-11   AT NODE        824   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    3.375E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     4 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.9167E-03
     TOTAL ARC LENGTH   5.750E-02, INCREMENT OF ARC LENGTH   2.250E-02


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       4.40       TIME AVG. FORCE        2.08    
 LARGEST RESIDUAL FORCE             1.114E-03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.553E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.553E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    6.556E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     8.159E-04   TIME AVG. MOMENT      3.834E-04
 LARGEST RESIDUAL MOMENT           -8.797E-07   AT NODE       2148   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.951E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.951E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       4.40       TIME AVG. FORCE        2.08    
 LARGEST RESIDUAL FORCE             3.367E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.553E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.011E-11   AT NODE       9546   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.159E-04   TIME AVG. MOMENT      3.834E-04
 LARGEST RESIDUAL MOMENT           -5.019E-09   AT NODE       2149   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.951E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.282E-11   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    5.063E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     5 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.0417E-03
     TOTAL ARC LENGTH   9.125E-02, INCREMENT OF ARC LENGTH   3.375E-02


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       6.59       TIME AVG. FORCE        2.83    
 LARGEST RESIDUAL FORCE             2.532E-03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.830E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.830E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.490E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.255E-03   TIME AVG. MOMENT      5.287E-04
 LARGEST RESIDUAL MOMENT            1.978E-06   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.427E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.427E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       6.59       TIME AVG. FORCE        2.83    
 LARGEST RESIDUAL FORCE             3.368E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.830E-07   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.276E-11   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.255E-03   TIME AVG. MOMENT      5.287E-04
 LARGEST RESIDUAL MOMENT           -6.541E-09   AT NODE      24679   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.427E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.189E-10   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    7.594E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     6 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         4.7291E-03
     TOTAL ARC LENGTH   0.142    , INCREMENT OF ARC LENGTH   5.063E-02


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       9.77       TIME AVG. FORCE        3.83    
 LARGEST RESIDUAL FORCE             5.695E-03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.474E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.474E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    3.351E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.909E-03   TIME AVG. MOMENT      7.259E-04
 LARGEST RESIDUAL MOMENT           -4.450E-06   AT NODE       2148   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.114E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.114E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       9.77       TIME AVG. FORCE        3.83    
 LARGEST RESIDUAL FORCE             4.179E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.474E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.124E-11   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.909E-03   TIME AVG. MOMENT      7.259E-04
 LARGEST RESIDUAL MOMENT           -5.191E-09   AT NODE      24673   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.114E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.676E-10   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.114    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     7 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         7.2604E-03
     TOTAL ARC LENGTH   0.218    , INCREMENT OF ARC LENGTH   7.594E-02


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       14.4       TIME AVG. FORCE        5.15    
 LARGEST RESIDUAL FORCE             1.280E-02   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.212E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.212E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    7.534E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.880E-03   TIME AVG. MOMENT      9.951E-04
 LARGEST RESIDUAL MOMENT            1.001E-05   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.671E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.671E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       14.4       TIME AVG. FORCE        5.15    
 LARGEST RESIDUAL FORCE             4.022E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.212E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.154E-10   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.880E-03   TIME AVG. MOMENT      9.951E-04
 LARGEST RESIDUAL MOMENT            6.343E-09   AT NODE      27494   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.671E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.027E-10   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.171    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     8 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.1057E-02
     TOTAL ARC LENGTH   0.332    , INCREMENT OF ARC LENGTH   0.114    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       21.2       TIME AVG. FORCE        6.93    
 LARGEST RESIDUAL FORCE             2.882E-02   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.317E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.317E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.696E-11
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     4.326E-03   TIME AVG. MOMENT      1.365E-03
 LARGEST RESIDUAL MOMENT            2.253E-05   AT NODE       2147   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.506E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.506E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       21.2       TIME AVG. FORCE        6.93    
 LARGEST RESIDUAL FORCE            -1.492E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.317E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.600E-10   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.327E-03   TIME AVG. MOMENT      1.365E-03
 LARGEST RESIDUAL MOMENT           -8.093E-09   AT NODE      24674   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.506E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.358E-09   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.256    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     9 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.6752E-02
     TOTAL ARC LENGTH   0.503    , INCREMENT OF ARC LENGTH   0.171    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       31.3       TIME AVG. FORCE        9.37    
 LARGEST RESIDUAL FORCE             6.486E-02   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.976E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.976E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.481E-03   TIME AVG. MOMENT      1.877E-03
 LARGEST RESIDUAL MOMENT           -5.068E-05   AT NODE       2148   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.760E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.760E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       31.3       TIME AVG. FORCE        9.37    
 LARGEST RESIDUAL FORCE             6.330E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.976E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.860E-10   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.482E-03   TIME AVG. MOMENT      1.877E-03
 LARGEST RESIDUAL MOMENT           -5.003E-09   AT NODE      26763   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.760E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.062E-09   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.384    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    10 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         2.5295E-02
     TOTAL ARC LENGTH   0.759    , INCREMENT OF ARC LENGTH   0.256    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       46.2       TIME AVG. FORCE        12.7    
 LARGEST RESIDUAL FORCE             0.146       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.464E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.464E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     9.696E-03   TIME AVG. MOMENT      2.588E-03
 LARGEST RESIDUAL MOMENT            1.140E-04   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.639E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.639E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       46.2       TIME AVG. FORCE        12.7    
 LARGEST RESIDUAL FORCE            -2.256E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.464E-06   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.322E-09   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     9.698E-03   TIME AVG. MOMENT      2.588E-03
 LARGEST RESIDUAL MOMENT           -4.624E-09   AT NODE      24673   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.639E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.910E-09   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.577    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    11 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.8108E-02
     TOTAL ARC LENGTH    1.14    , INCREMENT OF ARC LENGTH   0.384    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       68.3       TIME AVG. FORCE        17.4    
 LARGEST RESIDUAL FORCE             0.328       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.120E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.120E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.450E-02   TIME AVG. MOMENT      3.580E-03
 LARGEST RESIDUAL MOMENT            2.566E-04   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.459E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.459E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       68.3       TIME AVG. FORCE        17.4    
 LARGEST RESIDUAL FORCE            -5.866E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.120E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.988E-09   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.450E-02   TIME AVG. MOMENT      3.581E-03
 LARGEST RESIDUAL MOMENT           -7.846E-09   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.458E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.562E-08   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.865    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    12 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         5.7328E-02
     TOTAL ARC LENGTH    1.72    , INCREMENT OF ARC LENGTH   0.577    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       101.       TIME AVG. FORCE        23.8    
 LARGEST RESIDUAL FORCE             0.739       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.679E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.679E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     2.165E-02   TIME AVG. MOMENT      4.971E-03
 LARGEST RESIDUAL MOMENT           -5.774E-04   AT NODE       2148   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.269E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.269E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       101.       TIME AVG. FORCE        23.8    
 LARGEST RESIDUAL FORCE             4.112E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.679E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.765E-09   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.166E-02   TIME AVG. MOMENT      4.971E-03
 LARGEST RESIDUAL MOMENT           -2.292E-08   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.269E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.540E-08   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.30    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    13 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         8.6156E-02
     TOTAL ARC LENGTH    2.58    , INCREMENT OF ARC LENGTH   0.865    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       150.       TIME AVG. FORCE        32.8    
 LARGEST RESIDUAL FORCE              1.66       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.519E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.519E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     3.237E-02   TIME AVG. MOMENT      6.928E-03
 LARGEST RESIDUAL MOMENT            1.299E-03   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.903E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.903E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       150.       TIME AVG. FORCE        32.8    
 LARGEST RESIDUAL FORCE             1.214E-05   AT NODE      26456   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.519E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.537E-08   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     3.238E-02   TIME AVG. MOMENT      6.929E-03
 LARGEST RESIDUAL MOMENT           -7.818E-08   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.903E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.049E-08   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.95    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    14 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1294    
     TOTAL ARC LENGTH    3.88    , INCREMENT OF ARC LENGTH    1.30    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       222.       TIME AVG. FORCE        45.4    
 LARGEST RESIDUAL FORCE              3.74       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.778E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.778E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     4.837E-02   TIME AVG. MOMENT      9.692E-03
 LARGEST RESIDUAL MOMENT            2.924E-03   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.854E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.854E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       222.       TIME AVG. FORCE        45.4    
 LARGEST RESIDUAL FORCE             4.976E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.778E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.508E-08   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.841E-02   TIME AVG. MOMENT      9.695E-03
 LARGEST RESIDUAL MOMENT           -2.659E-07   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.854E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.840E-07   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.92    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    15 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1942    
     TOTAL ARC LENGTH    5.83    , INCREMENT OF ARC LENGTH    1.95    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       330.       TIME AVG. FORCE        63.2    
 LARGEST RESIDUAL FORCE              8.43       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.666E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.666E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     7.234E-02   TIME AVG. MOMENT      1.361E-02
 LARGEST RESIDUAL MOMENT            6.580E-03   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.280E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.280E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       330.       TIME AVG. FORCE        63.2    
 LARGEST RESIDUAL FORCE             1.379E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.665E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.068E-08   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.242E-02   TIME AVG. MOMENT      1.362E-02
 LARGEST RESIDUAL MOMENT            9.175E-07   AT NODE          8   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.280E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.241E-07   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     4.38    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    16 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2915    
     TOTAL ARC LENGTH    8.75    , INCREMENT OF ARC LENGTH    2.92    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       489.       TIME AVG. FORCE        88.3    
 LARGEST RESIDUAL FORCE              19.0       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.497E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.497E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.108       TIME AVG. MOMENT      1.919E-02
 LARGEST RESIDUAL MOMENT            1.481E-02   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.419E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.419E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       489.       TIME AVG. FORCE        88.2    
 LARGEST RESIDUAL FORCE            -3.790E-04   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.496E-05   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.876E-07   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.109       TIME AVG. MOMENT      1.920E-02
 LARGEST RESIDUAL MOMENT           -3.207E-06   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.418E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.897E-07   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     6.57    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    17 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.4373    
     TOTAL ARC LENGTH    13.1    , INCREMENT OF ARC LENGTH    4.38    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       726.       TIME AVG. FORCE        124.    
 LARGEST RESIDUAL FORCE              42.7       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.274E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.274E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.163       TIME AVG. MOMENT      2.719E-02
 LARGEST RESIDUAL MOMENT            3.334E-02   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.625E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.625E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       726.       TIME AVG. FORCE        124.    
 LARGEST RESIDUAL FORCE            -1.559E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.274E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.439E-07   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.163       TIME AVG. MOMENT      2.722E-02
 LARGEST RESIDUAL MOMENT           -1.145E-05   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.622E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.354E-06   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     9.85    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    18 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.6560    
     TOTAL ARC LENGTH    19.7    , INCREMENT OF ARC LENGTH    6.57    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.077E+03   TIME AVG. FORCE        174.    
 LARGEST RESIDUAL FORCE              96.3       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.910E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.910E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.246       TIME AVG. MOMENT      3.873E-02
 LARGEST RESIDUAL MOMENT            7.504E-02   AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.443E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.443E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.076E+03   TIME AVG. FORCE        174.    
 LARGEST RESIDUAL FORCE            -8.232E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.909E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.079E-06   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.247       TIME AVG. MOMENT      3.880E-02
 LARGEST RESIDUAL MOMENT           -4.210E-05   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.442E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.768E-06   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     14.8    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    19 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.9837    
     TOTAL ARC LENGTH    29.5    , INCREMENT OF ARC LENGTH    9.85    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.596E+03   TIME AVG. FORCE        245.    
 LARGEST RESIDUAL FORCE              217.       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.863E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.863E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.374       TIME AVG. MOMENT      5.555E-02
 LARGEST RESIDUAL MOMENT            0.169       AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.162E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.162E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.593E+03   TIME AVG. FORCE        245.    
 LARGEST RESIDUAL FORCE            -4.967E-02   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.860E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.754E-06   AT NODE       5253   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.377       TIME AVG. MOMENT      5.571E-02
 LARGEST RESIDUAL MOMENT           -1.624E-04   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.160E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.482E-05   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     22.2    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    20 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.475    
     TOTAL ARC LENGTH    44.3    , INCREMENT OF ARC LENGTH    14.8    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.363E+03   TIME AVG. FORCE        346.    
 LARGEST RESIDUAL FORCE              490.       AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.287E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.287E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.575       TIME AVG. MOMENT      8.044E-02
 LARGEST RESIDUAL MOMENT            0.380       AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.237E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.237E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.358E+03   TIME AVG. FORCE        345.    
 LARGEST RESIDUAL FORCE            -0.370       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.280E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.617E-06   AT NODE       5253   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    5.755E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.584       TIME AVG. MOMENT      8.086E-02
 LARGEST RESIDUAL MOMENT           -6.782E-04   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.232E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.113E-05   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.358E+03   TIME AVG. FORCE        345.    
 LARGEST RESIDUAL FORCE             3.056E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.280E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.545E-09   AT NODE      22360   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.584       TIME AVG. MOMENT      8.086E-02
 LARGEST RESIDUAL MOMENT           -9.472E-07   AT NODE          1   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.232E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.535E-09   AT NODE        571   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     33.3    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    21 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.209    
     TOTAL ARC LENGTH    66.5    , INCREMENT OF ARC LENGTH    22.2    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.494E+03   TIME AVG. FORCE        488.    
 LARGEST RESIDUAL FORCE             1.105E+03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.407E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.407E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.906       TIME AVG. MOMENT      0.118    
 LARGEST RESIDUAL MOMENT            0.853       AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.837E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.837E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.481E+03   TIME AVG. FORCE        488.    
 LARGEST RESIDUAL FORCE             -3.80       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.384E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.398E-05   AT NODE       5253   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.934       TIME AVG. MOMENT      0.120    
 LARGEST RESIDUAL MOMENT           -3.247E-03   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.821E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.304E-04   AT NODE        261   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.481E+03   TIME AVG. FORCE        488.    
 LARGEST RESIDUAL FORCE            -8.023E-05   AT NODE          7   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.384E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.044E-08   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.934       TIME AVG. MOMENT      0.120    
 LARGEST RESIDUAL MOMENT           -6.669E-06   AT NODE          1   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.821E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.131E-07   AT NODE        824   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     49.9    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    22 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.303    
     TOTAL ARC LENGTH    99.7    , INCREMENT OF ARC LENGTH    33.3    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.148E+03   TIME AVG. FORCE        691.    
 LARGEST RESIDUAL FORCE             2.474E+03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.525E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.525E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.51       TIME AVG. MOMENT      0.180    
 LARGEST RESIDUAL MOMENT             1.90       AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.189E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.189E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.122E+03   TIME AVG. FORCE        689.    
 LARGEST RESIDUAL FORCE             -68.8       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.419E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.558E-05   AT NODE       5402   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.63       TIME AVG. MOMENT      0.185    
 LARGEST RESIDUAL MOMENT           -2.010E-02   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.106E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.394E-04   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.123E+03   TIME AVG. FORCE        689.    
 LARGEST RESIDUAL FORCE            -6.167E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.420E-04   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.574E-07   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.63       TIME AVG. MOMENT      0.185    
 LARGEST RESIDUAL MOMENT           -1.312E-04   AT NODE        570   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.107E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.899E-06   AT NODE        824   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     74.8    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    23 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.915    
     TOTAL ARC LENGTH    150.    , INCREMENT OF ARC LENGTH    49.9    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.431E+03   TIME AVG. FORCE        970.    
 LARGEST RESIDUAL FORCE             5.266E+03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.381E-03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.381E-03   AT NODE       1226   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.91       TIME AVG. MOMENT      0.299    
 LARGEST RESIDUAL MOMENT             4.01       AT NODE       1225   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.038E-02   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.038E-02   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.210E+03   TIME AVG. FORCE        961.    
 LARGEST RESIDUAL FORCE            -4.160E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.404E-03   AT NODE      14016   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.688E-04   AT NODE      13867   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.79       TIME AVG. MOMENT      0.336    
 LARGEST RESIDUAL MOMENT           -0.204       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.504E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.135E-03   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.241E+03   TIME AVG. FORCE        962.    
 LARGEST RESIDUAL FORCE             -7.51       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.371E-03   AT NODE      14016   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.382E-05   AT NODE       5197   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.76       TIME AVG. MOMENT      0.334    
 LARGEST RESIDUAL MOMENT           -2.674E-02   AT NODE       1104   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.609E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.789E-04   AT NODE       5360   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.242E+03   TIME AVG. FORCE        962.    
 LARGEST RESIDUAL FORCE            -4.012E-02   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.370E-03   AT NODE      14016   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.231E-07   AT NODE       5374   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      3.76       TIME AVG. MOMENT      0.334    
 LARGEST RESIDUAL MOMENT            5.739E-04   AT NODE        748   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.612E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.208E-06   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     112.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    24 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          7.134    
     TOTAL ARC LENGTH    224.    , INCREMENT OF ARC LENGTH    74.8    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.472E+03   TIME AVG. FORCE       1.303E+03
 LARGEST RESIDUAL FORCE            -5.876E+04   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.981E-03   AT NODE      13867   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.981E-03   AT NODE      13867   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.83       TIME AVG. MOMENT      0.634    
 LARGEST RESIDUAL MOMENT            -4.62       AT NODE       2609   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.620E-02   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.620E-02   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.373E+03   TIME AVG. FORCE       1.259E+03
 LARGEST RESIDUAL FORCE            -9.255E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.291E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.861E-03   AT NODE      22062   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.26       TIME AVG. MOMENT      0.611    
 LARGEST RESIDUAL MOMENT            -2.14       AT NODE         12   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.573E-02   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.329E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.522E+03   TIME AVG. FORCE       1.265E+03
 LARGEST RESIDUAL FORCE            -2.608E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.442E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.224E-04   AT NODE      18216   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.87       TIME AVG. MOMENT      0.636    
 LARGEST RESIDUAL MOMENT            0.149       AT NODE        748   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.765E-02   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.964E-03   AT NODE        347   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.524E+03   TIME AVG. FORCE       1.265E+03
 LARGEST RESIDUAL FORCE            -0.582       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.441E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.663E-06   AT NODE      18128   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      7.87       TIME AVG. MOMENT      0.636    
 LARGEST RESIDUAL MOMENT            5.164E-03   AT NODE        185   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.763E-02   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.731E-05   AT NODE       1148   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.524E+03   TIME AVG. FORCE       1.265E+03
 LARGEST RESIDUAL FORCE            -1.819E-03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.441E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.025E-08   AT NODE      18240   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      7.87       TIME AVG. MOMENT      0.636    
 LARGEST RESIDUAL MOMENT            6.616E-06   AT NODE          5   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.763E-02   AT NODE        852   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.491E-07   AT NODE        749   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    25 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          8.402    
     TOTAL ARC LENGTH    337.    , INCREMENT OF ARC LENGTH    112.    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.887E+03   TIME AVG. FORCE       1.558E+03
 LARGEST RESIDUAL FORCE            -2.721E+04   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.053E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.053E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.85       TIME AVG. MOMENT      0.990    
 LARGEST RESIDUAL MOMENT           -0.478       AT NODE       2609   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.032E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.032E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.826E+03   TIME AVG. FORCE       1.556E+03
 LARGEST RESIDUAL FORCE             -329.       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.952E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.857E-04   AT NODE       9546   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.58       TIME AVG. MOMENT      0.979    
 LARGEST RESIDUAL MOMENT           -9.678E-02   AT NODE        748   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.178E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.451E-04   AT NODE        318   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.830E+03   TIME AVG. FORCE       1.556E+03
 LARGEST RESIDUAL FORCE            -0.474       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.954E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.713E-06   AT NODE      18134   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      9.58       TIME AVG. MOMENT      0.980    
 LARGEST RESIDUAL MOMENT           -4.217E-03   AT NODE        570   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.160E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.588E-05   AT NODE      16347   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    26 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          8.674    
     TOTAL ARC LENGTH    449.    , INCREMENT OF ARC LENGTH    112.    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.088E+03   TIME AVG. FORCE       1.835E+03
 LARGEST RESIDUAL FORCE             2.064E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.853E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.853E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      11.1       TIME AVG. MOMENT       1.35    
 LARGEST RESIDUAL MOMENT            0.410       AT NODE      25807   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.461E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.461E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.074E+03   TIME AVG. FORCE       1.834E+03
 LARGEST RESIDUAL FORCE             -83.1       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.784E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.588E-05   AT NODE       9546   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      10.9       TIME AVG. MOMENT       1.35    
 LARGEST RESIDUAL MOMENT            3.757E-02   AT NODE       1104   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.084E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.297E-04   AT NODE      22493   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.076E+03   TIME AVG. FORCE       1.834E+03
 LARGEST RESIDUAL FORCE            -8.429E-02   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.784E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.361E-06   AT NODE      18137   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      10.9       TIME AVG. MOMENT       1.35    
 LARGEST RESIDUAL MOMENT           -1.381E-03   AT NODE        570   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.079E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.003E-06   AT NODE      18228   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     168.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    27 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          8.853    
     TOTAL ARC LENGTH    561.    , INCREMENT OF ARC LENGTH    112.    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.384E+03   TIME AVG. FORCE       2.104E+03
 LARGEST RESIDUAL FORCE             3.986E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.078E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.078E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.8       TIME AVG. MOMENT       1.76    
 LARGEST RESIDUAL MOMENT            0.818       AT NODE      25812   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.014E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.014E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.376E+03   TIME AVG. FORCE       2.104E+03
 LARGEST RESIDUAL FORCE             -136.       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.971E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.065E-04   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.7       TIME AVG. MOMENT       1.75    
 LARGEST RESIDUAL MOMENT            4.377E-02   AT NODE       1104   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.731E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.800E-04   AT NODE       9562   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.379E+03   TIME AVG. FORCE       2.104E+03
 LARGEST RESIDUAL FORCE            -0.147       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.971E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.907E-06   AT NODE      18236   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      12.7       TIME AVG. MOMENT       1.75    
 LARGEST RESIDUAL MOMENT           -2.608E-03   AT NODE        570   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.723E-03   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.477E-06   AT NODE          7   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     253.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    28 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          9.051    
     TOTAL ARC LENGTH    729.    , INCREMENT OF ARC LENGTH    168.    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.775E+03   TIME AVG. FORCE       2.368E+03
 LARGEST RESIDUAL FORCE             7.937E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.808E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.808E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      15.0       TIME AVG. MOMENT       2.21    
 LARGEST RESIDUAL MOMENT             1.59       AT NODE      25817   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.408E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.408E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.803E+03   TIME AVG. FORCE       2.369E+03
 LARGEST RESIDUAL FORCE             -183.       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.651E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.571E-04   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.8       TIME AVG. MOMENT       2.20    
 LARGEST RESIDUAL MOMENT            5.146E-02   AT NODE       1104   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.370E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.739E-04   AT NODE       9563   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.807E+03   TIME AVG. FORCE       2.369E+03
 LARGEST RESIDUAL FORCE             0.331       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.650E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.504E-06   AT NODE       5418   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      14.8       TIME AVG. MOMENT       2.20    
 LARGEST RESIDUAL MOMENT           -4.284E-03   AT NODE        570   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.368E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.064E-05   AT NODE      18135   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     379.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    29 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          9.275    
     TOTAL ARC LENGTH    982.    , INCREMENT OF ARC LENGTH    253.    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.032E+04   TIME AVG. FORCE       2.634E+03
 LARGEST RESIDUAL FORCE             1.644E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.257E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.257E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      17.8       TIME AVG. MOMENT       2.72    
 LARGEST RESIDUAL MOMENT             3.02       AT NODE      25822   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.007E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.007E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.039E+04   TIME AVG. FORCE       2.637E+03
 LARGEST RESIDUAL FORCE             -179.       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.032E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.246E-04   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      17.6       TIME AVG. MOMENT       2.72    
 LARGEST RESIDUAL MOMENT            6.382E-02   AT NODE        570   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.975E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.040E-03   AT NODE       9562   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.039E+04   TIME AVG. FORCE       2.637E+03
 LARGEST RESIDUAL FORCE              1.75       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.026E-03   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.474E-05   AT NODE      13826   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      17.6       TIME AVG. MOMENT       2.72    
 LARGEST RESIDUAL MOMENT           -6.632E-03   AT NODE        185   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.973E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.356E-05   AT NODE        894   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     568.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    30 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          9.546    
     TOTAL ARC LENGTH   1.361E+03, INCREMENT OF ARC LENGTH    379.    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.129E+04   TIME AVG. FORCE       2.916E+03
 LARGEST RESIDUAL FORCE             3.564E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.172E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.172E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      21.5       TIME AVG. MOMENT       3.32    
 LARGEST RESIDUAL MOMENT             5.55       AT NODE      25822   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.925E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.925E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.145E+04   TIME AVG. FORCE       2.921E+03
 LARGEST RESIDUAL FORCE              139.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.141E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.098E-04   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      21.3       TIME AVG. MOMENT       3.31    
 LARGEST RESIDUAL MOMENT           -8.778E-02   AT NODE        422   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.903E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.510E-03   AT NODE       4880   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.146E+04   TIME AVG. FORCE       2.921E+03
 LARGEST RESIDUAL FORCE              7.30       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.139E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.385E-05   AT NODE      13828   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      21.3       TIME AVG. MOMENT       3.31    
 LARGEST RESIDUAL MOMENT           -9.124E-03   AT NODE        185   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.899E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.727E-04   AT NODE        893   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    3.874E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     852.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    31 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          9.896    
     TOTAL ARC LENGTH   1.929E+03, INCREMENT OF ARC LENGTH    568.    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.294E+04   TIME AVG. FORCE       3.234E+03
 LARGEST RESIDUAL FORCE             8.185E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.664E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.664E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      26.7       TIME AVG. MOMENT       4.04    
 LARGEST RESIDUAL MOMENT             11.1       AT NODE      26990   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.318E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.318E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.325E+04   TIME AVG. FORCE       3.244E+03
 LARGEST RESIDUAL FORCE              998.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.624E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.963E-04   AT NODE       4592   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      26.4       TIME AVG. MOMENT       4.04    
 LARGEST RESIDUAL MOMENT            0.161       AT NODE       7311   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.294E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.802E-03   AT NODE       4733   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.324E+04   TIME AVG. FORCE       3.244E+03
 LARGEST RESIDUAL FORCE              72.6       AT NODE        891   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.618E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.918E-04   AT NODE      13978   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      26.4       TIME AVG. MOMENT       4.04    
 LARGEST RESIDUAL MOMENT           -1.153E-02   AT NODE      15913   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.276E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.793E-03   AT NODE        891   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.287E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.324E+04   TIME AVG. FORCE       3.244E+03
 LARGEST RESIDUAL FORCE            -1.704E-02   AT NODE        324   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.618E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.427E-06   AT NODE       5339   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      26.4       TIME AVG. MOMENT       4.04    
 LARGEST RESIDUAL MOMENT            3.245E-04   AT NODE        891   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.276E-02   AT NODE        289   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.720E-05   AT NODE        325   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.278E+03

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    32 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          10.37    
     TOTAL ARC LENGTH   2.781E+03, INCREMENT OF ARC LENGTH    852.    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.554E+04   TIME AVG. FORCE       3.617E+03
 LARGEST RESIDUAL FORCE             2.032E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.365E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.365E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.2       TIME AVG. MOMENT       4.95    
 LARGEST RESIDUAL MOMENT             28.6       AT NODE       7313   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.344E-02   AT NODE        286   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.344E-02   AT NODE        286   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.601E+04   TIME AVG. FORCE       3.631E+03
 LARGEST RESIDUAL FORCE            -4.349E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.319E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.761E-04   AT NODE       4445   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.9       TIME AVG. MOMENT       4.94    
 LARGEST RESIDUAL MOMENT            0.575       AT NODE       7311   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.346E-02   AT NODE        284   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.661E-03   AT NODE       4439   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.595E+04   TIME AVG. FORCE       3.629E+03
 LARGEST RESIDUAL FORCE             1.846E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.308E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.213E-03   AT NODE       5338   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      33.9       TIME AVG. MOMENT       4.94    
 LARGEST RESIDUAL MOMENT            0.294       AT NODE       7273   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.226E-02   AT NODE        285   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.188E-03   AT NODE        327   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.596E+04   TIME AVG. FORCE       3.629E+03
 LARGEST RESIDUAL FORCE             -3.62       AT NODE        322   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.308E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.117E-05   AT NODE       5342   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      33.9       TIME AVG. MOMENT       4.94    
 LARGEST RESIDUAL MOMENT           -9.629E-03   AT NODE        326   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.217E-02   AT NODE        285   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.031E-04   AT NODE        322   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.918E+03

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    33 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          10.99    
     TOTAL ARC LENGTH   4.059E+03, INCREMENT OF ARC LENGTH   1.278E+03


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.991E+04   TIME AVG. FORCE       4.108E+03
 LARGEST RESIDUAL FORCE             5.218E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.380E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.380E-02   AT NODE      18188   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      46.1       TIME AVG. MOMENT       6.15    
 LARGEST RESIDUAL MOMENT             95.1       AT NODE       7269   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.100       AT NODE        330   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.100       AT NODE        330   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.988E+04   TIME AVG. FORCE       4.107E+03
 LARGEST RESIDUAL FORCE             1.463E+03   AT NODE        331   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.334E-02   AT NODE      18186   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.029E-03   AT NODE       4736   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      45.6       TIME AVG. MOMENT       6.14    
 LARGEST RESIDUAL MOMENT             2.79       AT NODE       7267   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.100       AT NODE        330   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.359E-02   AT NODE       4295   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.985E+04   TIME AVG. FORCE       4.106E+03
 LARGEST RESIDUAL FORCE            -6.248E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.328E-02   AT NODE      18185   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.695E-03   AT NODE      14347   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      45.5       TIME AVG. MOMENT       6.13    
 LARGEST RESIDUAL MOMENT            -1.14       AT NODE      15968   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.107       AT NODE        329   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.459E-02   AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.987E+04   TIME AVG. FORCE       4.107E+03
 LARGEST RESIDUAL FORCE             -215.       AT NODE        322   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.326E-02   AT NODE      18185   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.246E-04   AT NODE       5342   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      45.5       TIME AVG. MOMENT       6.13    
 LARGEST RESIDUAL MOMENT           -8.533E-02   AT NODE       7278   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.108       AT NODE        329   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.112E-03   AT NODE        322   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.987E+04   TIME AVG. FORCE       4.107E+03
 LARGEST RESIDUAL FORCE             -68.3       AT NODE        323   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.325E-02   AT NODE      18186   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.162E-04   AT NODE       5490   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      45.4       TIME AVG. MOMENT       6.13    
 LARGEST RESIDUAL MOMENT           -2.352E-02   AT NODE       3111   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.109       AT NODE        328   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.759E-03   AT NODE        323   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -4.849E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      1.987E+04   TIME AVG. FORCE       4.107E+03
 LARGEST RESIDUAL FORCE            -5.464E-02   AT NODE        321   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.325E-02   AT NODE      18186   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.783E-06   AT NODE       5640   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      45.4       TIME AVG. MOMENT       6.13    
 LARGEST RESIDUAL MOMENT            8.042E-04   AT NODE        303   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.109       AT NODE        328   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.791E-05   AT NODE        322   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    34 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          11.79    
     TOTAL ARC LENGTH   5.977E+03, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.402E+04   TIME AVG. FORCE       4.676E+03
 LARGEST RESIDUAL FORCE            -5.561E+06   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.272E-02   AT NODE      18192   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.272E-02   AT NODE      18192   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      58.3       TIME AVG. MOMENT       7.62    
 LARGEST RESIDUAL MOMENT             143.       AT NODE       7273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.115       AT NODE        327   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.115       AT NODE        327   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.378E+04   TIME AVG. FORCE       4.669E+03
 LARGEST RESIDUAL FORCE            -3.718E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.242E-02   AT NODE      18193   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.138E-03   AT NODE       4591   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      57.6       TIME AVG. MOMENT       7.60    
 LARGEST RESIDUAL MOMENT             5.47       AT NODE       7271   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.115       AT NODE        327   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.452E-02   AT NODE       4150   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.385E+04   TIME AVG. FORCE       4.671E+03
 LARGEST RESIDUAL FORCE             1.336E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.242E-02   AT NODE       9110   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.672E-04   AT NODE      18154   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      57.4       TIME AVG. MOMENT       7.60    
 LARGEST RESIDUAL MOMENT            0.216       AT NODE      15985   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.116       AT NODE        326   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.341E-03   AT NODE        820   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.386E+04   TIME AVG. FORCE       4.671E+03
 LARGEST RESIDUAL FORCE             -55.3       AT NODE        321   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.244E-02   AT NODE       9110   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.798E-04   AT NODE       5641   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      57.4       TIME AVG. MOMENT       7.59    
 LARGEST RESIDUAL MOMENT           -3.690E-02   AT NODE       3115   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.117       AT NODE        326   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.594E-03   AT NODE        321   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    2.720E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.385E+04   TIME AVG. FORCE       4.671E+03
 LARGEST RESIDUAL FORCE             0.286       AT NODE        821   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.244E-02   AT NODE       9110   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.136E-05   AT NODE      14494   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      57.4       TIME AVG. MOMENT       7.59    
 LARGEST RESIDUAL MOMENT            2.090E-03   AT NODE      11805   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.117       AT NODE        326   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.124E-04   AT NODE        820   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    35 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          12.60    
     TOTAL ARC LENGTH   7.894E+03, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.812E+04   TIME AVG. FORCE       5.322E+03
 LARGEST RESIDUAL FORCE            -5.668E+06   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.223E-02   AT NODE       9110   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.223E-02   AT NODE       9110   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      70.2       TIME AVG. MOMENT       9.33    
 LARGEST RESIDUAL MOMENT             155.       AT NODE       7275   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.118       AT NODE        325   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.118       AT NODE        325   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.793E+04   TIME AVG. FORCE       5.317E+03
 LARGEST RESIDUAL FORCE             6.969E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.204E-02   AT NODE       8960   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.541E-04   AT NODE       6997   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      69.7       TIME AVG. MOMENT       9.32    
 LARGEST RESIDUAL MOMENT             6.17       AT NODE       7273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        325   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.192E-02   AT NODE       4004   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.792E+04   TIME AVG. FORCE       5.317E+03
 LARGEST RESIDUAL FORCE            -2.466E+03   AT NODE      18160   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.216E-02   AT NODE       8960   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.540E-03   AT NODE      18221   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      69.6       TIME AVG. MOMENT       9.32    
 LARGEST RESIDUAL MOMENT            -1.28       AT NODE      17170   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        325   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.304E-02   AT NODE      18164   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.794E+04   TIME AVG. FORCE       5.318E+03
 LARGEST RESIDUAL FORCE              20.7       AT NODE      18159   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.217E-02   AT NODE       8960   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.157E-04   AT NODE      18164   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      69.6       TIME AVG. MOMENT       9.32    
 LARGEST RESIDUAL MOMENT           -0.116       AT NODE      18215   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.909E-03   AT NODE      18220   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.794E+04   TIME AVG. FORCE       5.318E+03
 LARGEST RESIDUAL FORCE             0.770       AT NODE      18169   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.217E-02   AT NODE       8960   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.501E-05   AT NODE      18157   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      69.6       TIME AVG. MOMENT       9.32    
 LARGEST RESIDUAL MOMENT            5.721E-03   AT NODE      20262   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.704E-04   AT NODE      18166   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    36 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.44    
     TOTAL ARC LENGTH   9.812E+03, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.225E+04   TIME AVG. FORCE       6.045E+03
 LARGEST RESIDUAL FORCE            -5.587E+06   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.228E-02   AT NODE      18157   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.228E-02   AT NODE      18157   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      83.0       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT             160.       AT NODE       7276   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.120       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.120       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.210E+04   TIME AVG. FORCE       6.041E+03
 LARGEST RESIDUAL FORCE             8.480E+03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.211E-02   AT NODE      18158   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.395E-03   AT NODE      18151   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      82.2       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT             7.30       AT NODE       7275   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.120       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.702E-02   AT NODE      18158   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.213E+04   TIME AVG. FORCE       6.042E+03
 LARGEST RESIDUAL FORCE             -341.       AT NODE        277   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.223E-02   AT NODE      18158   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.390E-04   AT NODE      18213   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      82.0       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT            0.209       AT NODE       3158   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.135E-03   AT NODE        277   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.212E+04   TIME AVG. FORCE       6.042E+03
 LARGEST RESIDUAL FORCE              46.8       AT NODE      18213   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.253E-02   AT NODE      18218   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.227E-04   AT NODE      18216   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      82.0       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT           -5.648E-02   AT NODE      20306   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.001E-03   AT NODE      18209   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.213E+04   TIME AVG. FORCE       6.042E+03
 LARGEST RESIDUAL FORCE             0.130       AT NODE      18211   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.255E-02   AT NODE      18218   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.750E-05   AT NODE      18216   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      82.0       TIME AVG. MOMENT       11.3    
 LARGEST RESIDUAL MOMENT           -3.016E-03   AT NODE      20305   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.119       AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.562E-04   AT NODE      18208   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    37 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          14.26    
     TOTAL ARC LENGTH   1.173E+04, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.653E+04   TIME AVG. FORCE       6.844E+03
 LARGEST RESIDUAL FORCE             5.311E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.219E-02   AT NODE      23247   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.219E-02   AT NODE      23247   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      95.4       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT            -151.       AT NODE      15919   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.117       AT NODE        886   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.117       AT NODE        886   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.639E+04   TIME AVG. FORCE       6.841E+03
 LARGEST RESIDUAL FORCE            -8.607E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.251E-02   AT NODE       8796   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.130E-03   AT NODE      18222   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      94.6       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT             8.61       AT NODE       7346   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.117       AT NODE        886   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.236E-02   AT NODE      18215   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.640E+04   TIME AVG. FORCE       6.841E+03
 LARGEST RESIDUAL FORCE            -1.603E+03   AT NODE        278   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.321E-02   AT NODE       8798   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.795E-04   AT NODE       8801   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      94.4       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT            0.801       AT NODE       7322   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.116       AT NODE        886   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.127E-03   AT NODE        278   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.641E+04   TIME AVG. FORCE       6.841E+03
 LARGEST RESIDUAL FORCE             -22.6       AT NODE        279   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.332E-02   AT NODE       8798   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.208E-04   AT NODE       8950   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      94.4       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT            7.133E-02   AT NODE       7319   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.116       AT NODE        255   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.106E-03   AT NODE        279   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.641E+04   TIME AVG. FORCE       6.841E+03
 LARGEST RESIDUAL FORCE             0.607       AT NODE       7321   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.333E-02   AT NODE       8798   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.332E-06   AT NODE       8801   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      94.4       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT            6.529E-03   AT NODE       7319   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.116       AT NODE        255   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.124E-04   AT NODE        279   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    38 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          15.03    
     TOTAL ARC LENGTH   1.365E+04, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.102E+04   TIME AVG. FORCE       7.717E+03
 LARGEST RESIDUAL FORCE             4.841E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.716E-02   AT NODE       8801   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.716E-02   AT NODE       8801   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      107.       TIME AVG. MOMENT       15.9    
 LARGEST RESIDUAL MOMENT             170.       AT NODE       6744   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.117       AT NODE        255   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.117       AT NODE        255   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.055E+04   TIME AVG. FORCE       7.705E+03
 LARGEST RESIDUAL FORCE             3.521E+04   AT NODE       8503   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.730E-02   AT NODE       8950   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.035E-02   AT NODE       9099   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      106.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT             21.8       AT NODE      23996   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.117       AT NODE       7758   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.475E-02   AT NODE       8056   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.018E+04   TIME AVG. FORCE       7.696E+03
 LARGEST RESIDUAL FORCE             5.036E+04   AT NODE        862   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.793E-02   AT NODE      22658   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.223E-02   AT NODE      22360   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      106.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT            -27.3       AT NODE      23105   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.150       AT NODE      24148   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.039E-02   AT NODE      23254   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.030E+04   TIME AVG. FORCE       7.699E+03
 LARGEST RESIDUAL FORCE            -7.900E+03   AT NODE      22062   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.826E-02   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.098E-03   AT NODE      10440   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      106.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT            -7.06       AT NODE      22360   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.148       AT NODE      24148   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.891E-02   AT NODE      22360   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.025E+04   TIME AVG. FORCE       7.698E+03
 LARGEST RESIDUAL FORCE             4.926E+03   AT NODE      21019   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.975E-02   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.680E-03   AT NODE      21466   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      106.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT             1.40       AT NODE       1025   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.166       AT NODE      20870   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.290E-02   AT NODE      20721   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      4.028E+04   TIME AVG. FORCE       7.698E+03
 LARGEST RESIDUAL FORCE              126.       AT NODE      18806   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.958E-02   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.796E-04   AT NODE      18214   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      105.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT           -0.233       AT NODE       1030   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.165       AT NODE      20870   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.921E-03   AT NODE      18207   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      4.027E+04   TIME AVG. FORCE       7.698E+03
 LARGEST RESIDUAL FORCE              128.       AT NODE      18806   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.958E-02   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.909E-04   AT NODE      18213   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      105.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT            0.144       AT NODE      20303   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.165       AT NODE      20870   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.882E-03   AT NODE      18206   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     8


 AVERAGE FORCE                      4.027E+04   TIME AVG. FORCE       7.698E+03
 LARGEST RESIDUAL FORCE              3.65       AT NODE      18806   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.958E-02   AT NODE      22509   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.116E-05   AT NODE      18214   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      105.       TIME AVG. MOMENT       15.8    
 LARGEST RESIDUAL MOMENT            2.377E-02   AT NODE      20303   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.166       AT NODE      20870   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.224E-04   AT NODE      18207   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   8 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  8 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    39 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          15.58    
     TOTAL ARC LENGTH   1.556E+04, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.473E+04   TIME AVG. FORCE       8.624E+03
 LARGEST RESIDUAL FORCE             4.551E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.149E-02   AT NODE      22956   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.149E-02   AT NODE      22956   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      119.       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT             180.       AT NODE       6745   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.104       AT NODE        817   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.104       AT NODE        817   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.440E+04   TIME AVG. FORCE       8.616E+03
 LARGEST RESIDUAL FORCE            -6.708E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.120E-02   AT NODE      23105   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.038E-03   AT NODE        277   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      119.       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT             8.02       AT NODE       7345   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.104       AT NODE        817   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.936E-03   AT NODE        861   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.439E+04   TIME AVG. FORCE       8.616E+03
 LARGEST RESIDUAL FORCE             2.735E+03   AT NODE        882   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.133E-02   AT NODE      23254   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.624E-04   AT NODE       1606   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      119.       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT            -1.61       AT NODE      15923   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.102       AT NODE        887   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.226E-02   AT NODE        882   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.440E+04   TIME AVG. FORCE       8.616E+03
 LARGEST RESIDUAL FORCE              16.4       AT NODE        883   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.133E-02   AT NODE      23254   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.999E-05   AT NODE      15327   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      119.       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT            0.163       AT NODE        882   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.102       AT NODE        887   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.033E-03   AT NODE        883   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.440E+04   TIME AVG. FORCE       8.616E+03
 LARGEST RESIDUAL FORCE              1.01       AT NODE      15625   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.133E-02   AT NODE      23254   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.251E-06   AT NODE      15327   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      119.       TIME AVG. MOMENT       18.4    
 LARGEST RESIDUAL MOMENT            1.054E-02   AT NODE      15920   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.102       AT NODE        887   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.289E-04   AT NODE        883   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    40 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.17    
     TOTAL ARC LENGTH   1.748E+04, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.913E+04   TIME AVG. FORCE       9.604E+03
 LARGEST RESIDUAL FORCE             4.099E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.113E-02   AT NODE      23403   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.113E-02   AT NODE      23403   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      133.       TIME AVG. MOMENT       21.2    
 LARGEST RESIDUAL MOMENT            -224.       AT NODE      15326   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.101       AT NODE        816   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.101       AT NODE        816   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.847E+04   TIME AVG. FORCE       9.588E+03
 LARGEST RESIDUAL FORCE            -4.866E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.087E-02   AT NODE      23552   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.163E-03   AT NODE        841   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      133.       TIME AVG. MOMENT       21.2    
 LARGEST RESIDUAL MOMENT            -8.98       AT NODE      15624   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.101       AT NODE        816   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.068E-02   AT NODE        860   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.840E+04   TIME AVG. FORCE       9.586E+03
 LARGEST RESIDUAL FORCE             3.622E+03   AT NODE        883   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.112E-02   AT NODE      23552   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.242E-03   AT NODE       1606   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      133.       TIME AVG. MOMENT       21.2    
 LARGEST RESIDUAL MOMENT            -2.28       AT NODE      15923   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.107       AT NODE       1435   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.457E-02   AT NODE        883   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.841E+04   TIME AVG. FORCE       9.587E+03
 LARGEST RESIDUAL FORCE              45.0       AT NODE        883   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.112E-02   AT NODE      23552   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.670E-05   AT NODE       6684   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      133.       TIME AVG. MOMENT       21.2    
 LARGEST RESIDUAL MOMENT            0.281       AT NODE        883   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.106       AT NODE       1435   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.755E-03   AT NODE        883   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.841E+04   TIME AVG. FORCE       9.587E+03
 LARGEST RESIDUAL FORCE             -2.97       AT NODE       6982   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.112E-02   AT NODE      23552   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.251E-05   AT NODE       6684   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      133.       TIME AVG. MOMENT       21.2    
 LARGEST RESIDUAL MOMENT            3.116E-02   AT NODE       7277   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.106       AT NODE       1435   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.845E-04   AT NODE        320   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    41 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.55    
     TOTAL ARC LENGTH   1.940E+04, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.365E+04   TIME AVG. FORCE       1.064E+04
 LARGEST RESIDUAL FORCE             3.581E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.108E-02   AT NODE      23701   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.108E-02   AT NODE      23701   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      147.       TIME AVG. MOMENT       24.2    
 LARGEST RESIDUAL MOMENT            -290.       AT NODE      15325   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.120       AT NODE       1435   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.120       AT NODE       1435   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.220E+04   TIME AVG. FORCE       1.060E+04
 LARGEST RESIDUAL FORCE             3.657E+03   AT NODE      15813   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.087E-02   AT NODE      23850   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.375E-03   AT NODE        842   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      147.       TIME AVG. MOMENT       24.2    
 LARGEST RESIDUAL MOMENT            -15.6       AT NODE      15663   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.119       AT NODE       1436   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.314E-02   AT NODE        859   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.213E+04   TIME AVG. FORCE       1.060E+04
 LARGEST RESIDUAL FORCE            -7.232E+03   AT NODE       2207   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.132E-02   AT NODE      23850   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.986E-03   AT NODE       1606   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      147.       TIME AVG. MOMENT       24.2    
 LARGEST RESIDUAL MOMENT             3.61       AT NODE      15946   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.139       AT NODE       2510   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.173E-02   AT NODE       1606   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.213E+04   TIME AVG. FORCE       1.060E+04
 LARGEST RESIDUAL FORCE              131.       AT NODE        884   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.131E-02   AT NODE      23850   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.590E-04   AT NODE       1606   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      147.       TIME AVG. MOMENT       24.2    
 LARGEST RESIDUAL MOMENT            0.686       AT NODE      15797   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.136       AT NODE       2510   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.027E-03   AT NODE        884   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.213E+04   TIME AVG. FORCE       1.060E+04
 LARGEST RESIDUAL FORCE             -22.0       AT NODE       1326   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.130E-02   AT NODE      23850   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.149E-04   AT NODE       1606   DOF  2
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      147.       TIME AVG. MOMENT       24.2    
 LARGEST RESIDUAL MOMENT            7.833E-02   AT NODE       7277   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.135       AT NODE       1436   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.603E-03   AT NODE       1606   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.832E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    42 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.63    
     TOTAL ARC LENGTH   2.132E+04, INCREMENT OF ARC LENGTH   1.918E+03


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.839E+04   TIME AVG. FORCE       1.171E+04
 LARGEST RESIDUAL FORCE             3.000E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.140E-02   AT NODE      24000   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.140E-02   AT NODE      24000   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      162.       TIME AVG. MOMENT       27.4    
 LARGEST RESIDUAL MOMENT            -396.       AT NODE       6448   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.155       AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.155       AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.535E+04   TIME AVG. FORCE       1.164E+04
 LARGEST RESIDUAL FORCE             8.721E+03   AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.125E-02   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.729E-03   AT NODE        843   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      161.       TIME AVG. MOMENT       27.4    
 LARGEST RESIDUAL MOMENT             33.6       AT NODE        861   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.156       AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.763E-02   AT NODE      15813   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.521E+04   TIME AVG. FORCE       1.164E+04
 LARGEST RESIDUAL FORCE            -5.213E+04   AT NODE       2697   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.207E-02   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.408E-03   AT NODE      28618   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      162.       TIME AVG. MOMENT       27.4    
 LARGEST RESIDUAL MOMENT             10.0       AT NODE       2696   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.192       AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.214E-02   AT NODE      28124   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.520E+04   TIME AVG. FORCE       1.164E+04
 LARGEST RESIDUAL FORCE              363.       AT NODE       1327   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.205E-02   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.673E-04   AT NODE       2377   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      162.       TIME AVG. MOMENT       27.4    
 LARGEST RESIDUAL MOMENT             1.95       AT NODE      15958   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.187       AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.402E-03   AT NODE       1438   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.691E+04   TIME AVG. FORCE       1.168E+04
 LARGEST RESIDUAL FORCE            -2.054E+05   AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.538E-02   AT NODE      17730   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.690E-02   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      165.       TIME AVG. MOMENT       27.5    
 LARGEST RESIDUAL MOMENT            -628.       AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.261       AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.196       AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    43 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.287E+04   TIME AVG. FORCE       1.158E+04
 LARGEST RESIDUAL FORCE             1.910E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.849E-03   AT NODE      24000   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.849E-03   AT NODE      24000   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      150.       TIME AVG. MOMENT       27.1    
 LARGEST RESIDUAL MOMENT             24.0       AT NODE      15028   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.877E-02   AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.877E-02   AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.299E+04   TIME AVG. FORCE       1.159E+04
 LARGEST RESIDUAL FORCE             -128.       AT NODE      15796   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.859E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.378E-04   AT NODE      15648   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      150.       TIME AVG. MOMENT       27.1    
 LARGEST RESIDUAL MOMENT            -1.38       AT NODE      15962   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.939E-02   AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.307E-03   AT NODE        282   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.299E+04   TIME AVG. FORCE       1.159E+04
 LARGEST RESIDUAL FORCE             -20.6       AT NODE       7170   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.869E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.846E-05   AT NODE       1606   DOF  2
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      150.       TIME AVG. MOMENT       27.1    
 LARGEST RESIDUAL MOMENT           -0.134       AT NODE      15948   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.013E-02   AT NODE      24935   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.069E-03   AT NODE       1606   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    1.037E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    43 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.58    
     TOTAL ARC LENGTH   2.180E+04, INCREMENT OF ARC LENGTH    479.    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.366E+04   TIME AVG. FORCE       1.254E+04
 LARGEST RESIDUAL FORCE             1.815E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.884E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.884E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      154.       TIME AVG. MOMENT       30.0    
 LARGEST RESIDUAL MOMENT            -27.2       AT NODE      15922   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.146E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.146E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.378E+04   TIME AVG. FORCE       1.254E+04
 LARGEST RESIDUAL FORCE              151.       AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.894E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.462E-04   AT NODE      15797   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      154.       TIME AVG. MOMENT       30.0    
 LARGEST RESIDUAL MOMENT             1.87       AT NODE      15797   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.209E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.341E-03   AT NODE        845   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.379E+04   TIME AVG. FORCE       1.254E+04
 LARGEST RESIDUAL FORCE             -59.9       AT NODE       2679   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.904E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.685E-04   AT NODE      18212   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    6.676E-05
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      154.       TIME AVG. MOMENT       30.0    
 LARGEST RESIDUAL MOMENT            0.167       AT NODE       7316   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.280E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.780E-03   AT NODE       2989   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.379E+04   TIME AVG. FORCE       1.254E+04
 LARGEST RESIDUAL FORCE             0.272       AT NODE      15946   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.903E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.748E-06   AT NODE      18213   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      154.       TIME AVG. MOMENT       30.0    
 LARGEST RESIDUAL MOMENT            4.678E-03   AT NODE        503   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.280E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.399E-05   AT NODE      18802   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     719.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    44 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.50    
     TOTAL ARC LENGTH   2.228E+04, INCREMENT OF ARC LENGTH    479.    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.480E+04   TIME AVG. FORCE       1.348E+04
 LARGEST RESIDUAL FORCE             3.850E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.187E-02   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.187E-02   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       32.9    
 LARGEST RESIDUAL MOMENT            -64.5       AT NODE      15922   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.619E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.619E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.492E+04   TIME AVG. FORCE       1.349E+04
 LARGEST RESIDUAL FORCE              434.       AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.189E-02   AT NODE      24151   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.194E-04   AT NODE      15811   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       32.9    
 LARGEST RESIDUAL MOMENT             6.07       AT NODE      15797   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.703E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.491E-03   AT NODE        858   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.484E+04   TIME AVG. FORCE       1.348E+04
 LARGEST RESIDUAL FORCE            -6.107E+03   AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.359E-02   AT NODE      17582   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.880E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       32.9    
 LARGEST RESIDUAL MOMENT            -4.78       AT NODE      16829   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.906E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.210E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.488E+04   TIME AVG. FORCE       1.349E+04
 LARGEST RESIDUAL FORCE             -53.9       AT NODE      16971   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.369E-02   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.396E-04   AT NODE      18172   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.975E-05
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       32.9    
 LARGEST RESIDUAL MOMENT            0.424       AT NODE      17423   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.904E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.617E-03   AT NODE      17122   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.480E+04   TIME AVG. FORCE       1.348E+04
 LARGEST RESIDUAL FORCE             1.526E+04   AT NODE      17279   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.195E-02   AT NODE      24151   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.389E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       32.9    
 LARGEST RESIDUAL MOMENT             12.8       AT NODE      19548   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.902E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.100E-02   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    45 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.418E+04   TIME AVG. FORCE       1.347E+04
 LARGEST RESIDUAL FORCE             5.509E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.451E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.451E-03   AT NODE       8056   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      156.       TIME AVG. MOMENT       32.8    
 LARGEST RESIDUAL MOMENT            -10.6       AT NODE      15922   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.482E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.482E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.422E+04   TIME AVG. FORCE       1.347E+04
 LARGEST RESIDUAL FORCE              42.2       AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.456E-03   AT NODE      24151   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.921E-05   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    9.785E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      156.       TIME AVG. MOMENT       32.8    
 LARGEST RESIDUAL MOMENT            0.846       AT NODE      15797   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.510E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.113E-03   AT NODE        885   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.422E+04   TIME AVG. FORCE       1.347E+04
 LARGEST RESIDUAL FORCE             -13.4       AT NODE      17124   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.455E-03   AT NODE      24151   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.358E-04   AT NODE      18164   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.297E-05
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      156.       TIME AVG. MOMENT       32.8    
 LARGEST RESIDUAL MOMENT            9.432E-02   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.518E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.468E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.638E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     270.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    45 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.45    
     TOTAL ARC LENGTH   2.246E+04, INCREMENT OF ARC LENGTH    180.    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.475E+04   TIME AVG. FORCE       1.437E+04
 LARGEST RESIDUAL FORCE             1.185E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.845E-03   AT NODE      17732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.845E-03   AT NODE      17732   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       35.6    
 LARGEST RESIDUAL MOMENT            -22.7       AT NODE      15922   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.837E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.837E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.485E+04   TIME AVG. FORCE       1.437E+04
 LARGEST RESIDUAL FORCE              96.8       AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.946E-03   AT NODE      17582   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.439E-04   AT NODE      18960   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       35.6    
 LARGEST RESIDUAL MOMENT            -2.16       AT NODE      15811   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.883E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.964E-03   AT NODE        858   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.472E+04   TIME AVG. FORCE       1.437E+04
 LARGEST RESIDUAL FORCE            -1.334E+04   AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.061E-02   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.198E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       35.6    
 LARGEST RESIDUAL MOMENT            -10.6       AT NODE      16829   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.042E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.784E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.482E+04   TIME AVG. FORCE       1.437E+04
 LARGEST RESIDUAL FORCE             -210.       AT NODE      16970   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.074E-02   AT NODE      17730   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.710E-04   AT NODE      18021   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       35.6    
 LARGEST RESIDUAL MOMENT             1.13       AT NODE      17422   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.098E-02   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.126E-03   AT NODE      17121   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.470E+04   TIME AVG. FORCE       1.437E+04
 LARGEST RESIDUAL FORCE            -2.648E+04   AT NODE      19098   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.713E-03   AT NODE       7909   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.756E-03   AT NODE      18214   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      160.       TIME AVG. MOMENT       35.6    
 LARGEST RESIDUAL MOMENT            -26.3       AT NODE      18803   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.918E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.765E-02   AT NODE      18803   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    46 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.445E+04   TIME AVG. FORCE       1.436E+04
 LARGEST RESIDUAL FORCE             1.700E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.567E-03   AT NODE      17732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.567E-03   AT NODE      17732   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       35.5    
 LARGEST RESIDUAL MOMENT            -4.26       AT NODE      15962   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.439E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.439E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.447E+04   TIME AVG. FORCE       1.436E+04
 LARGEST RESIDUAL FORCE              14.5       AT NODE      15772   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.643E-03   AT NODE      17582   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.601E-05   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -8.183E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       35.5    
 LARGEST RESIDUAL MOMENT            0.321       AT NODE      15797   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.449E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.052E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.446E+04   TIME AVG. FORCE       1.436E+04
 LARGEST RESIDUAL FORCE             -20.3       AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.779E-03   AT NODE      17582   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.635E-04   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       35.5    
 LARGEST RESIDUAL MOMENT            0.120       AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.448E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.827E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.446E+04   TIME AVG. FORCE       1.436E+04
 LARGEST RESIDUAL FORCE             0.344       AT NODE      16094   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.790E-03   AT NODE      17582   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.397E-05   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       35.5    
 LARGEST RESIDUAL MOMENT            1.028E-02   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.448E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.562E-04   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    1.340E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     101.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    46 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.41    
     TOTAL ARC LENGTH   2.252E+04, INCREMENT OF ARC LENGTH    67.4    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.474E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             3.669E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.660E-03   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.660E-03   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT             8.51       AT NODE      15945   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.296E-02   AT NODE      17426   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.296E-02   AT NODE      17426   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.481E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             -29.9       AT NODE      15797   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.750E-03   AT NODE      17880   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.336E-04   AT NODE      18174   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.087E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT           -0.741       AT NODE      15811   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.406E-02   AT NODE      18802   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.535E-03   AT NODE      17422   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.470E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             2.532E+04   AT NODE      17279   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.261E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.697E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT             23.8       AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.170E-02   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.576E-02   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.483E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE              613.       AT NODE      19406   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.307E-03   AT NODE      18164   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.182E-04   AT NODE      18022   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT             3.14       AT NODE      17423   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.034E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.854E-03   AT NODE      17121   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.482E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             2.217E+03   AT NODE      19103   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.617E-03   AT NODE      18462   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.693E-03   AT NODE      18212   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT             1.70       AT NODE      19546   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.214E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.864E-02   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      5.482E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             -138.       AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.193E-03   AT NODE      18462   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.322E-04   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            0.326       AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.213E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.826E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      5.479E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             1.860E+03   AT NODE      19103   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.767E-03   AT NODE       7905   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.593E-03   AT NODE      18212   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT             1.44       AT NODE      19547   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.211E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.769E-02   AT NODE      18950   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     8


 AVERAGE FORCE                      5.481E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             -33.2       AT NODE      17272   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.764E-03   AT NODE       7905   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.005E-04   AT NODE      18027   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            0.155       AT NODE       1022   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.210E-02   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.381E-03   AT NODE      17573   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     9


 AVERAGE FORCE                      5.479E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE            -2.625E+03   AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.744E-03   AT NODE      18645   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.861E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      159.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            -2.06       AT NODE      16829   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.415E-02   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.108E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    47 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.459E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             5.304E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.747E-03   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.747E-03   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            -2.16       AT NODE      15943   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.608E-03   AT NODE      17426   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.608E-03   AT NODE      17426   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.460E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE              6.70       AT NODE      15772   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.817E-03   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.272E-05   AT NODE      18214   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.045E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT           -0.115       AT NODE      15811   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.532E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.322E-04   AT NODE      18803   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.460E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             -114.       AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.153E-03   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.883E-04   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            0.290       AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.393E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.397E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.460E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE              5.02       AT NODE      19253   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.222E-03   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.239E-05   AT NODE      18213   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            5.947E-02   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.484E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.108E-04   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.460E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             -59.9       AT NODE      17273   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.464E-03   AT NODE      17730   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.863E-04   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            0.210       AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.803E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.196E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      5.460E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE              7.01       AT NODE      19253   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.537E-03   AT NODE      17730   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.065E-04   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            7.082E-02   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.912E-02   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.085E-03   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      5.459E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             2.226E+03   AT NODE      17279   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.554E-03   AT NODE       7905   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.831E-03   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            -1.84       AT NODE      16829   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.062E-03   AT NODE      27226   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.953E-02   AT NODE      17574   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    47 STARTS. ATTEMPT NUMBER  3

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.448E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             -61.3       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.638E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.638E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT           -0.210       AT NODE      15943   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.070E-04   AT NODE      17426   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.070E-04   AT NODE      17426   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.448E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE             0.478       AT NODE      15772   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.666E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.346E-06   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.611E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT            5.832E-03   AT NODE       7154   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.431E-04   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.693E-05   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.448E+04   TIME AVG. FORCE       1.522E+04
 LARGEST RESIDUAL FORCE            -6.442E-02   AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.664E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.275E-07   AT NODE      18213   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       38.1    
 LARGEST RESIDUAL MOMENT           -4.672E-04   AT NODE      15963   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.417E-04   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.909E-06   AT NODE        844   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     9.48    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    47 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.41    
     TOTAL ARC LENGTH   2.253E+04, INCREMENT OF ARC LENGTH    6.32    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.450E+04   TIME AVG. FORCE       1.603E+04
 LARGEST RESIDUAL FORCE              124.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.511E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.511E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       40.6    
 LARGEST RESIDUAL MOMENT           -0.315       AT NODE      15943   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.282E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.282E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.450E+04   TIME AVG. FORCE       1.603E+04
 LARGEST RESIDUAL FORCE             0.730       AT NODE      15772   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.558E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.637E-06   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -3.321E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       40.6    
 LARGEST RESIDUAL MOMENT           -8.669E-03   AT NODE      15796   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.345E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.256E-05   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.450E+04   TIME AVG. FORCE       1.603E+04
 LARGEST RESIDUAL FORCE            -9.657E-02   AT NODE      15812   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.558E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.606E-08   AT NODE      15514   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      157.       TIME AVG. MOMENT       40.6    
 LARGEST RESIDUAL MOMENT           -7.093E-04   AT NODE      15963   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.346E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.842E-06   AT NODE        844   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     14.2    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    48 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.40    
     TOTAL ARC LENGTH   2.254E+04, INCREMENT OF ARC LENGTH    9.48    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.454E+04   TIME AVG. FORCE       1.682E+04
 LARGEST RESIDUAL FORCE              260.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.911E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.911E-04   AT NODE      17731   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       43.0    
 LARGEST RESIDUAL MOMENT           -0.472       AT NODE      15943   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.118E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.118E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.454E+04   TIME AVG. FORCE       1.682E+04
 LARGEST RESIDUAL FORCE              1.09       AT NODE      15772   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.037E-04   AT NODE      17880   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.526E-05   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -6.407E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       43.0    
 LARGEST RESIDUAL MOMENT           -1.319E-02   AT NODE      15813   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.288E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.700E-04   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.454E+04   TIME AVG. FORCE       1.682E+04
 LARGEST RESIDUAL FORCE             0.153       AT NODE       7170   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.057E-04   AT NODE      17880   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.452E-06   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       43.0    
 LARGEST RESIDUAL MOMENT            1.778E-03   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.315E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.721E-05   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -3.667E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     21.3    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    49 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.40    
     TOTAL ARC LENGTH   2.255E+04, INCREMENT OF ARC LENGTH    14.2    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.459E+04   TIME AVG. FORCE       1.757E+04
 LARGEST RESIDUAL FORCE             -553.       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.341E-04   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.341E-04   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       45.3    
 LARGEST RESIDUAL MOMENT           -0.706       AT NODE      15943   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.817E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.817E-03   AT NODE      18951   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.458E+04   TIME AVG. FORCE       1.757E+04
 LARGEST RESIDUAL FORCE              1.57       AT NODE      15772   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.715E-04   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.384E-05   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.246E-07
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       45.3    
 LARGEST RESIDUAL MOMENT            3.204E-02   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.307E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.897E-04   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.459E+04   TIME AVG. FORCE       1.757E+04
 LARGEST RESIDUAL FORCE             -1.20       AT NODE      17123   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.040E-04   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.835E-05   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          DISP.    CORRECTION TOO LARGE COMPARED TO DISP.    INCREMENT

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       45.3    
 LARGEST RESIDUAL MOMENT            2.807E-02   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.735E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.281E-04   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    172691     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.87E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.459E+04   TIME AVG. FORCE       1.757E+04
 LARGEST RESIDUAL FORCE             4.052E-02   AT NODE      16094   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.053E-04   AT NODE      17581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.708E-06   AT NODE      18163   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      158.       TIME AVG. MOMENT       45.3    
 LARGEST RESIDUAL MOMENT            1.240E-03   AT NODE       1023   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.753E-03   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.862E-05   AT NODE      17425   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     32.0    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    50 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          16.39    
     TOTAL ARC LENGTH   2.257E+04, INCREMENT OF ARC LENGTH    21.3    


     THE NUMBER OF INCS ON THE *STEP CARD HAS BEEN COMPLETED - THE STEP ENDS



     ANALYSIS SUMMARY:
     TOTAL OF         50  INCREMENTS
                       5  CUTBACKS IN AUTOMATIC INCREMENTATION
                     191  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     191  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     191  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       0  WARNING MESSAGES DURING USER INPUT PROCESSING
                      57  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                      57  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   171.50    
       SYSTEM TIME (SEC)    =   5.8000    
       TOTAL CPU TIME (SEC) =   177.30    
       WALLCLOCK TIME (SEC) =        303
