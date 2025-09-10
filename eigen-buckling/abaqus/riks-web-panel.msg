1

   Abaqus 2020                                  Date 08-Sep-2025   Time 19:38:40
   For use by UNIVERSITY OF BRITISH COLUMBIA under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC LOAD CONTROL WITH -
          A SUGGESTED INITIAL INCREMENT IN ARC LENGTH           1.000E-03
          AND A TOTAL ARC LENGTH SCALE FACTOR OF                 70.0    
          THE MINIMUM ARC LENGTH INCREMENT ALLOWED IS           7.000E-04
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

     CHARACTERISTIC ELEMENT LENGTH     2.002E-02

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.671E-02   TIME AVG. FORCE       1.671E-02
 LARGEST RESIDUAL FORCE             3.961E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.481E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.481E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.107E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     5.322E-06   TIME AVG. MOMENT      5.322E-06
 LARGEST RESIDUAL MOMENT            1.318E-08   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.584E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.584E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.671E-02   TIME AVG. FORCE       1.671E-02
 LARGEST RESIDUAL FORCE             3.391E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.481E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.093E-13   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.323E-06   TIME AVG. MOMENT      5.323E-06
 LARGEST RESIDUAL MOMENT            5.028E-09   AT NODE         85   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.584E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.786E-12   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     1 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.4286E-05
     TOTAL ARC LENGTH   1.000E-03, INCREMENT OF ARC LENGTH   1.000E-03


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.268E-02   TIME AVG. FORCE       3.470E-02
 LARGEST RESIDUAL FORCE             1.549E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.481E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.481E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.623E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.481E-05   TIME AVG. MOMENT      1.007E-05
 LARGEST RESIDUAL MOMENT            1.210E-08   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.584E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.584E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.268E-02   TIME AVG. FORCE       3.470E-02
 LARGEST RESIDUAL FORCE            -1.294E-05   AT NODE      23707   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.481E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.071E-13   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.481E-05   TIME AVG. MOMENT      1.007E-05
 LARGEST RESIDUAL MOMENT           -2.742E-09   AT NODE      18863   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.584E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.746E-12   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.500E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     2 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         2.8571E-05
     TOTAL ARC LENGTH   2.000E-03, INCREMENT OF ARC LENGTH   1.000E-03


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.103E-02   TIME AVG. FORCE       5.348E-02
 LARGEST RESIDUAL FORCE             2.152E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.222E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.222E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.254E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.577E-05   TIME AVG. MOMENT      1.530E-05
 LARGEST RESIDUAL MOMENT            2.795E-08   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.876E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.876E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.103E-02   TIME AVG. FORCE       5.348E-02
 LARGEST RESIDUAL FORCE            -2.907E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.222E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.444E-13   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.577E-05   TIME AVG. MOMENT      1.530E-05
 LARGEST RESIDUAL MOMENT            2.725E-09   AT NODE       6260   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.876E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.985E-12   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    2.250E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     3 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         4.9999E-05
     TOTAL ARC LENGTH   3.500E-03, INCREMENT OF ARC LENGTH   1.500E-03


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.148       TIME AVG. FORCE       7.714E-02
 LARGEST RESIDUAL FORCE            -2.164E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.333E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.333E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.266E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     4.211E-05   TIME AVG. MOMENT      2.200E-05
 LARGEST RESIDUAL MOMENT           -6.319E-08   AT NODE        757   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.031E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.031E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.148       TIME AVG. FORCE       7.713E-02
 LARGEST RESIDUAL FORCE            -2.628E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.333E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.446E-13   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.211E-05   TIME AVG. MOMENT      2.200E-05
 LARGEST RESIDUAL MOMENT           -2.506E-09   AT NODE      17438   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.031E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.858E-12   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    3.375E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     4 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         8.2141E-05
     TOTAL ARC LENGTH   5.750E-03, INCREMENT OF ARC LENGTH   2.250E-03


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.233       TIME AVG. FORCE       0.108    
 LARGEST RESIDUAL FORCE            -7.552E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.999E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.999E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -7.910E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     6.649E-05   TIME AVG. MOMENT      3.090E-05
 LARGEST RESIDUAL MOMENT           -1.415E-07   AT NODE        757   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.547E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.547E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.233       TIME AVG. FORCE       0.108    
 LARGEST RESIDUAL FORCE             1.123E-05   AT NODE      22972   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.999E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.230E-12   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.649E-05   TIME AVG. MOMENT      3.090E-05
 LARGEST RESIDUAL MOMENT           -2.521E-09   AT NODE       6266   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.547E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.998E-11   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    5.063E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     5 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.3035E-04
     TOTAL ARC LENGTH   9.125E-03, INCREMENT OF ARC LENGTH   3.375E-03


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.359       TIME AVG. FORCE       0.150    
 LARGEST RESIDUAL FORCE             4.674E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.499E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.499E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.895E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.028E-04   TIME AVG. MOMENT      4.289E-05
 LARGEST RESIDUAL MOMENT            3.183E-07   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.321E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.321E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.359       TIME AVG. FORCE       0.150    
 LARGEST RESIDUAL FORCE             4.903E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.498E-08   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.765E-12   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.028E-04   TIME AVG. MOMENT      4.289E-05
 LARGEST RESIDUAL MOMENT           -2.383E-09   AT NODE         93   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.321E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.494E-11   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    7.594E-03

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     6 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         2.0267E-04
     TOTAL ARC LENGTH   1.419E-02, INCREMENT OF ARC LENGTH   5.063E-03


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.547       TIME AVG. FORCE       0.207    
 LARGEST RESIDUAL FORCE             8.404E-05   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.125E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.125E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    8.802E-14
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.571E-04   TIME AVG. MOMENT      5.921E-05
 LARGEST RESIDUAL MOMENT            7.161E-07   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.481E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.481E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.547       TIME AVG. FORCE       0.207    
 LARGEST RESIDUAL FORCE            -4.150E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.125E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.221E-12   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.571E-04   TIME AVG. MOMENT      5.921E-05
 LARGEST RESIDUAL MOMENT           -1.987E-09   AT NODE         10   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.481E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.011E-10   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.139E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     7 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.1114E-04
     TOTAL ARC LENGTH   2.178E-02, INCREMENT OF ARC LENGTH   7.594E-03


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.825       TIME AVG. FORCE       0.284    
 LARGEST RESIDUAL FORCE             1.919E-04   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.687E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.687E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.010E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.380E-04   TIME AVG. MOMENT      8.155E-05
 LARGEST RESIDUAL MOMENT            1.611E-06   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.221E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.221E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.825       TIME AVG. FORCE       0.284    
 LARGEST RESIDUAL FORCE            -6.427E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.687E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.400E-11   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.380E-04   TIME AVG. MOMENT      8.155E-05
 LARGEST RESIDUAL MOMENT            1.480E-09   AT NODE      12358   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.220E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.275E-10   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.709E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     8 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         4.7383E-04
     TOTAL ARC LENGTH   3.317E-02, INCREMENT OF ARC LENGTH   1.139E-02


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.23       TIME AVG. FORCE       0.389    
 LARGEST RESIDUAL FORCE             4.303E-04   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.530E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.530E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.507E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     3.586E-04   TIME AVG. MOMENT      1.123E-04
 LARGEST RESIDUAL MOMENT           -3.623E-06   AT NODE        757   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.830E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.830E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.23       TIME AVG. FORCE       0.389    
 LARGEST RESIDUAL FORCE             3.907E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.530E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.150E-11   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     3.587E-04   TIME AVG. MOMENT      1.123E-04
 LARGEST RESIDUAL MOMENT           -1.164E-09   AT NODE      23702   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.830E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.122E-10   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    2.563E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     9 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         7.1784E-04
     TOTAL ARC LENGTH   5.026E-02, INCREMENT OF ARC LENGTH   1.709E-02


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.79       TIME AVG. FORCE       0.529    
 LARGEST RESIDUAL FORCE             9.679E-04   AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.795E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.795E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.014E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     5.387E-04   TIME AVG. MOMENT      1.550E-04
 LARGEST RESIDUAL MOMENT            8.151E-06   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.174E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.174E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.79       TIME AVG. FORCE       0.529    
 LARGEST RESIDUAL FORCE            -3.908E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.795E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.089E-11   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.387E-04   TIME AVG. MOMENT      1.550E-04
 LARGEST RESIDUAL MOMENT            1.645E-09   AT NODE      23707   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.174E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.153E-09   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    3.844E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    10 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.0838E-03
     TOTAL ARC LENGTH   7.589E-02, INCREMENT OF ARC LENGTH   2.563E-02


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       2.54       TIME AVG. FORCE       0.711    
 LARGEST RESIDUAL FORCE             2.175E-03   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.692E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.692E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.278E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     8.074E-04   TIME AVG. MOMENT      2.143E-04
 LARGEST RESIDUAL MOMENT            1.833E-05   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.761E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.761E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       2.54       TIME AVG. FORCE       0.711    
 LARGEST RESIDUAL FORCE             3.030E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.691E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.596E-10   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.074E-04   TIME AVG. MOMENT      2.143E-04
 LARGEST RESIDUAL MOMENT           -1.501E-09   AT NODE      23705   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.761E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.596E-09   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    5.767E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    11 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.6326E-03
     TOTAL ARC LENGTH   0.114    , INCREMENT OF ARC LENGTH   3.844E-02


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       3.68       TIME AVG. FORCE       0.959    
 LARGEST RESIDUAL FORCE             4.895E-03   AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.535E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.535E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.208E-03   TIME AVG. MOMENT      2.971E-04
 LARGEST RESIDUAL MOMENT            4.122E-05   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.641E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.641E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       3.68       TIME AVG. FORCE       0.959    
 LARGEST RESIDUAL FORCE            -5.088E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.533E-07   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.592E-10   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.208E-03   TIME AVG. MOMENT      2.971E-04
 LARGEST RESIDUAL MOMENT            2.626E-09   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.640E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.849E-09   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    8.650E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    12 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         2.4556E-03
     TOTAL ARC LENGTH   0.172    , INCREMENT OF ARC LENGTH   5.767E-02


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       5.38       TIME AVG. FORCE        1.30    
 LARGEST RESIDUAL FORCE             1.100E-02   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.280E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.280E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.806E-03   TIME AVG. MOMENT      4.132E-04
 LARGEST RESIDUAL MOMENT           -9.268E-05   AT NODE        757   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.960E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.960E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       5.38       TIME AVG. FORCE        1.30    
 LARGEST RESIDUAL FORCE            -3.527E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.279E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.086E-10   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.806E-03   TIME AVG. MOMENT      4.132E-04
 LARGEST RESIDUAL MOMENT           -4.974E-09   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.958E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.318E-08   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.130    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    13 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.6894E-03
     TOTAL ARC LENGTH   0.258    , INCREMENT OF ARC LENGTH   8.650E-02


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       7.90       TIME AVG. FORCE        1.77    
 LARGEST RESIDUAL FORCE             2.473E-02   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.918E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.918E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     2.700E-03   TIME AVG. MOMENT      5.766E-04
 LARGEST RESIDUAL MOMENT            2.083E-04   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.936E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.936E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       7.90       TIME AVG. FORCE        1.77    
 LARGEST RESIDUAL FORCE            -4.548E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.918E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.821E-09   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.700E-03   TIME AVG. MOMENT      5.766E-04
 LARGEST RESIDUAL MOMENT            1.549E-08   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.933E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.974E-08   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.195    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    14 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         5.5387E-03
     TOTAL ARC LENGTH   0.388    , INCREMENT OF ARC LENGTH   0.130    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       11.7       TIME AVG. FORCE        2.43    
 LARGEST RESIDUAL FORCE             5.555E-02   AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.875E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.875E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     4.034E-03   TIME AVG. MOMENT      8.071E-04
 LARGEST RESIDUAL MOMENT            4.678E-04   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.894E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.894E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       11.7       TIME AVG. FORCE        2.43    
 LARGEST RESIDUAL FORCE            -3.153E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.873E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.104E-09   AT NODE       6234   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.035E-03   TIME AVG. MOMENT      8.071E-04
 LARGEST RESIDUAL MOMENT            4.716E-08   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.888E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.720E-08   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.292    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    15 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         8.3093E-03
     TOTAL ARC LENGTH   0.583    , INCREMENT OF ARC LENGTH   0.195    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       17.2       TIME AVG. FORCE        3.35    
 LARGEST RESIDUAL FORCE             0.125       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.306E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.306E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.008E-03   TIME AVG. MOMENT      1.132E-03
 LARGEST RESIDUAL MOMENT           -1.050E-03   AT NODE        757   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.332E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.332E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       17.2       TIME AVG. FORCE        3.35    
 LARGEST RESIDUAL FORCE            -1.276E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.302E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.253E-09   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.010E-03   TIME AVG. MOMENT      1.132E-03
 LARGEST RESIDUAL MOMENT           -1.643E-07   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.331E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.521E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.438    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    16 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.2458E-02
     TOTAL ARC LENGTH   0.875    , INCREMENT OF ARC LENGTH   0.292    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       25.5       TIME AVG. FORCE        4.65    
 LARGEST RESIDUAL FORCE             0.279       AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.446E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.446E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.920E-03   TIME AVG. MOMENT      1.590E-03
 LARGEST RESIDUAL MOMENT            2.352E-03   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.993E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.993E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       25.5       TIME AVG. FORCE        4.65    
 LARGEST RESIDUAL FORCE             5.886E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.436E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.089E-08   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     8.924E-03   TIME AVG. MOMENT      1.591E-03
 LARGEST RESIDUAL MOMENT            5.573E-07   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.990E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.454E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.657    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    17 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.8665E-02
     TOTAL ARC LENGTH    1.31    , INCREMENT OF ARC LENGTH   0.438    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       37.7       TIME AVG. FORCE        6.49    
 LARGEST RESIDUAL FORCE             0.625       AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.638E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.638E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.327E-02   TIME AVG. MOMENT      2.240E-03
 LARGEST RESIDUAL MOMENT            5.258E-03   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.979E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.979E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       37.7       TIME AVG. FORCE        6.49    
 LARGEST RESIDUAL FORCE            -8.313E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.614E-06   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.720E-08   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.328E-02   TIME AVG. MOMENT      2.240E-03
 LARGEST RESIDUAL MOMENT            1.906E-06   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.972E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.878E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.985    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    18 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         2.7937E-02
     TOTAL ARC LENGTH    1.97    , INCREMENT OF ARC LENGTH   0.657    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       55.8       TIME AVG. FORCE        9.08    
 LARGEST RESIDUAL FORCE              1.39       AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.439E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.439E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.976E-02   TIME AVG. MOMENT      3.162E-03
 LARGEST RESIDUAL MOMENT            1.172E-02   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.445E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.445E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       55.7       TIME AVG. FORCE        9.08    
 LARGEST RESIDUAL FORCE             1.460E-04   AT NODE        157   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.433E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.069E-07   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.978E-02   TIME AVG. MOMENT      3.163E-03
 LARGEST RESIDUAL MOMENT           -6.558E-06   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.427E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.808E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.48    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    19 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         4.1757E-02
     TOTAL ARC LENGTH    2.95    , INCREMENT OF ARC LENGTH   0.985    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       82.5       TIME AVG. FORCE        12.8    
 LARGEST RESIDUAL FORCE              3.10       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.141E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.141E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     2.949E-02   TIME AVG. MOMENT      4.480E-03
 LARGEST RESIDUAL MOMENT            2.597E-02   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.611E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.611E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       82.4       TIME AVG. FORCE        12.7    
 LARGEST RESIDUAL FORCE             5.205E-04   AT NODE        157   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.129E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.425E-07   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    4.079E-11
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.954E-02   TIME AVG. MOMENT      4.482E-03
 LARGEST RESIDUAL MOMENT            2.274E-05   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.569E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.189E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.70000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       82.4       TIME AVG. FORCE        12.7    
 LARGEST RESIDUAL FORCE             7.248E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.129E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.629E-11   AT NODE       6683   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.954E-02   TIME AVG. MOMENT      4.482E-03
 LARGEST RESIDUAL MOMENT           -2.685E-09   AT NODE       1059   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.569E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.266E-10   AT NODE        751   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.22    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    20 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         6.2286E-02
     TOTAL ARC LENGTH    4.43    , INCREMENT OF ARC LENGTH    1.48    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       122.       TIME AVG. FORCE        17.9    
 LARGEST RESIDUAL FORCE              6.82       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.173E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.173E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     4.410E-02   TIME AVG. MOMENT      6.369E-03
 LARGEST RESIDUAL MOMENT            5.705E-02   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.786E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.786E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       121.       TIME AVG. FORCE        17.9    
 LARGEST RESIDUAL FORCE            -1.881E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.144E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.513E-07   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.522E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     4.421E-02   TIME AVG. MOMENT      6.374E-03
 LARGEST RESIDUAL MOMENT           -7.977E-05   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.688E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.827E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       121.       TIME AVG. FORCE        17.9    
 LARGEST RESIDUAL FORCE             3.754E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.144E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.894E-11   AT NODE      17709   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.421E-02   TIME AVG. MOMENT      6.374E-03
 LARGEST RESIDUAL MOMENT            1.793E-08   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.688E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.349E-09   AT NODE        454   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     3.33    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    21 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         9.2603E-02
     TOTAL ARC LENGTH    6.65    , INCREMENT OF ARC LENGTH    2.22    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       179.       TIME AVG. FORCE        25.2    
 LARGEST RESIDUAL FORCE              14.8       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.668E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.668E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     6.609E-02   TIME AVG. MOMENT      9.088E-03
 LARGEST RESIDUAL MOMENT            0.124       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.437E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.437E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       178.       TIME AVG. FORCE        25.2    
 LARGEST RESIDUAL FORCE            -7.158E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.598E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.252E-06   AT NODE      17260   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -6.055E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     6.632E-02   TIME AVG. MOMENT      9.099E-03
 LARGEST RESIDUAL MOMENT           -2.834E-04   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.414E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.343E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          3

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       178.       TIME AVG. FORCE        25.2    
 LARGEST RESIDUAL FORCE             9.889E-06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.598E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.855E-10   AT NODE       6683   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.632E-02   TIME AVG. MOMENT      9.099E-03
 LARGEST RESIDUAL MOMENT           -1.269E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.414E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.285E-09   AT NODE        751   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     4.99    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    22 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1369    
     TOTAL ARC LENGTH    9.97    , INCREMENT OF ARC LENGTH    3.33    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       260.       TIME AVG. FORCE        35.4    
 LARGEST RESIDUAL FORCE              31.4       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.780E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.780E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     9.931E-02   TIME AVG. MOMENT      1.302E-02
 LARGEST RESIDUAL MOMENT            0.261       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.082E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.082E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       258.       TIME AVG. FORCE        35.3    
 LARGEST RESIDUAL FORCE            -2.917E-02   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.718E-05   AT NODE       5936   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.820E-06   AT NODE      17409   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.619E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.100       TIME AVG. MOMENT      1.304E-02
 LARGEST RESIDUAL MOMENT           -1.018E-03   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.025E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.685E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       258.       TIME AVG. FORCE        35.3    
 LARGEST RESIDUAL FORCE            -3.438E-05   AT NODE        157   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.718E-05   AT NODE       5936   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.606E-09   AT NODE      17710   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.100       TIME AVG. MOMENT      1.304E-02
 LARGEST RESIDUAL MOMENT           -1.009E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.025E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.978E-08   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     7.48    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    23 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2006    
     TOTAL ARC LENGTH    15.0    , INCREMENT OF ARC LENGTH    4.99    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       373.       TIME AVG. FORCE        49.4    
 LARGEST RESIDUAL FORCE              63.8       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.049E-04   AT NODE       5936   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.049E-04   AT NODE       5936   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.150       TIME AVG. MOMENT      1.874E-02
 LARGEST RESIDUAL MOMENT            0.525       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.942E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.942E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       368.       TIME AVG. FORCE        49.1    
 LARGEST RESIDUAL FORCE            -0.130       AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.110E-04   AT NODE       5936   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.138E-06   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.254E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.151       TIME AVG. MOMENT      1.878E-02
 LARGEST RESIDUAL MOMENT           -3.638E-03   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.803E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.390E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       368.       TIME AVG. FORCE        49.1    
 LARGEST RESIDUAL FORCE            -1.707E-04   AT NODE        157   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.110E-04   AT NODE       5936   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.495E-08   AT NODE      17041   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.151       TIME AVG. MOMENT      1.878E-02
 LARGEST RESIDUAL MOMENT           -8.564E-06   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.803E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.730E-07   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     11.2    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    24 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2892    
     TOTAL ARC LENGTH    22.4    , INCREMENT OF ARC LENGTH    7.48    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.90000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       522.       TIME AVG. FORCE        68.1    
 LARGEST RESIDUAL FORCE              119.       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.749E-04   AT NODE      17111   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.749E-04   AT NODE      17111   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.227       TIME AVG. MOMENT      2.710E-02
 LARGEST RESIDUAL MOMENT            0.968       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.968E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.968E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       509.       TIME AVG. FORCE        67.6    
 LARGEST RESIDUAL FORCE             0.728       AT NODE        757   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.869E-04   AT NODE      17111   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.211E-05   AT NODE      17559   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.228       TIME AVG. MOMENT      2.717E-02
 LARGEST RESIDUAL MOMENT           -1.228E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.639E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.296E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       509.       TIME AVG. FORCE        67.6    
 LARGEST RESIDUAL FORCE             1.175E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.869E-04   AT NODE      17111   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.539E-08   AT NODE      16893   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.228       TIME AVG. MOMENT      2.717E-02
 LARGEST RESIDUAL MOMENT           -6.463E-05   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.640E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.531E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     16.8    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    25 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.4049    
     TOTAL ARC LENGTH    33.7    , INCREMENT OF ARC LENGTH    11.2    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       698.       TIME AVG. FORCE        91.8    
 LARGEST RESIDUAL FORCE              313.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.950E-04   AT NODE      17112   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.950E-04   AT NODE      17112   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.342       TIME AVG. MOMENT      3.929E-02
 LARGEST RESIDUAL MOMENT             1.51       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.915E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.915E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       673.       TIME AVG. FORCE        90.9    
 LARGEST RESIDUAL FORCE              3.35       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.134E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.193E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.345       TIME AVG. MOMENT      3.937E-02
 LARGEST RESIDUAL MOMENT           -3.480E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.228E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.867E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       673.       TIME AVG. FORCE        90.9    
 LARGEST RESIDUAL FORCE             5.549E-03   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.134E-04   AT NODE      17112   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.860E-07   AT NODE      16747   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.344       TIME AVG. MOMENT      3.937E-02
 LARGEST RESIDUAL MOMENT           -3.043E-04   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.233E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.415E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       673.       TIME AVG. FORCE        90.9    
 LARGEST RESIDUAL FORCE            -3.092E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.134E-04   AT NODE      17112   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.140E-10   AT NODE       5872   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.344       TIME AVG. MOMENT      3.937E-02
 LARGEST RESIDUAL MOMENT           -1.882E-07   AT NODE      12085   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.233E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.186E-09   AT NODE        455   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     25.3    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    26 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.5407    
     TOTAL ARC LENGTH    50.5    , INCREMENT OF ARC LENGTH    16.8    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       878.       TIME AVG. FORCE        120.    
 LARGEST RESIDUAL FORCE             1.016E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.873E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.873E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.515       TIME AVG. MOMENT      5.697E-02
 LARGEST RESIDUAL MOMENT             1.81       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.307E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.307E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       840.       TIME AVG. FORCE        119.    
 LARGEST RESIDUAL FORCE              8.99       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.048E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.520E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.516       TIME AVG. MOMENT      5.702E-02
 LARGEST RESIDUAL MOMENT            7.273E-02   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.218E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.088E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       841.       TIME AVG. FORCE        119.    
 LARGEST RESIDUAL FORCE             1.315E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.050E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.081E-06   AT NODE      16601   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.516       TIME AVG. MOMENT      5.702E-02
 LARGEST RESIDUAL MOMENT            6.211E-04   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.227E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.294E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       841.       TIME AVG. FORCE        119.    
 LARGEST RESIDUAL FORCE            -6.770E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.050E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.178E-10   AT NODE       5874   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.516       TIME AVG. MOMENT      5.702E-02
 LARGEST RESIDUAL MOMENT           -3.647E-07   AT NODE      11937   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.227E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.434E-08   AT NODE        455   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     37.9    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    27 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.6780    
     TOTAL ARC LENGTH    75.8    , INCREMENT OF ARC LENGTH    25.3    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.025E+03   TIME AVG. FORCE        151.    
 LARGEST RESIDUAL FORCE            -2.846E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.662E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.662E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.770       TIME AVG. MOMENT      8.248E-02
 LARGEST RESIDUAL MOMENT             1.60       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.898E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.898E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       981.       TIME AVG. FORCE        149.    
 LARGEST RESIDUAL FORCE              12.0       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.716E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.970E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.770       TIME AVG. MOMENT      8.247E-02
 LARGEST RESIDUAL MOMENT            9.518E-02   AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.675E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.223E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       981.       TIME AVG. FORCE        149.    
 LARGEST RESIDUAL FORCE            -1.354E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.722E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.540E-06   AT NODE      16603   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.770       TIME AVG. MOMENT      8.247E-02
 LARGEST RESIDUAL MOMENT            4.679E-04   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.679E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.197E-06   AT NODE        454   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                       981.       TIME AVG. FORCE        149.    
 LARGEST RESIDUAL FORCE            -4.233E-05   AT NODE       1059   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.722E-04   AT NODE      17261   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.448E-10   AT NODE      16900   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.770       TIME AVG. MOMENT      8.247E-02
 LARGEST RESIDUAL MOMENT            3.172E-07   AT NODE        454   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.679E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.076E-08   AT NODE        455   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     56.8    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    28 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.7996    
     TOTAL ARC LENGTH    114.    , INCREMENT OF ARC LENGTH    37.9    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.132E+03   TIME AVG. FORCE        183.    
 LARGEST RESIDUAL FORCE            -7.089E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.153E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.153E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.15       TIME AVG. MOMENT      0.119    
 LARGEST RESIDUAL MOMENT             1.18       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.088E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.088E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.098E+03   TIME AVG. FORCE        182.    
 LARGEST RESIDUAL FORCE              8.73       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.146E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.270E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.15       TIME AVG. MOMENT      0.119    
 LARGEST RESIDUAL MOMENT           -8.468E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.053E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.035E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.098E+03   TIME AVG. FORCE        182.    
 LARGEST RESIDUAL FORCE             5.710E-03   AT NODE       1059   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.147E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.649E-07   AT NODE      16747   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.15       TIME AVG. MOMENT      0.119    
 LARGEST RESIDUAL MOMENT           -1.452E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.053E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.168E-06   AT NODE        454   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     85.2    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    29 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.9033    
     TOTAL ARC LENGTH    170.    , INCREMENT OF ARC LENGTH    56.8    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.238E+03   TIME AVG. FORCE        217.    
 LARGEST RESIDUAL FORCE            -1.664E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.707E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.707E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.71       TIME AVG. MOMENT      0.172    
 LARGEST RESIDUAL MOMENT            0.930       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.593E-03   AT NODE        567   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.593E-03   AT NODE        567   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.223E+03   TIME AVG. FORCE        217.    
 LARGEST RESIDUAL FORCE              3.26       AT NODE        756   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.695E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.869E-05   AT NODE        757   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.70       TIME AVG. MOMENT      0.172    
 LARGEST RESIDUAL MOMENT           -5.867E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.601E-03   AT NODE        567   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.917E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.223E+03   TIME AVG. FORCE        217.    
 LARGEST RESIDUAL FORCE            -6.724E-03   AT NODE      12085   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.695E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.872E-07   AT NODE      17815   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.70       TIME AVG. MOMENT      0.172    
 LARGEST RESIDUAL MOMENT           -4.900E-05   AT NODE        751   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.601E-03   AT NODE        567   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.254E-06   AT NODE        455   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     128.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    30 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.002    
     TOTAL ARC LENGTH    256.    , INCREMENT OF ARC LENGTH    85.2    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.386E+03   TIME AVG. FORCE        255.    
 LARGEST RESIDUAL FORCE            -3.823E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.528E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.528E-03   AT NODE      17410   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.54       TIME AVG. MOMENT      0.248    
 LARGEST RESIDUAL MOMENT             1.09       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.421E-03   AT NODE        568   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.421E-03   AT NODE        568   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.400E+03   TIME AVG. FORCE        255.    
 LARGEST RESIDUAL FORCE             -5.99       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.515E-03   AT NODE       6384   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.332E-05   AT NODE       4708   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.54       TIME AVG. MOMENT      0.248    
 LARGEST RESIDUAL MOMENT           -2.645E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.474E-03   AT NODE        271   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.095E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.400E+03   TIME AVG. FORCE        255.    
 LARGEST RESIDUAL FORCE             1.414E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.516E-03   AT NODE       6384   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.840E-06   AT NODE       6045   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      2.54       TIME AVG. MOMENT      0.248    
 LARGEST RESIDUAL MOMENT           -1.015E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.476E-03   AT NODE        271   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.895E-06   AT NODE        306   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     192.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    31 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.123    
     TOTAL ARC LENGTH    384.    , INCREMENT OF ARC LENGTH    128.    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.645E+03   TIME AVG. FORCE        298.    
 LARGEST RESIDUAL FORCE            -8.662E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.753E-03   AT NODE       6383   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.753E-03   AT NODE       6383   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.79       TIME AVG. MOMENT      0.359    
 LARGEST RESIDUAL MOMENT             2.20       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.294E-03   AT NODE        272   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.294E-03   AT NODE        272   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.699E+03   TIME AVG. FORCE        300.    
 LARGEST RESIDUAL FORCE              74.9       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.731E-03   AT NODE       6232   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.919E-04   AT NODE       5150   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.77       TIME AVG. MOMENT      0.358    
 LARGEST RESIDUAL MOMENT           -0.164       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.408E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.145E-04   AT NODE        302   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.698E+03   TIME AVG. FORCE        300.    
 LARGEST RESIDUAL FORCE             -3.53       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.734E-03   AT NODE       6232   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.133E-05   AT NODE       6189   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.76       TIME AVG. MOMENT      0.358    
 LARGEST RESIDUAL MOMENT           -1.861E-03   AT NODE        757   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.416E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.107E-04   AT NODE        316   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.698E+03   TIME AVG. FORCE        300.    
 LARGEST RESIDUAL FORCE             5.789E-05   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.734E-03   AT NODE       6232   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.829E-07   AT NODE       6483   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      3.76       TIME AVG. MOMENT      0.358    
 LARGEST RESIDUAL MOMENT           -1.197E-05   AT NODE        757   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.416E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.115E-07   AT NODE        320   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     288.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    32 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.309    
     TOTAL ARC LENGTH    575.    , INCREMENT OF ARC LENGTH    192.    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.112E+03   TIME AVG. FORCE        355.    
 LARGEST RESIDUAL FORCE            -1.834E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.514E-03   AT NODE       6230   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.514E-03   AT NODE       6230   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      5.55       TIME AVG. MOMENT      0.515    
 LARGEST RESIDUAL MOMENT             5.88       AT NODE        756   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.263E-02   AT NODE        275   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.263E-02   AT NODE        275   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.184E+03   TIME AVG. FORCE        357.    
 LARGEST RESIDUAL FORCE              918.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.443E-03   AT NODE       5927   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.649E-04   AT NODE       5590   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      5.44       TIME AVG. MOMENT      0.512    
 LARGEST RESIDUAL MOMENT           -0.717       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.254E-02   AT NODE        280   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.810E-03   AT NODE        310   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.179E+03   TIME AVG. FORCE        357.    
 LARGEST RESIDUAL FORCE              155.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.437E-03   AT NODE       5926   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.931E-04   AT NODE       6633   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      5.39       TIME AVG. MOMENT      0.511    
 LARGEST RESIDUAL MOMENT            5.061E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.249E-02   AT NODE        282   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.004E-04   AT NODE        320   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.178E+03   TIME AVG. FORCE        357.    
 LARGEST RESIDUAL FORCE            -0.138       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.435E-03   AT NODE       5926   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.895E-06   AT NODE       6927   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      5.39       TIME AVG. MOMENT      0.511    
 LARGEST RESIDUAL MOMENT            2.343E-03   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.248E-02   AT NODE        282   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.514E-05   AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     431.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    33 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.587    
     TOTAL ARC LENGTH    863.    , INCREMENT OF ARC LENGTH    288.    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.742E+03   TIME AVG. FORCE        427.    
 LARGEST RESIDUAL FORCE            -2.858E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.889E-03   AT NODE       5620   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.889E-03   AT NODE       5620   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.52       TIME AVG. MOMENT      0.717    
 LARGEST RESIDUAL MOMENT             15.1       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.815E-02   AT NODE        291   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.815E-02   AT NODE        291   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.5000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.735E+03   TIME AVG. FORCE        427.    
 LARGEST RESIDUAL FORCE            -5.390E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.876E-03   AT NODE       5464   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.538E-03   AT NODE       6329   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.21       TIME AVG. MOMENT      0.708    
 LARGEST RESIDUAL MOMENT           -0.575       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.806E-02   AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.889E-03   AT NODE        320   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.732E+03   TIME AVG. FORCE        427.    
 LARGEST RESIDUAL FORCE              406.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.999E-03   AT NODE       5463   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.313E-04   AT NODE       7227   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.12       TIME AVG. MOMENT      0.705    
 LARGEST RESIDUAL MOMENT            0.222       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.846E-02   AT NODE        300   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.718E-03   AT NODE        320   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.728E+03   TIME AVG. FORCE        427.    
 LARGEST RESIDUAL FORCE             -12.4       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.999E-03   AT NODE       5462   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.721E-05   AT NODE       7372   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.10       TIME AVG. MOMENT      0.704    
 LARGEST RESIDUAL MOMENT            4.938E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.846E-02   AT NODE        300   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.260E-04   AT NODE        326   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.728E+03   TIME AVG. FORCE        427.    
 LARGEST RESIDUAL FORCE             6.939E-03   AT NODE       1059   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.999E-03   AT NODE       5462   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.296E-06   AT NODE       7520   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      7.10       TIME AVG. MOMENT      0.704    
 LARGEST RESIDUAL MOMENT            9.448E-04   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.846E-02   AT NODE        300   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.889E-06   AT NODE        327   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    34 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.888    
     TOTAL ARC LENGTH   1.294E+03, INCREMENT OF ARC LENGTH    431.    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.076E+03   TIME AVG. FORCE        502.    
 LARGEST RESIDUAL FORCE             1.574E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.948E-03   AT NODE       5604   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.948E-03   AT NODE       5604   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.56       TIME AVG. MOMENT      0.929    
 LARGEST RESIDUAL MOMENT             12.4       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.890E-02   AT NODE        307   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.890E-02   AT NODE        307   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.108E+03   TIME AVG. FORCE        503.    
 LARGEST RESIDUAL FORCE             -624.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.925E-03   AT NODE      16478   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.289E-04   AT NODE      18542   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.40       TIME AVG. MOMENT      0.924    
 LARGEST RESIDUAL MOMENT           -0.149       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.923E-02   AT NODE        607   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.447E-03   AT NODE        623   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.097E+03   TIME AVG. FORCE        503.    
 LARGEST RESIDUAL FORCE              61.2       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.093E-03   AT NODE      16626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.015E-04   AT NODE      18844   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.36       TIME AVG. MOMENT      0.923    
 LARGEST RESIDUAL MOMENT            0.149       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.985E-02   AT NODE        609   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.324E-03   AT NODE        622   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.096E+03   TIME AVG. FORCE        503.    
 LARGEST RESIDUAL FORCE             0.362       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.096E-03   AT NODE      16626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.715E-05   AT NODE      18542   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      8.35       TIME AVG. MOMENT      0.923    
 LARGEST RESIDUAL MOMENT            1.022E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.986E-02   AT NODE        609   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.388E-05   AT NODE        628   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.096E+03   TIME AVG. FORCE        503.    
 LARGEST RESIDUAL FORCE            -9.872E-04   AT NODE      11937   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.096E-03   AT NODE      16626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.155E-07   AT NODE      18841   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      8.35       TIME AVG. MOMENT      0.923    
 LARGEST RESIDUAL MOMENT            8.829E-05   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.986E-02   AT NODE        609   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.877E-07   AT NODE        627   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    35 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.085    
     TOTAL ARC LENGTH   1.726E+03, INCREMENT OF ARC LENGTH    431.    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.355E+03   TIME AVG. FORCE        582.    
 LARGEST RESIDUAL FORCE             1.305E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.037E-03   AT NODE      16771   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.037E-03   AT NODE      16771   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.64       TIME AVG. MOMENT       1.16    
 LARGEST RESIDUAL MOMENT             10.1       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.061E-02   AT NODE        612   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.061E-02   AT NODE        612   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.422E+03   TIME AVG. FORCE        584.    
 LARGEST RESIDUAL FORCE              120.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.897E-03   AT NODE      16621   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.340E-04   AT NODE      19731   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.55       TIME AVG. MOMENT       1.16    
 LARGEST RESIDUAL MOMENT           -0.111       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.088E-02   AT NODE        615   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.920E-03   AT NODE      19597   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.414E+03   TIME AVG. FORCE        584.    
 LARGEST RESIDUAL FORCE              23.0       AT NODE        627   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.018E-03   AT NODE      16769   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.686E-04   AT NODE      19286   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.52       TIME AVG. MOMENT       1.16    
 LARGEST RESIDUAL MOMENT            0.122       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.138E-02   AT NODE        616   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.003E-03   AT NODE        627   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.414E+03   TIME AVG. FORCE        584.    
 LARGEST RESIDUAL FORCE             2.847E-02   AT NODE        157   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.020E-03   AT NODE      16769   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.934E-06   AT NODE      18836   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      9.52       TIME AVG. MOMENT       1.16    
 LARGEST RESIDUAL MOMENT            4.383E-03   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.138E-02   AT NODE        616   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.298E-05   AT NODE        633   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    36 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.254    
     TOTAL ARC LENGTH   2.157E+03, INCREMENT OF ARC LENGTH    431.    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.668E+03   TIME AVG. FORCE        667.    
 LARGEST RESIDUAL FORCE             1.297E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.851E-03   AT NODE      16916   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.851E-03   AT NODE      16916   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      10.8       TIME AVG. MOMENT       1.42    
 LARGEST RESIDUAL MOMENT             9.71       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.196E-02   AT NODE        618   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.196E-02   AT NODE        618   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.90000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.744E+03   TIME AVG. FORCE        669.    
 LARGEST RESIDUAL FORCE             -116.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.661E-03   AT NODE      16765   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.395E-04   AT NODE      20323   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      10.7       TIME AVG. MOMENT       1.42    
 LARGEST RESIDUAL MOMENT           -7.121E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.218E-02   AT NODE        620   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.773E-03   AT NODE      20187   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.740E+03   TIME AVG. FORCE        669.    
 LARGEST RESIDUAL FORCE              14.2       AT NODE        632   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.738E-03   AT NODE      16914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.876E-04   AT NODE      19729   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      10.7       TIME AVG. MOMENT       1.42    
 LARGEST RESIDUAL MOMENT            0.109       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.255E-02   AT NODE        621   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.880E-04   AT NODE        632   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.739E+03   TIME AVG. FORCE        669.    
 LARGEST RESIDUAL FORCE            -1.574E-02   AT NODE          9   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.739E-03   AT NODE      16914   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.877E-06   AT NODE      18982   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      10.7       TIME AVG. MOMENT       1.42    
 LARGEST RESIDUAL MOMENT            2.823E-03   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.256E-02   AT NODE        621   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.372E-05   AT NODE        636   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     647.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    37 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.422    
     TOTAL ARC LENGTH   2.589E+03, INCREMENT OF ARC LENGTH    431.    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.207E+03   TIME AVG. FORCE        762.    
 LARGEST RESIDUAL FORCE             3.033E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.131E-02   AT NODE      17061   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.131E-02   AT NODE      17061   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.7       TIME AVG. MOMENT       1.72    
 LARGEST RESIDUAL MOMENT             21.7       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.458E-02   AT NODE        623   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.458E-02   AT NODE        623   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.289E+03   TIME AVG. FORCE        764.    
 LARGEST RESIDUAL FORCE              469.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.088E-02   AT NODE      16762   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.404E-04   AT NODE      18705   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.7       TIME AVG. MOMENT       1.72    
 LARGEST RESIDUAL MOMENT           -0.381       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.476E-02   AT NODE        625   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.188E-03   AT NODE      20331   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.282E+03   TIME AVG. FORCE        764.    
 LARGEST RESIDUAL FORCE             -138.       AT NODE        340   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.105E-02   AT NODE      17059   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.178E-04   AT NODE       8996   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.6       TIME AVG. MOMENT       1.71    
 LARGEST RESIDUAL MOMENT            0.541       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.577E-02   AT NODE        626   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.456E-03   AT NODE        339   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.281E+03   TIME AVG. FORCE        764.    
 LARGEST RESIDUAL FORCE            -0.732       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.107E-02   AT NODE      17059   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.172E-05   AT NODE      19276   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      12.6       TIME AVG. MOMENT       1.71    
 LARGEST RESIDUAL MOMENT            3.834E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.580E-02   AT NODE        626   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.224E-04   AT NODE        640   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.281E+03   TIME AVG. FORCE        764.    
 LARGEST RESIDUAL FORCE             3.002E-03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.107E-02   AT NODE      17059   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.649E-07   AT NODE      19426   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      12.6       TIME AVG. MOMENT       1.71    
 LARGEST RESIDUAL MOMENT            7.179E-04   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.580E-02   AT NODE        626   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.951E-06   AT NODE        639   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    38 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.687    
     TOTAL ARC LENGTH   3.236E+03, INCREMENT OF ARC LENGTH    647.    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.830E+03   TIME AVG. FORCE        868.    
 LARGEST RESIDUAL FORCE             3.211E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.063E-02   AT NODE      17206   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.063E-02   AT NODE      17206   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.7       TIME AVG. MOMENT       2.05    
 LARGEST RESIDUAL MOMENT             21.6       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.687E-02   AT NODE        630   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.687E-02   AT NODE        630   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.90000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.904E+03   TIME AVG. FORCE        870.    
 LARGEST RESIDUAL FORCE              353.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.028E-02   AT NODE      16759   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.695E-04   AT NODE      20911   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.6       TIME AVG. MOMENT       2.05    
 LARGEST RESIDUAL MOMENT           -0.366       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.702E-02   AT NODE        631   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.266E-03   AT NODE      20772   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.898E+03   TIME AVG. FORCE        870.    
 LARGEST RESIDUAL FORCE             -121.       AT NODE        344   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.043E-02   AT NODE      17056   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.384E-04   AT NODE       9290   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.6       TIME AVG. MOMENT       2.05    
 LARGEST RESIDUAL MOMENT            0.596       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.797E-02   AT NODE        632   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.307E-03   AT NODE        344   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.897E+03   TIME AVG. FORCE        870.    
 LARGEST RESIDUAL FORCE             0.324       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.044E-02   AT NODE      17205   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.398E-05   AT NODE       8544   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      14.6       TIME AVG. MOMENT       2.05    
 LARGEST RESIDUAL MOMENT            3.586E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.800E-02   AT NODE        632   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.027E-04   AT NODE        347   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.897E+03   TIME AVG. FORCE        870.    
 LARGEST RESIDUAL FORCE             2.093E-03   AT NODE        911   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.044E-02   AT NODE      17205   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.722E-07   AT NODE       8694   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      14.6       TIME AVG. MOMENT       2.05    
 LARGEST RESIDUAL MOMENT            4.975E-04   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.800E-02   AT NODE        632   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.256E-06   AT NODE        347   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    39 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.962    
     TOTAL ARC LENGTH   3.883E+03, INCREMENT OF ARC LENGTH    647.    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.490E+03   TIME AVG. FORCE        986.    
 LARGEST RESIDUAL FORCE            -3.374E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.009E-02   AT NODE      17055   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.009E-02   AT NODE      17055   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      16.8       TIME AVG. MOMENT       2.41    
 LARGEST RESIDUAL MOMENT             21.2       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.899E-02   AT NODE        635   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.899E-02   AT NODE        635   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.571E+03   TIME AVG. FORCE        988.    
 LARGEST RESIDUAL FORCE              266.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.823E-03   AT NODE      16608   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.273E-04   AT NODE      10179   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      16.7       TIME AVG. MOMENT       2.41    
 LARGEST RESIDUAL MOMENT           -0.426       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.906E-02   AT NODE        339   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.315E-03   AT NODE      21065   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.563E+03   TIME AVG. FORCE        987.    
 LARGEST RESIDUAL FORCE              88.9       AT NODE        645   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.909E-03   AT NODE       5879   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.379E-04   AT NODE       9584   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      16.7       TIME AVG. MOMENT       2.41    
 LARGEST RESIDUAL MOMENT            0.586       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.981E-02   AT NODE        340   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.982E-03   AT NODE        348   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.563E+03   TIME AVG. FORCE        987.    
 LARGEST RESIDUAL FORCE             0.136       AT NODE        648   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.916E-03   AT NODE       5879   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.639E-05   AT NODE       8839   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      16.7       TIME AVG. MOMENT       2.41    
 LARGEST RESIDUAL MOMENT            2.946E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.984E-02   AT NODE        340   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.718E-05   AT NODE        351   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.563E+03   TIME AVG. FORCE        987.    
 LARGEST RESIDUAL FORCE            -1.251E-03   AT NODE      11937   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.916E-03   AT NODE       5879   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.128E-07   AT NODE       8988   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      16.7       TIME AVG. MOMENT       2.41    
 LARGEST RESIDUAL MOMENT            2.407E-04   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.984E-02   AT NODE        340   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.786E-07   AT NODE        351   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    40 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.247    
     TOTAL ARC LENGTH   4.530E+03, INCREMENT OF ARC LENGTH    647.    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.192E+03   TIME AVG. FORCE       1.114E+03
 LARGEST RESIDUAL FORCE            -3.546E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.642E-03   AT NODE       5581   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.642E-03   AT NODE       5581   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      19.0       TIME AVG. MOMENT       2.82    
 LARGEST RESIDUAL MOMENT             21.3       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.062E-02   AT NODE        342   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.062E-02   AT NODE        342   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.270E+03   TIME AVG. FORCE       1.116E+03
 LARGEST RESIDUAL FORCE              212.       AT NODE        649   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.449E-03   AT NODE       4986   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.776E-04   AT NODE      10325   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      18.9       TIME AVG. MOMENT       2.81    
 LARGEST RESIDUAL MOMENT           -0.506       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.063E-02   AT NODE        343   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.266E-03   AT NODE      10184   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.266E+03   TIME AVG. FORCE       1.116E+03
 LARGEST RESIDUAL FORCE              56.5       AT NODE        649   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.483E-03   AT NODE       5283   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.400E-04   AT NODE       9879   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      18.9       TIME AVG. MOMENT       2.81    
 LARGEST RESIDUAL MOMENT            0.530       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.117E-02   AT NODE        344   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.582E-03   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.266E+03   TIME AVG. FORCE       1.116E+03
 LARGEST RESIDUAL FORCE             5.741E-02   AT NODE        651   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.486E-03   AT NODE       5283   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.653E-06   AT NODE       9135   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      18.9       TIME AVG. MOMENT       2.81    
 LARGEST RESIDUAL MOMENT            2.040E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.119E-02   AT NODE        344   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.004E-05   AT NODE        353   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      6.266E+03   TIME AVG. FORCE       1.116E+03
 LARGEST RESIDUAL FORCE             6.454E-04   AT NODE        911   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.486E-03   AT NODE       5283   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.755E-08   AT NODE       9282   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      18.9       TIME AVG. MOMENT       2.81    
 LARGEST RESIDUAL MOMENT           -7.861E-05   AT NODE          1   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.119E-02   AT NODE        344   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.717E-07   AT NODE        356   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    41 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.544    
     TOTAL ARC LENGTH   5.177E+03, INCREMENT OF ARC LENGTH    647.    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.944E+03   TIME AVG. FORCE       1.255E+03
 LARGEST RESIDUAL FORCE            -3.723E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.279E-03   AT NODE       4837   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.279E-03   AT NODE       4837   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      21.3       TIME AVG. MOMENT       3.25    
 LARGEST RESIDUAL MOMENT             22.3       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.174E-02   AT NODE        346   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.174E-02   AT NODE        346   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.90000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.029E+03   TIME AVG. FORCE       1.257E+03
 LARGEST RESIDUAL FORCE              195.       AT NODE        651   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.143E-03   AT NODE       4539   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.269E-04   AT NODE      10471   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      21.2       TIME AVG. MOMENT       3.25    
 LARGEST RESIDUAL MOMENT           -0.583       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.178E-02   AT NODE        347   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.130E-03   AT NODE      10479   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.028E+03   TIME AVG. FORCE       1.257E+03
 LARGEST RESIDUAL FORCE              34.6       AT NODE        652   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.153E-03   AT NODE       4688   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.697E-04   AT NODE      10025   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      21.2       TIME AVG. MOMENT       3.25    
 LARGEST RESIDUAL MOMENT            0.468       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.212E-02   AT NODE        347   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.243E-03   AT NODE        354   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.028E+03   TIME AVG. FORCE       1.257E+03
 LARGEST RESIDUAL FORCE            -2.148E-02   AT NODE        357   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.154E-03   AT NODE       4688   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.300E-06   AT NODE       9430   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      21.2       TIME AVG. MOMENT       3.25    
 LARGEST RESIDUAL MOMENT            1.294E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.213E-02   AT NODE        347   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.008E-05   AT NODE        355   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    42 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.854    
     TOTAL ARC LENGTH   5.824E+03, INCREMENT OF ARC LENGTH    647.    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.751E+03   TIME AVG. FORCE       1.408E+03
 LARGEST RESIDUAL FORCE            -3.887E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.005E-03   AT NODE       4242   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.005E-03   AT NODE       4242   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.7       TIME AVG. MOMENT       3.73    
 LARGEST RESIDUAL MOMENT             24.0       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.253E-02   AT NODE        349   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.253E-02   AT NODE        349   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.835E+03   TIME AVG. FORCE       1.410E+03
 LARGEST RESIDUAL FORCE              177.       AT NODE        654   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.915E-03   AT NODE       3945   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.828E-04   AT NODE      10618   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.6       TIME AVG. MOMENT       3.73    
 LARGEST RESIDUAL MOMENT           -0.647       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.255E-02   AT NODE        350   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.960E-03   AT NODE      10625   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.832E+03   TIME AVG. FORCE       1.410E+03
 LARGEST RESIDUAL FORCE              23.9       AT NODE        654   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.917E-03   AT NODE       4094   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.288E-04   AT NODE      21347   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      23.6       TIME AVG. MOMENT       3.73    
 LARGEST RESIDUAL MOMENT            0.427       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.286E-02   AT NODE        350   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.035E-03   AT NODE        654   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.832E+03   TIME AVG. FORCE       1.410E+03
 LARGEST RESIDUAL FORCE             1.556E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.917E-03   AT NODE       4094   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.114E-06   AT NODE       9577   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      23.6       TIME AVG. MOMENT       3.73    
 LARGEST RESIDUAL MOMENT            8.681E-03   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.287E-02   AT NODE        350   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.934E-05   AT NODE        357   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     971.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    43 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.176    
     TOTAL ARC LENGTH   6.472E+03, INCREMENT OF ARC LENGTH    647.    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.155E+03   TIME AVG. FORCE       1.586E+03
 LARGEST RESIDUAL FORCE            -9.042E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.328E-02   AT NODE       2474   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.328E-02   AT NODE       2474   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.5       TIME AVG. MOMENT       4.27    
 LARGEST RESIDUAL MOMENT             58.0       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.486E-02   AT NODE        351   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.486E-02   AT NODE        351   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.154E+03   TIME AVG. FORCE       1.586E+03
 LARGEST RESIDUAL FORCE             -606.       AT NODE        307   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.335E-02   AT NODE       2326   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.989E-04   AT NODE      10372   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.5       TIME AVG. MOMENT       4.27    
 LARGEST RESIDUAL MOMENT            -2.97       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.487E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.288E-03   AT NODE        307   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.156E+03   TIME AVG. FORCE       1.586E+03
 LARGEST RESIDUAL FORCE             2.733E+04   AT NODE        609   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.320E-02   AT NODE      13205   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.831E-03   AT NODE      21243   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.9       TIME AVG. MOMENT       4.28    
 LARGEST RESIDUAL MOMENT            -4.60       AT NODE      22882   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.357E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.531E-02   AT NODE        609   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.718E+05   TIME AVG. FORCE       5.283E+03
 LARGEST RESIDUAL FORCE            -1.728E+07   AT NODE        298   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -0.194       AT NODE      10082   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -0.199       AT NODE      10082   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      350.       TIME AVG. MOMENT       11.6    
 LARGEST RESIDUAL MOMENT           -2.317E+05   AT NODE      11870   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION       1.28       AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION      1.29       AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    44 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.263E+03   TIME AVG. FORCE       1.566E+03
 LARGEST RESIDUAL FORCE            -1.283E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.979E-03   AT NODE       2474   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.979E-03   AT NODE       2474   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.0       TIME AVG. MOMENT       4.21    
 LARGEST RESIDUAL MOMENT             8.53       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.436E-02   AT NODE        351   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.436E-02   AT NODE        351   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.313E+03   TIME AVG. FORCE       1.567E+03
 LARGEST RESIDUAL FORCE             -41.2       AT NODE        306   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.992E-03   AT NODE       2475   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.872E-04   AT NODE      10372   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.0       TIME AVG. MOMENT       4.21    
 LARGEST RESIDUAL MOMENT           -9.513E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.440E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.368E-03   AT NODE        306   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.315E+03   TIME AVG. FORCE       1.567E+03
 LARGEST RESIDUAL FORCE             -50.6       AT NODE        304   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.003E-03   AT NODE       2475   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.228E-04   AT NODE      10224   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.0       TIME AVG. MOMENT       4.21    
 LARGEST RESIDUAL MOMENT            4.483E-02   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.452E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.507E-03   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.316E+03   TIME AVG. FORCE       1.567E+03
 LARGEST RESIDUAL FORCE             -2.79       AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.006E-03   AT NODE       2624   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.406E-05   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -3.668E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.0       TIME AVG. MOMENT       4.21    
 LARGEST RESIDUAL MOMENT           -1.052E-03   AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.453E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.541E-04   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -8.310E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     364.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    44 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.362    
     TOTAL ARC LENGTH   6.714E+03, INCREMENT OF ARC LENGTH    243.    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.992E+03   TIME AVG. FORCE       1.732E+03
 LARGEST RESIDUAL FORCE            -2.850E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.763E-03   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.763E-03   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.7       TIME AVG. MOMENT       4.73    
 LARGEST RESIDUAL MOMENT             19.3       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.757E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.757E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.792E+04   TIME AVG. FORCE       1.930E+03
 LARGEST RESIDUAL FORCE            -1.196E+06   AT NODE        313   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.384E-02   AT NODE      10235   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.240E-02   AT NODE      10383   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      38.2       TIME AVG. MOMENT       4.97    
 LARGEST RESIDUAL MOMENT           -2.092E+03   AT NODE      10374   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.288       AT NODE      10374   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.285       AT NODE      10374   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.169E+03   TIME AVG. FORCE       1.736E+03
 LARGEST RESIDUAL FORCE            -2.728E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.759E-02   AT NODE      10215   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.130E-02   AT NODE       9273   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.4       TIME AVG. MOMENT       4.84    
 LARGEST RESIDUAL MOMENT            -410.       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.256       AT NODE      10374   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.408E-02   AT NODE        364   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.728E+03   TIME AVG. FORCE       1.726E+03
 LARGEST RESIDUAL FORCE            -3.198E+05   AT NODE        296   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.431E-02   AT NODE      10066   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.151E-02   AT NODE      10236   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.9       TIME AVG. MOMENT       4.78    
 LARGEST RESIDUAL MOMENT            -140.       AT NODE      11869   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.164       AT NODE        314   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.124       AT NODE        295   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.841E+03   TIME AVG. FORCE       1.729E+03
 LARGEST RESIDUAL FORCE             2.026E+06   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.277E-02   AT NODE       9916   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.473E-02   AT NODE      20449   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      32.0       TIME AVG. MOMENT       4.83    
 LARGEST RESIDUAL MOMENT             72.7       AT NODE      22829   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.151       AT NODE        314   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.268E-02   AT NODE        661   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      9.087E+03   TIME AVG. FORCE       1.734E+03
 LARGEST RESIDUAL FORCE             8.810E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.227E-02   AT NODE       9768   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.706E-03   AT NODE      10380   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      31.4       TIME AVG. MOMENT       4.82    
 LARGEST RESIDUAL MOMENT             12.1       AT NODE      22894   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.139       AT NODE        314   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.453E-02   AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      8.679E+03   TIME AVG. FORCE       1.725E+03
 LARGEST RESIDUAL FORCE             1.020E+05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.433E-02   AT NODE       9620   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.399E-03   AT NODE      20601   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.7       TIME AVG. MOMENT       4.80    
 LARGEST RESIDUAL MOMENT             15.7       AT NODE      22834   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.143       AT NODE        314   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.374E-02   AT NODE        657   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     8


 AVERAGE FORCE                      8.727E+03   TIME AVG. FORCE       1.726E+03
 LARGEST RESIDUAL FORCE            -2.746E+04   AT NODE        300   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.566E-02   AT NODE       9621   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.058E-03   AT NODE       9782   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.4       TIME AVG. MOMENT       4.79    
 LARGEST RESIDUAL MOMENT            -6.12       AT NODE      11868   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.146       AT NODE        312   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.524E-02   AT NODE        300   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     9


 AVERAGE FORCE                      8.611E+03   TIME AVG. FORCE       1.723E+03
 LARGEST RESIDUAL FORCE             2.093E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.767E-02   AT NODE       9474   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.825E-03   AT NODE      20511   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      30.3       TIME AVG. MOMENT       4.79    
 LARGEST RESIDUAL MOMENT             3.81       AT NODE      22895   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.151       AT NODE        311   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.918E-02   AT NODE        596   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE RATE OF CONVERGENCE IS VERY SLOW. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    45 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.826E+03   TIME AVG. FORCE       1.728E+03
 LARGEST RESIDUAL FORCE            -1.615E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.822E-03   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.822E-03   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.0       TIME AVG. MOMENT       4.72    
 LARGEST RESIDUAL MOMENT             11.1       AT NODE          1   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.819E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.819E-02   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.106E+03   TIME AVG. FORCE       1.734E+03
 LARGEST RESIDUAL FORCE            -1.674E+05   AT NODE        296   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.326E-02   AT NODE      10086   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.212E-02   AT NODE      10233   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.0       TIME AVG. MOMENT       4.76    
 LARGEST RESIDUAL MOMENT             62.5       AT NODE      11872   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.999E-02   AT NODE        295   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.810E-02   AT NODE        296   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.688E+03   TIME AVG. FORCE       1.725E+03
 LARGEST RESIDUAL FORCE            -5.075E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.004E-02   AT NODE      10235   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.455E-03   AT NODE       9786   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      28.2       TIME AVG. MOMENT       4.74    
 LARGEST RESIDUAL MOMENT             11.2       AT NODE      11873   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.928E-02   AT NODE        295   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.839E-02   AT NODE        360   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.548E+03   TIME AVG. FORCE       1.722E+03
 LARGEST RESIDUAL FORCE             1.037E+05   AT NODE        608   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.148E-02   AT NODE      20944   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.458E-02   AT NODE      20350   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.2       TIME AVG. MOMENT       4.77    
 LARGEST RESIDUAL MOMENT             28.7       AT NODE      22882   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.131       AT NODE        611   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.757E-02   AT NODE        608   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.349E+03   TIME AVG. FORCE       1.718E+03
 LARGEST RESIDUAL FORCE             1.164E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.112E-02   AT NODE      20795   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.348E-03   AT NODE      18866   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      28.5       TIME AVG. MOMENT       4.75    
 LARGEST RESIDUAL MOMENT            -4.53       AT NODE      22894   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.126       AT NODE        611   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.675E-02   AT NODE      19901   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      8.253E+03   TIME AVG. FORCE       1.715E+03
 LARGEST RESIDUAL FORCE            -1.057E+05   AT NODE        302   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.745E-02   AT NODE       9474   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.102E-02   AT NODE       9482   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      29.3       TIME AVG. MOMENT       4.77    
 LARGEST RESIDUAL MOMENT            -24.9       AT NODE      11866   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.149       AT NODE        310   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.880E-02   AT NODE        302   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    45 STARTS. ATTEMPT NUMBER  3

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.423E+03   TIME AVG. FORCE       1.719E+03
 LARGEST RESIDUAL FORCE            -5.684E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.091E-03   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.091E-03   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.4       TIME AVG. MOMENT       4.68    
 LARGEST RESIDUAL MOMENT           -0.503       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.288E-03   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.288E-03   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.417E+03   TIME AVG. FORCE       1.719E+03
 LARGEST RESIDUAL FORCE             1.132E+03   AT NODE        600   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.117E-03   AT NODE      20064   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.090E-03   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.3       TIME AVG. MOMENT       4.68    
 LARGEST RESIDUAL MOMENT           -6.783E-02   AT NODE      22451   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.178E-03   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.130E-03   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.427E+03   TIME AVG. FORCE       1.719E+03
 LARGEST RESIDUAL FORCE             -266.       AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.104E-03   AT NODE       2474   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.240E-04   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.4       TIME AVG. MOMENT       4.68    
 LARGEST RESIDUAL MOMENT           -2.679E-02   AT NODE      22891   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.462E-03   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.461E-03   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.439E+03   TIME AVG. FORCE       1.720E+03
 LARGEST RESIDUAL FORCE            -3.728E+03   AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.583E-03   AT NODE      10374   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.945E-03   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.5       TIME AVG. MOMENT       4.68    
 LARGEST RESIDUAL MOMENT            0.202       AT NODE      11723   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.174E-02   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.293E-02   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    45 STARTS. ATTEMPT NUMBER  4

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.326E+03   TIME AVG. FORCE       1.717E+03
 LARGEST RESIDUAL FORCE              14.0       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.116E-05   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.116E-05   AT NODE       4263   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       4.67    
 LARGEST RESIDUAL MOMENT            2.259E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.479E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.479E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.326E+03   TIME AVG. FORCE       1.717E+03
 LARGEST RESIDUAL FORCE            -4.872E-02   AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.184E-05   AT NODE       4710   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.109E-06   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.469E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       4.67    
 LARGEST RESIDUAL MOMENT           -1.644E-04   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.488E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.677E-05   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.80000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.326E+03   TIME AVG. FORCE       1.717E+03
 LARGEST RESIDUAL FORCE             1.112E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.185E-05   AT NODE       4710   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.184E-08   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       4.67    
 LARGEST RESIDUAL MOMENT            1.352E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.488E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.372E-07   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     17.1    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    45 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.364    
     TOTAL ARC LENGTH   6.726E+03, INCREMENT OF ARC LENGTH    11.4    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.337E+03   TIME AVG. FORCE       1.861E+03
 LARGEST RESIDUAL FORCE              29.8       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.744E-05   AT NODE       4561   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.744E-05   AT NODE       4561   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.12    
 LARGEST RESIDUAL MOMENT            3.468E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.727E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.727E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.337E+03   TIME AVG. FORCE       1.861E+03
 LARGEST RESIDUAL FORCE            -4.489E-02   AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.809E-05   AT NODE       4710   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.823E-06   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.029E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.12    
 LARGEST RESIDUAL MOMENT            2.336E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.733E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.489E-05   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.337E+03   TIME AVG. FORCE       1.861E+03
 LARGEST RESIDUAL FORCE            -2.084E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.809E-05   AT NODE       4710   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.050E-08   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.12    
 LARGEST RESIDUAL MOMENT           -2.220E-06   AT NODE          2   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.733E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.269E-07   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     25.6    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    46 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.366    
     TOTAL ARC LENGTH   6.743E+03, INCREMENT OF ARC LENGTH    17.1    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.353E+03   TIME AVG. FORCE       1.999E+03
 LARGEST RESIDUAL FORCE              64.3       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.184E-04   AT NODE       4859   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.184E-04   AT NODE       4859   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.54    
 LARGEST RESIDUAL MOMENT            5.119E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.605E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.605E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.352E+03   TIME AVG. FORCE       1.999E+03
 LARGEST RESIDUAL FORCE             -1.61       AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.220E-04   AT NODE       5009   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.106E-05   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.031E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.54    
 LARGEST RESIDUAL MOMENT           -1.101E-03   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.612E-04   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.690E-04   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.352E+03   TIME AVG. FORCE       1.999E+03
 LARGEST RESIDUAL FORCE            -2.562E-02   AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.223E-04   AT NODE       5009   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.235E-06   AT NODE      10077   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -8.317E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.54    
 LARGEST RESIDUAL MOMENT            2.301E-04   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.948E-04   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.391E-05   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -7.091E-06
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     38.4    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    47 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.370    
     TOTAL ARC LENGTH   6.768E+03, INCREMENT OF ARC LENGTH    25.6    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.371E+03   TIME AVG. FORCE       2.132E+03
 LARGEST RESIDUAL FORCE             -114.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.006E-04   AT NODE      10375   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.006E-04   AT NODE      10375   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.2       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            6.708E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.160E-03   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.160E-03   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.375E+03   TIME AVG. FORCE       2.132E+03
 LARGEST RESIDUAL FORCE              178.       AT NODE        600   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.915E-04   AT NODE       2026   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.396E-04   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.2       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            2.611E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.480E-04   AT NODE        352   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.823E-03   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.380E+03   TIME AVG. FORCE       2.132E+03
 LARGEST RESIDUAL FORCE             -193.       AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -3.076E-04   AT NODE      10374   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.388E-04   AT NODE      10226   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.2       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            1.559E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.291E-03   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.941E-03   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.4000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.383E+03   TIME AVG. FORCE       2.132E+03
 LARGEST RESIDUAL FORCE              286.       AT NODE        600   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.011E-04   AT NODE      20063   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.547E-04   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.2       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            2.588E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.407E-03   AT NODE        596   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.583E-03   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    48 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          2

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.363E+03   TIME AVG. FORCE       2.132E+03
 LARGEST RESIDUAL FORCE             -17.2       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.118E-04   AT NODE      10375   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.118E-04   AT NODE      10375   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            2.546E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.036E-04   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.036E-04   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.90000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.90000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.361E+03   TIME AVG. FORCE       2.132E+03
 LARGEST RESIDUAL FORCE              154.       AT NODE        600   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -5.181E-04   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.082E-04   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.2       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT           -2.423E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.431E-03   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.627E-03   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.352E+03   TIME AVG. FORCE       2.131E+03
 LARGEST RESIDUAL FORCE              44.8       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.830E-04   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.019E-04   AT NODE      15884   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            4.615E-02   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.059E-03   AT NODE        599   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.857E-04   AT NODE        649   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.352E+03   TIME AVG. FORCE       2.131E+03
 LARGEST RESIDUAL FORCE             8.540E-02   AT NODE        599   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -4.738E-04   AT NODE      21103   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.247E-06   AT NODE      21253   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.762E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       5.95    
 LARGEST RESIDUAL MOMENT            2.350E-04   AT NODE        157   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.997E-03   AT NODE        599   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.191E-05   AT NODE        599   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -6.004E-07
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     14.4    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    48 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.367    
     TOTAL ARC LENGTH   6.778E+03, INCREMENT OF ARC LENGTH    9.60    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.2000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.2000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.347E+03   TIME AVG. FORCE       2.258E+03
 LARGEST RESIDUAL FORCE             -38.9       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.785E-04   AT NODE       9780   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.785E-04   AT NODE       9780   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       6.34    
 LARGEST RESIDUAL MOMENT           -4.454E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.031E-03   AT NODE        302   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.031E-03   AT NODE        302   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.347E+03   TIME AVG. FORCE       2.258E+03
 LARGEST RESIDUAL FORCE             0.171       AT NODE        600   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.663E-04   AT NODE       9781   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.298E-05   AT NODE      21252   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -5.697E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       6.34    
 LARGEST RESIDUAL MOMENT           -3.508E-04   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.473E-04   AT NODE        301   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.750E-05   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.347E+03   TIME AVG. FORCE       2.258E+03
 LARGEST RESIDUAL FORCE             2.059E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -1.662E-04   AT NODE       9781   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.062E-07   AT NODE      21252   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       6.34    
 LARGEST RESIDUAL MOMENT           -2.652E-06   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.466E-04   AT NODE        301   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.238E-07   AT NODE        600   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     21.6    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    49 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.363    
     TOTAL ARC LENGTH   6.792E+03, INCREMENT OF ARC LENGTH    14.4    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.3000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.3000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.340E+03   TIME AVG. FORCE       2.380E+03
 LARGEST RESIDUAL FORCE             -93.5       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.334E-04   AT NODE       9781   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.334E-04   AT NODE       9781   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       6.72    
 LARGEST RESIDUAL MOMENT           -6.784E-02   AT NODE          9   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.308E-03   AT NODE        301   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.308E-03   AT NODE        301   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.4000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.5000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.340E+03   TIME AVG. FORCE       2.380E+03
 LARGEST RESIDUAL FORCE            -0.213       AT NODE        303   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.204E-04   AT NODE       9632   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.447E-05   AT NODE      10225   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    3.300E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       6.72    
 LARGEST RESIDUAL MOMENT           -2.787E-04   AT NODE        751   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.215E-03   AT NODE        301   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.770E-05   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    141929     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.47E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   1.1000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   1.1000    
       ELAPSED WALLCLOCK TIME (SEC) =          2
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.340E+03   TIME AVG. FORCE       2.380E+03
 LARGEST RESIDUAL FORCE             1.234E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.        -2.204E-04   AT NODE       9632   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.855E-09   AT NODE      10676   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      25.1       TIME AVG. MOMENT       6.72    
 LARGEST RESIDUAL MOMENT           -3.613E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.215E-03   AT NODE        301   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.284E-08   AT NODE        300   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     32.4    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    50 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.358    
     TOTAL ARC LENGTH   6.814E+03, INCREMENT OF ARC LENGTH    21.6    


     THE NUMBER OF INCS ON THE *STEP CARD HAS BEEN COMPLETED - THE STEP ENDS



     ANALYSIS SUMMARY:
     TOTAL OF         50  INCREMENTS
                       5  CUTBACKS IN AUTOMATIC INCREMENTATION
                     181  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     181  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     181  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       0  WARNING MESSAGES DURING USER INPUT PROCESSING
                      17  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                      17  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   286.20    
       SYSTEM TIME (SEC)    =   7.6000    
       TOTAL CPU TIME (SEC) =   293.80    
       WALLCLOCK TIME (SEC) =        460
