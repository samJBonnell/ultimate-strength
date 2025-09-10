1

   Abaqus 2020                                  Date 09-Sep-2025   Time 14:53:02
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

     CHARACTERISTIC ELEMENT LENGTH     2.006E-02

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.349       TIME AVG. FORCE       0.349    
 LARGEST RESIDUAL FORCE            -1.820E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.960E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.960E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.070E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     5.711E-05   TIME AVG. MOMENT      5.711E-05
 LARGEST RESIDUAL MOMENT           -1.986E-07   AT NODE       1687   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.788E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.788E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.349       TIME AVG. FORCE       0.349    
 LARGEST RESIDUAL FORCE             2.689E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.960E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.079E-12   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.711E-05   TIME AVG. MOMENT      5.711E-05
 LARGEST RESIDUAL MOMENT           -6.280E-09   AT NODE       1529   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.788E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.483E-11   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     1 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.3333E-04
     TOTAL ARC LENGTH   1.000E-02, INCREMENT OF ARC LENGTH   1.000E-02


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.14       TIME AVG. FORCE       0.745    
 LARGEST RESIDUAL FORCE             1.980E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.960E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.960E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.176E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.450E-04   TIME AVG. MOMENT      1.010E-04
 LARGEST RESIDUAL MOMENT            1.949E-07   AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.788E-06   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.788E-06   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.14       TIME AVG. FORCE       0.745    
 LARGEST RESIDUAL FORCE            -2.312E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.960E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.034E-12   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.450E-04   TIME AVG. MOMENT      1.010E-04
 LARGEST RESIDUAL MOMENT            5.137E-09   AT NODE      25670   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.788E-06   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.456E-11   AT NODE          6   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.500E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     2 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         6.6666E-04
     TOTAL ARC LENGTH   2.000E-02, INCREMENT OF ARC LENGTH   1.000E-02


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.97       TIME AVG. FORCE        1.15    
 LARGEST RESIDUAL FORCE            -3.136E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.941E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.941E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.863E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.506E-04   TIME AVG. MOMENT      1.509E-04
 LARGEST RESIDUAL MOMENT            4.379E-07   AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.682E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.682E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       1.97       TIME AVG. FORCE        1.15    
 LARGEST RESIDUAL FORCE             3.938E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.941E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.085E-12   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.506E-04   TIME AVG. MOMENT      1.509E-04
 LARGEST RESIDUAL MOMENT            5.861E-09   AT NODE       1208   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.682E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.272E-11   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    2.250E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     3 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.1666E-03
     TOTAL ARC LENGTH   3.500E-02, INCREMENT OF ARC LENGTH   1.500E-02


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       3.19       TIME AVG. FORCE        1.66    
 LARGEST RESIDUAL FORCE            -7.600E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.411E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.411E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -4.515E-13
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     4.078E-04   TIME AVG. MOMENT      2.151E-04
 LARGEST RESIDUAL MOMENT           -9.848E-07   AT NODE       1687   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.022E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.022E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       3.19       TIME AVG. FORCE        1.66    
 LARGEST RESIDUAL FORCE             4.554E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.411E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.044E-11   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     4.078E-04   TIME AVG. MOMENT      2.151E-04
 LARGEST RESIDUAL MOMENT           -4.883E-09   AT NODE      23449   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.022E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.353E-11   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    3.375E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     4 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.9166E-03
     TOTAL ARC LENGTH   5.750E-02, INCREMENT OF ARC LENGTH   2.250E-02


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       5.02       TIME AVG. FORCE        2.33    
 LARGEST RESIDUAL FORCE            -1.744E-03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.616E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.616E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.036E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     6.420E-04   TIME AVG. MOMENT      3.005E-04
 LARGEST RESIDUAL MOMENT           -2.216E-06   AT NODE       1369   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.033E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.033E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       5.02       TIME AVG. FORCE        2.33    
 LARGEST RESIDUAL FORCE             5.449E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.616E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.601E-11   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     6.420E-04   TIME AVG. MOMENT      3.005E-04
 LARGEST RESIDUAL MOMENT            4.850E-09   AT NODE      24945   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.033E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.655E-10   AT NODE          6   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    5.063E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     5 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.0415E-03
     TOTAL ARC LENGTH   9.125E-02, INCREMENT OF ARC LENGTH   3.375E-02


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       7.74       TIME AVG. FORCE        3.23    
 LARGEST RESIDUAL FORCE             3.867E-03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.923E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.923E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    2.297E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     9.896E-04   TIME AVG. MOMENT      4.153E-04
 LARGEST RESIDUAL MOMENT            4.985E-06   AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.049E-06   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.049E-06   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       7.74       TIME AVG. FORCE        3.23    
 LARGEST RESIDUAL FORCE            -5.304E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.923E-07   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.036E-10   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     9.896E-04   TIME AVG. MOMENT      4.153E-04
 LARGEST RESIDUAL MOMENT           -4.090E-09   AT NODE         21   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.049E-06   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.725E-10   AT NODE          6   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    7.594E-02

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     6 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         4.7288E-03
     TOTAL ARC LENGTH   0.142    , INCREMENT OF ARC LENGTH   5.063E-02


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       11.8       TIME AVG. FORCE        4.46    
 LARGEST RESIDUAL FORCE            -8.613E-03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.488E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.488E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -5.117E-12
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     1.508E-03   TIME AVG. MOMENT      5.714E-04
 LARGEST RESIDUAL MOMENT           -1.122E-05   AT NODE       1369   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.357E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.357E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.70000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.80000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       11.8       TIME AVG. FORCE        4.46    
 LARGEST RESIDUAL FORCE             6.399E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.488E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.333E-10   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.508E-03   TIME AVG. MOMENT      5.714E-04
 LARGEST RESIDUAL MOMENT           -6.817E-09   AT NODE      24189   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.357E-05   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.390E-10   AT NODE         12   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.114    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     7 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         7.2595E-03
     TOTAL ARC LENGTH   0.218    , INCREMENT OF ARC LENGTH   7.594E-02


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       17.8       TIME AVG. FORCE        6.13    
 LARGEST RESIDUAL FORCE             1.933E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.232E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.232E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.149E-11
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     2.281E-03   TIME AVG. MOMENT      7.851E-04
 LARGEST RESIDUAL MOMENT            2.523E-05   AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.036E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.036E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       17.8       TIME AVG. FORCE        6.13    
 LARGEST RESIDUAL FORCE            -1.123E-05   AT NODE      24930   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.232E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.256E-10   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.281E-03   TIME AVG. MOMENT      7.852E-04
 LARGEST RESIDUAL MOMENT           -6.964E-09   AT NODE      23450   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.036E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.891E-09   AT NODE          6   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.171    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     8 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.1055E-02
     TOTAL ARC LENGTH   0.332    , INCREMENT OF ARC LENGTH   0.114    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       26.8       TIME AVG. FORCE        8.43    
 LARGEST RESIDUAL FORCE             4.350E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.348E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.348E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     3.435E-03   TIME AVG. MOMENT      1.080E-03
 LARGEST RESIDUAL MOMENT            5.676E-05   AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.053E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.053E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       26.8       TIME AVG. FORCE        8.43    
 LARGEST RESIDUAL FORCE            -2.623E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.348E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.185E-09   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     3.435E-03   TIME AVG. MOMENT      1.080E-03
 LARGEST RESIDUAL MOMENT            4.605E-09   AT NODE      24944   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.053E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.268E-09   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.256    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT     9 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         1.6748E-02
     TOTAL ARC LENGTH   0.503    , INCREMENT OF ARC LENGTH   0.171    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       40.2       TIME AVG. FORCE        11.6    
 LARGEST RESIDUAL FORCE             9.780E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.021E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.021E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     5.156E-03   TIME AVG. MOMENT      1.487E-03
 LARGEST RESIDUAL MOMENT            1.277E-04   AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.578E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.578E-05   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       40.1       TIME AVG. FORCE        11.6    
 LARGEST RESIDUAL FORCE            -3.280E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.021E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.674E-09   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.155E-03   TIME AVG. MOMENT      1.487E-03
 LARGEST RESIDUAL MOMENT            6.733E-09   AT NODE      24189   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.577E-05   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.648E-09   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.384    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    10 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         2.5284E-02
     TOTAL ARC LENGTH   0.759    , INCREMENT OF ARC LENGTH   0.256    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       59.9       TIME AVG. FORCE        16.0    
 LARGEST RESIDUAL FORCE             0.220       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.530E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.530E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     7.723E-03   TIME AVG. MOMENT      2.054E-03
 LARGEST RESIDUAL MOMENT            2.872E-04   AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.864E-05   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.864E-05   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       59.9       TIME AVG. FORCE        16.0    
 LARGEST RESIDUAL FORCE            -4.211E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.528E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.044E-09   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     7.723E-03   TIME AVG. MOMENT      2.054E-03
 LARGEST RESIDUAL MOMENT           -1.760E-08   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.862E-05   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.186E-08   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.577    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    11 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         3.8084E-02
     TOTAL ARC LENGTH    1.14    , INCREMENT OF ARC LENGTH   0.384    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       89.0       TIME AVG. FORCE        22.1    
 LARGEST RESIDUAL FORCE            -0.494       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.129E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.129E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.155E-02   TIME AVG. MOMENT      2.846E-03
 LARGEST RESIDUAL MOMENT           -6.458E-04   AT NODE       1369   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.029E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.029E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       89.0       TIME AVG. FORCE        22.1    
 LARGEST RESIDUAL FORCE            -5.668E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.129E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.369E-08   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.155E-02   TIME AVG. MOMENT      2.846E-03
 LARGEST RESIDUAL MOMENT            6.057E-08   AT NODE        584   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.029E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.972E-08   AT NODE         12   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    0.865    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    12 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         5.7273E-02
     TOTAL ARC LENGTH    1.72    , INCREMENT OF ARC LENGTH   0.577    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       132.       TIME AVG. FORCE        30.5    
 LARGEST RESIDUAL FORCE             -1.11       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.692E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.692E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     1.728E-02   TIME AVG. MOMENT      3.956E-03
 LARGEST RESIDUAL MOMENT            1.452E-03   AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.542E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.542E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       132.       TIME AVG. FORCE        30.5    
 LARGEST RESIDUAL FORCE             1.339E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.691E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.112E-08   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     1.728E-02   TIME AVG. MOMENT      3.956E-03
 LARGEST RESIDUAL MOMENT           -2.067E-07   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.541E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.137E-07   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.30    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    13 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         8.6030E-02
     TOTAL ARC LENGTH    2.58    , INCREMENT OF ARC LENGTH   0.865    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       194.       TIME AVG. FORCE        42.2    
 LARGEST RESIDUAL FORCE             -2.49       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.536E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.536E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     2.582E-02   TIME AVG. MOMENT      5.518E-03
 LARGEST RESIDUAL MOMENT           -3.263E-03   AT NODE       1687   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.310E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.310E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       194.       TIME AVG. FORCE        42.2    
 LARGEST RESIDUAL FORCE            -2.256E-05   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.533E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.108E-08   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     2.582E-02   TIME AVG. MOMENT      5.518E-03
 LARGEST RESIDUAL MOMENT           -7.165E-07   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.307E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.620E-07   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     1.95    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    14 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1291    
     TOTAL ARC LENGTH    3.88    , INCREMENT OF ARC LENGTH    1.30    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       286.       TIME AVG. FORCE        58.4    
 LARGEST RESIDUAL FORCE              5.58       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.797E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.797E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     3.860E-02   TIME AVG. MOMENT      7.723E-03
 LARGEST RESIDUAL MOMENT            7.328E-03   AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.457E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.457E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       286.       TIME AVG. FORCE        58.4    
 LARGEST RESIDUAL FORCE            -7.215E-05   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.792E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.636E-07   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     3.860E-02   TIME AVG. MOMENT      7.723E-03
 LARGEST RESIDUAL MOMENT            2.526E-06   AT NODE        169   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.451E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.113E-07   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     2.92    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    15 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.1936    
     TOTAL ARC LENGTH    5.83    , INCREMENT OF ARC LENGTH    1.95    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       420.       TIME AVG. FORCE        81.0    
 LARGEST RESIDUAL FORCE              12.5       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.680E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.680E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     5.765E-02   TIME AVG. MOMENT      1.084E-02
 LARGEST RESIDUAL MOMENT            1.644E-02   AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.167E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.167E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       419.       TIME AVG. FORCE        81.0    
 LARGEST RESIDUAL FORCE             9.605E-05   AT NODE        554   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.669E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.807E-07   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     5.765E-02   TIME AVG. MOMENT      1.084E-02
 LARGEST RESIDUAL MOMENT           -9.103E-06   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.153E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.452E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     4.38    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    16 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.2900    
     TOTAL ARC LENGTH    8.75    , INCREMENT OF ARC LENGTH    2.92    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       616.       TIME AVG. FORCE        112.    
 LARGEST RESIDUAL FORCE              27.8       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.484E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.484E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     8.609E-02   TIME AVG. MOMENT      1.527E-02
 LARGEST RESIDUAL MOMENT            3.679E-02   AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.706E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.706E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       615.       TIME AVG. FORCE        112.    
 LARGEST RESIDUAL FORCE            -5.923E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.455E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.005E-07   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.922E-11
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     8.608E-02   TIME AVG. MOMENT      1.527E-02
 LARGEST RESIDUAL MOMENT           -3.385E-05   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.672E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.545E-06   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     6.57    

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    17 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.4337    
     TOTAL ARC LENGTH    13.1    , INCREMENT OF ARC LENGTH    4.38    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       903.       TIME AVG. FORCE        156.    
 LARGEST RESIDUAL FORCE              61.3       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.263E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.263E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      2.156E-02
 LARGEST RESIDUAL MOMENT            8.204E-02   AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.145E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.145E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                       900.       TIME AVG. FORCE        156.    
 LARGEST RESIDUAL FORCE            -5.936E-03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.256E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.180E-06   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -2.112E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      2.156E-02
 LARGEST RESIDUAL MOMENT            1.317E-04   AT NODE        169   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.136E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.012E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                       900.       TIME AVG. FORCE        156.    
 LARGEST RESIDUAL FORCE             3.347E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.256E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.393E-10   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      2.156E-02
 LARGEST RESIDUAL MOMENT            3.804E-08   AT NODE          7   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.136E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.732E-09   AT NODE        376   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     9.85    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    18 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.6472    
     TOTAL ARC LENGTH    19.7    , INCREMENT OF ARC LENGTH    6.57    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.317E+03   TIME AVG. FORCE        217.    
 LARGEST RESIDUAL FORCE              133.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.871E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.871E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.192       TIME AVG. MOMENT      3.052E-02
 LARGEST RESIDUAL MOMENT            0.181       AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.689E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.689E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.309E+03   TIME AVG. FORCE        217.    
 LARGEST RESIDUAL FORCE            -4.441E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.850E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.443E-06   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -1.817E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.192       TIME AVG. MOMENT      3.051E-02
 LARGEST RESIDUAL MOMENT           -5.461E-04   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.666E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.429E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.309E+03   TIME AVG. FORCE        217.    
 LARGEST RESIDUAL FORCE             3.311E-05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.850E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.793E-09   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.192       TIME AVG. MOMENT      3.051E-02
 LARGEST RESIDUAL MOMENT           -4.137E-07   AT NODE        584   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.666E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.591E-08   AT NODE        376   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     14.8    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    19 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR         0.9618    
     TOTAL ARC LENGTH    29.5    , INCREMENT OF ARC LENGTH    9.85    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.898E+03   TIME AVG. FORCE        301.    
 LARGEST RESIDUAL FORCE              278.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.738E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.738E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.285       TIME AVG. MOMENT      4.326E-02
 LARGEST RESIDUAL MOMENT            0.394       AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.456E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.456E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.879E+03   TIME AVG. FORCE        300.    
 LARGEST RESIDUAL FORCE             0.290       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.676E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.405E-05   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION    1.466E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.285       TIME AVG. MOMENT      4.323E-02
 LARGEST RESIDUAL MOMENT           -2.454E-03   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.388E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.079E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.879E+03   TIME AVG. FORCE        300.    
 LARGEST RESIDUAL FORCE             2.330E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.676E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.953E-08   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.285       TIME AVG. MOMENT      4.323E-02
 LARGEST RESIDUAL MOMENT            6.240E-06   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.388E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.541E-07   AT NODE          6   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     22.2    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    20 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          1.417    
     TOTAL ARC LENGTH    44.3    , INCREMENT OF ARC LENGTH    14.8    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.670E+03   TIME AVG. FORCE        413.    
 LARGEST RESIDUAL FORCE              526.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.892E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.892E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.421       TIME AVG. MOMENT      6.121E-02
 LARGEST RESIDUAL MOMENT            0.817       AT NODE       1368   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.450E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.450E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.611E+03   TIME AVG. FORCE        410.    
 LARGEST RESIDUAL FORCE             -1.36       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.689E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.629E-05   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -9.340E-08
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                     0.417       TIME AVG. MOMENT      6.104E-02
 LARGEST RESIDUAL MOMENT           -1.182E-02   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.233E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.224E-04   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.611E+03   TIME AVG. FORCE        410.    
 LARGEST RESIDUAL FORCE            -4.725E-03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.691E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.292E-07   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.417       TIME AVG. MOMENT      6.105E-02
 LARGEST RESIDUAL MOMENT            1.017E-04   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.235E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.752E-06   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     33.3    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    21 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.044    
     TOTAL ARC LENGTH    66.5    , INCREMENT OF ARC LENGTH    22.2    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.597E+03   TIME AVG. FORCE        555.    
 LARGEST RESIDUAL FORCE              703.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.120E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.120E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.597       TIME AVG. MOMENT      8.539E-02
 LARGEST RESIDUAL MOMENT             1.48       AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.423E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.423E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.449E+03   TIME AVG. FORCE        548.    
 LARGEST RESIDUAL FORCE              8.11       AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.318E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.866E-05   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.584       TIME AVG. MOMENT      8.481E-02
 LARGEST RESIDUAL MOMENT           -5.268E-02   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.773E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.634E-04   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.452E+03   TIME AVG. FORCE        548.    
 LARGEST RESIDUAL FORCE             5.182E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.308E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.417E-06   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.584       TIME AVG. MOMENT      8.483E-02
 LARGEST RESIDUAL MOMENT            1.098E-03   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.786E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.358E-05   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.452E+03   TIME AVG. FORCE        548.    
 LARGEST RESIDUAL FORCE             1.922E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.308E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.264E-10   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.584       TIME AVG. MOMENT      8.483E-02
 LARGEST RESIDUAL MOMENT           -6.743E-07   AT NODE      18976   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.786E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.053E-08   AT NODE        377   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     49.9    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    22 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          2.806    
     TOTAL ARC LENGTH    99.7    , INCREMENT OF ARC LENGTH    33.3    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.468E+03   TIME AVG. FORCE        719.    
 LARGEST RESIDUAL FORCE             1.080E+03   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.779E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.779E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.816       TIME AVG. MOMENT      0.117    
 LARGEST RESIDUAL MOMENT             1.94       AT NODE       1686   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.529E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.529E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.215E+03   TIME AVG. FORCE        708.    
 LARGEST RESIDUAL FORCE              53.1       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.704E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.286E-04   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                     0.788       TIME AVG. MOMENT      0.115    
 LARGEST RESIDUAL MOMENT           -0.139       AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.268E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.275E-03   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.221E+03   TIME AVG. FORCE        708.    
 LARGEST RESIDUAL FORCE             0.111       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.693E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.797E-06   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.789       TIME AVG. MOMENT      0.115    
 LARGEST RESIDUAL MOMENT            2.669E-03   AT NODE        406   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.293E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.555E-05   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.221E+03   TIME AVG. FORCE        708.    
 LARGEST RESIDUAL FORCE            -4.839E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         9.693E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.549E-10   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                     0.789       TIME AVG. MOMENT      0.115    
 LARGEST RESIDUAL MOMENT           -1.572E-06   AT NODE      18976   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.294E-03   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.465E-08   AT NODE       1146   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     74.8    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    23 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          3.536    
     TOTAL ARC LENGTH    150.    , INCREMENT OF ARC LENGTH    49.9    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.028E+03   TIME AVG. FORCE        888.    
 LARGEST RESIDUAL FORCE             3.581E+03   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.511E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.511E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.05       TIME AVG. MOMENT      0.155    
 LARGEST RESIDUAL MOMENT            -1.69       AT NODE       1687   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.265E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.265E-03   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.780E+03   TIME AVG. FORCE        878.    
 LARGEST RESIDUAL FORCE              78.1       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.559E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.348E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.02       TIME AVG. MOMENT      0.153    
 LARGEST RESIDUAL MOMENT           -0.168       AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.676E-03   AT NODE         19   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.297E-03   AT NODE          5   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.782E+03   TIME AVG. FORCE        878.    
 LARGEST RESIDUAL FORCE            -3.791E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.559E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.643E-07   AT NODE       1528   DOF  1
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.02       TIME AVG. MOMENT      0.153    
 LARGEST RESIDUAL MOMENT            1.072E-03   AT NODE        406   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.679E-03   AT NODE         19   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.730E-06   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.782E+03   TIME AVG. FORCE        878.    
 LARGEST RESIDUAL FORCE            -2.436E-04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.559E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.285E-11   AT NODE      14952   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.02       TIME AVG. MOMENT      0.153    
 LARGEST RESIDUAL MOMENT           -6.782E-07   AT NODE      10482   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.679E-03   AT NODE         15   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.046E-08   AT NODE       1118   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     112.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    24 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.102    
     TOTAL ARC LENGTH    224.    , INCREMENT OF ARC LENGTH    74.8    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.340E+03   TIME AVG. FORCE       1.056E+03
 LARGEST RESIDUAL FORCE             1.057E+04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.359E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.359E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.33       TIME AVG. MOMENT      0.200    
 LARGEST RESIDUAL MOMENT            -1.32       AT NODE       1369   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.501E-03   AT NODE         15   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.501E-03   AT NODE         15   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.179E+03   TIME AVG. FORCE       1.050E+03
 LARGEST RESIDUAL FORCE              49.4       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.369E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.004E-04   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.31       TIME AVG. MOMENT      0.199    
 LARGEST RESIDUAL MOMENT           -0.136       AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.146E-03   AT NODE         19   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.685E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.179E+03   TIME AVG. FORCE       1.050E+03
 LARGEST RESIDUAL FORCE            -1.336E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.369E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.420E-07   AT NODE       8381   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.31       TIME AVG. MOMENT      0.199    
 LARGEST RESIDUAL MOMENT           -1.677E-04   AT NODE         11   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.146E-03   AT NODE         15   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.943E-06   AT NODE        405   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     168.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    25 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.517    
     TOTAL ARC LENGTH    337.    , INCREMENT OF ARC LENGTH    112.    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.591E+03   TIME AVG. FORCE       1.224E+03
 LARGEST RESIDUAL FORCE            -2.515E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.546E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.546E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.71       TIME AVG. MOMENT      0.258    
 LARGEST RESIDUAL MOMENT            -1.16       AT NODE       1687   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.699E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.699E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.504E+03   TIME AVG. FORCE       1.221E+03
 LARGEST RESIDUAL FORCE              22.4       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.523E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.425E-05   AT NODE       1210   DOF  1
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      1.70       TIME AVG. MOMENT      0.257    
 LARGEST RESIDUAL MOMENT           -9.887E-02   AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.822E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.660E-04   AT NODE          8   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.503E+03   TIME AVG. FORCE       1.221E+03
 LARGEST RESIDUAL FORCE            -2.345E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.523E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.408E-07   AT NODE      12345   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      1.70       TIME AVG. MOMENT      0.257    
 LARGEST RESIDUAL MOMENT           -9.680E-05   AT NODE        406   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.826E-03   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.029E-06   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     253.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    26 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          4.838    
     TOTAL ARC LENGTH    505.    , INCREMENT OF ARC LENGTH    168.    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.932E+03   TIME AVG. FORCE       1.396E+03
 LARGEST RESIDUAL FORCE            -5.729E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.237E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.237E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.28       TIME AVG. MOMENT      0.332    
 LARGEST RESIDUAL MOMENT            -1.27       AT NODE       1687   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.437E-03   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.437E-03   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.894E+03   TIME AVG. FORCE       1.394E+03
 LARGEST RESIDUAL FORCE             -7.52       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.162E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.723E-05   AT NODE       2809   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      2.28       TIME AVG. MOMENT      0.332    
 LARGEST RESIDUAL MOMENT           -7.221E-02   AT NODE        406   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.710E-03   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.460E-04   AT NODE          1   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.893E+03   TIME AVG. FORCE       1.394E+03
 LARGEST RESIDUAL FORCE            -1.988E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.159E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.175E-06   AT NODE       3554   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      2.28       TIME AVG. MOMENT      0.332    
 LARGEST RESIDUAL MOMENT           -7.522E-04   AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.738E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.816E-05   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     379.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    27 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.124    
     TOTAL ARC LENGTH    758.    , INCREMENT OF ARC LENGTH    253.    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.496E+03   TIME AVG. FORCE       1.576E+03
 LARGEST RESIDUAL FORCE            -1.297E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.610E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.610E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.17       TIME AVG. MOMENT      0.433    
 LARGEST RESIDUAL MOMENT             1.92       AT NODE      23996   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.207E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.207E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.526E+03   TIME AVG. FORCE       1.577E+03
 LARGEST RESIDUAL FORCE             -115.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.456E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.538E-04   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      3.19       TIME AVG. MOMENT      0.434    
 LARGEST RESIDUAL MOMENT           -5.606E-02   AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.258E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.569E-04   AT NODE        229   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.524E+03   TIME AVG. FORCE       1.577E+03
 LARGEST RESIDUAL FORCE             0.592       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.438E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.870E-05   AT NODE      12196   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      3.20       TIME AVG. MOMENT      0.434    
 LARGEST RESIDUAL MOMENT           -3.945E-03   AT NODE        406   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.271E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.305E-04   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.524E+03   TIME AVG. FORCE       1.577E+03
 LARGEST RESIDUAL FORCE            -6.330E-04   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         7.438E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.867E-09   AT NODE      12345   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      3.20       TIME AVG. MOMENT      0.434    
 LARGEST RESIDUAL MOMENT            3.301E-06   AT NODE      14804   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.271E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.044E-07   AT NODE       1146   DOF  5
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     568.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    28 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.436    
     TOTAL ARC LENGTH   1.136E+03, INCREMENT OF ARC LENGTH    379.    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.446E+03   TIME AVG. FORCE       1.780E+03
 LARGEST RESIDUAL FORCE            -2.968E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.090E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.090E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      4.64       TIME AVG. MOMENT      0.579    
 LARGEST RESIDUAL MOMENT            -3.86       AT NODE      25476   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.997E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.997E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.570E+03   TIME AVG. FORCE       1.784E+03
 LARGEST RESIDUAL FORCE            -1.052E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.068E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.759E-04   AT NODE      12444   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      4.71       TIME AVG. MOMENT      0.582    
 LARGEST RESIDUAL MOMENT           -4.858E-02   AT NODE        406   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.064E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.920E-03   AT NODE        888   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.556E+03   TIME AVG. FORCE       1.784E+03
 LARGEST RESIDUAL FORCE              33.0       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.061E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.132E-04   AT NODE       4043   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      4.73       TIME AVG. MOMENT      0.582    
 LARGEST RESIDUAL MOMENT           -1.762E-02   AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.102E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.060E-04   AT NODE        230   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.556E+03   TIME AVG. FORCE       1.783E+03
 LARGEST RESIDUAL FORCE             1.906E-03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.061E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.490E-07   AT NODE       4172   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      4.73       TIME AVG. MOMENT      0.582    
 LARGEST RESIDUAL MOMENT            3.106E-05   AT NODE          8   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.102E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.942E-06   AT NODE        251   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     852.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    29 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          5.851    
     TOTAL ARC LENGTH   1.704E+03, INCREMENT OF ARC LENGTH    568.    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.010E+03   TIME AVG. FORCE       2.024E+03
 LARGEST RESIDUAL FORCE            -7.284E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.571E-02   AT NODE      21285   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.571E-02   AT NODE      21285   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.27       TIME AVG. MOMENT      0.805    
 LARGEST RESIDUAL MOMENT            -6.68       AT NODE      25451   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.317E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.317E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.187E+03   TIME AVG. FORCE       2.030E+03
 LARGEST RESIDUAL FORCE            -1.694E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.546E-02   AT NODE      21285   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.087E-03   AT NODE       3944   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.51       TIME AVG. MOMENT      0.813    
 LARGEST RESIDUAL MOMENT            0.397       AT NODE      14678   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.452E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.973E-03   AT NODE        332   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.098E+03   TIME AVG. FORCE       2.027E+03
 LARGEST RESIDUAL FORCE            -1.220E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.582E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.404E-03   AT NODE       4299   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.71       TIME AVG. MOMENT      0.820    
 LARGEST RESIDUAL MOMENT           -0.207       AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.679E-02   AT NODE        860   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.878E-03   AT NODE       4162   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.144E+03   TIME AVG. FORCE       2.029E+03
 LARGEST RESIDUAL FORCE             -958.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.597E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -6.602E-04   AT NODE       4150   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.71       TIME AVG. MOMENT      0.820    
 LARGEST RESIDUAL MOMENT           -5.552E-02   AT NODE          7   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.805E-02   AT NODE        861   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.719E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      9.139E+03   TIME AVG. FORCE       2.029E+03
 LARGEST RESIDUAL FORCE             -60.1       AT NODE        273   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.589E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.748E-04   AT NODE       4299   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      7.69       TIME AVG. MOMENT      0.819    
 LARGEST RESIDUAL MOMENT            1.094E-02   AT NODE          7   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.690E-02   AT NODE        861   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.628E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      9.141E+03   TIME AVG. FORCE       2.029E+03
 LARGEST RESIDUAL FORCE             0.468       AT NODE        836   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.590E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.502E-05   AT NODE       4150   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      7.69       TIME AVG. MOMENT      0.819    
 LARGEST RESIDUAL MOMENT            1.192E-03   AT NODE          2   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.698E-02   AT NODE        861   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.438E-04   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    30 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          6.417    
     TOTAL ARC LENGTH   2.557E+03, INCREMENT OF ARC LENGTH    852.    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.056E+04   TIME AVG. FORCE       2.304E+03
 LARGEST RESIDUAL FORCE            -1.556E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.507E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.507E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.0       TIME AVG. MOMENT       1.18    
 LARGEST RESIDUAL MOMENT             23.7       AT NODE      25605   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.923E-02   AT NODE        780   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.923E-02   AT NODE        780   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.067E+04   TIME AVG. FORCE       2.307E+03
 LARGEST RESIDUAL FORCE             2.878E+03   AT NODE        814   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.540E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.905E-03   AT NODE       4129   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      12.3       TIME AVG. MOMENT       1.19    
 LARGEST RESIDUAL MOMENT            -1.95       AT NODE       6087   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.318E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.181E-02   AT NODE       4139   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.067E+04   TIME AVG. FORCE       2.308E+03
 LARGEST RESIDUAL FORCE             3.656E+04   AT NODE        818   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.495E-02   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.657E-03   AT NODE      12958   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      13.6       TIME AVG. MOMENT       1.23    
 LARGEST RESIDUAL MOMENT            -6.21       AT NODE      14746   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.293E-02   AT NODE        851   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.300E-02   AT NODE      12915   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.067E+04   TIME AVG. FORCE       2.307E+03
 LARGEST RESIDUAL FORCE            -9.271E+04   AT NODE        273   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.790E-02   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.097E-02   AT NODE       4299   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.6       TIME AVG. MOMENT       1.26    
 LARGEST RESIDUAL MOMENT             22.9       AT NODE       4281   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.988E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.393E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    31 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.433E+03   TIME AVG. FORCE       2.268E+03
 LARGEST RESIDUAL FORCE            -9.751E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.768E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.768E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.65       TIME AVG. MOMENT       1.07    
 LARGEST RESIDUAL MOMENT             1.48       AT NODE      25605   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.229E-02   AT NODE        329   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.229E-02   AT NODE        329   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.479E+03   TIME AVG. FORCE       2.269E+03
 LARGEST RESIDUAL FORCE             -43.8       AT NODE        273   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.747E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.405E-04   AT NODE      12792   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      8.64       TIME AVG. MOMENT       1.07    
 LARGEST RESIDUAL MOMENT           -2.264E-02   AT NODE      14729   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.188E-02   AT NODE        780   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.400E-03   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.479E+03   TIME AVG. FORCE       2.269E+03
 LARGEST RESIDUAL FORCE              4.39       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.752E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.756E-05   AT NODE       4299   DOF  3
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF DISP.    CORRECTION   -5.767E-06
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      8.63       TIME AVG. MOMENT       1.07    
 LARGEST RESIDUAL MOMENT            1.211E-03   AT NODE        940   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.189E-02   AT NODE        781   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.026E-04   AT NODE       4160   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    2.154E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    31 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          6.543    
     TOTAL ARC LENGTH   2.770E+03, INCREMENT OF ARC LENGTH    213.    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.812E+03   TIME AVG. FORCE       2.505E+03
 LARGEST RESIDUAL FORCE            -9.256E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.735E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.735E-03   AT NODE      16964   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.64       TIME AVG. MOMENT       1.34    
 LARGEST RESIDUAL MOMENT             1.29       AT NODE      25605   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.156E-02   AT NODE        782   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.156E-02   AT NODE        782   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.844E+03   TIME AVG. FORCE       2.506E+03
 LARGEST RESIDUAL FORCE              23.3       AT NODE        871   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.744E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.790E-04   AT NODE      12826   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      9.63       TIME AVG. MOMENT       1.34    
 LARGEST RESIDUAL MOMENT            8.210E-03   AT NODE      14767   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.134E-02   AT NODE        783   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.015E-03   AT NODE        801   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.843E+03   TIME AVG. FORCE       2.506E+03
 LARGEST RESIDUAL FORCE             -5.75       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.746E-03   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.521E-05   AT NODE       4262   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      9.63       TIME AVG. MOMENT       1.34    
 LARGEST RESIDUAL MOMENT           -1.202E-03   AT NODE        586   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.134E-02   AT NODE        324   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.367E-04   AT NODE       4325   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -3.465E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     320.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    32 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          6.664    
     TOTAL ARC LENGTH   2.983E+03, INCREMENT OF ARC LENGTH    213.    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.037E+04   TIME AVG. FORCE       2.744E+03
 LARGEST RESIDUAL FORCE            -2.008E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.661E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.661E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      11.2       TIME AVG. MOMENT       1.64    
 LARGEST RESIDUAL MOMENT            -2.59       AT NODE      24110   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.757E-02   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.757E-02   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.042E+04   TIME AVG. FORCE       2.746E+03
 LARGEST RESIDUAL FORCE             -93.6       AT NODE        304   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.718E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.388E-04   AT NODE      12823   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      11.2       TIME AVG. MOMENT       1.64    
 LARGEST RESIDUAL MOMENT            5.338E-02   AT NODE      14763   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.852E-02   AT NODE        311   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.298E-03   AT NODE      12962   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.042E+04   TIME AVG. FORCE       2.746E+03
 LARGEST RESIDUAL FORCE              20.8       AT NODE        850   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.738E-03   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.529E-04   AT NODE      12927   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      11.2       TIME AVG. MOMENT       1.64    
 LARGEST RESIDUAL MOMENT           -3.102E-03   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.939E-02   AT NODE        311   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.139E-03   AT NODE      12917   DOF  5
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -6.620E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.042E+04   TIME AVG. FORCE       2.746E+03
 LARGEST RESIDUAL FORCE            -1.071E-02   AT NODE        309   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.738E-03   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.864E-06   AT NODE      12757   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      11.2       TIME AVG. MOMENT       1.64    
 LARGEST RESIDUAL MOMENT           -4.188E-05   AT NODE        554   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.937E-02   AT NODE        311   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.188E-05   AT NODE        872   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     479.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    33 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          6.835    
     TOTAL ARC LENGTH   3.302E+03, INCREMENT OF ARC LENGTH    320.    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.119E+04   TIME AVG. FORCE       2.994E+03
 LARGEST RESIDUAL FORCE            -4.748E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.709E-03   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.709E-03   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      13.9       TIME AVG. MOMENT       2.00    
 LARGEST RESIDUAL MOMENT             5.24       AT NODE      25600   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.142E-02   AT NODE        800   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.142E-02   AT NODE        800   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.123E+04   TIME AVG. FORCE       2.995E+03
 LARGEST RESIDUAL FORCE              420.       AT NODE        834   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.812E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.271E-04   AT NODE       4150   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.0       TIME AVG. MOMENT       2.00    
 LARGEST RESIDUAL MOMENT           -0.157       AT NODE       6059   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.153E-02   AT NODE        802   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.462E-03   AT NODE       4284   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.124E+04   TIME AVG. FORCE       2.995E+03
 LARGEST RESIDUAL FORCE             -173.       AT NODE        259   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.851E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.639E-04   AT NODE       4313   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      14.1       TIME AVG. MOMENT       2.00    
 LARGEST RESIDUAL MOMENT            9.271E-03   AT NODE       5659   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.239E-02   AT NODE        802   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.776E-03   AT NODE        259   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION   -1.635E-04
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.124E+04   TIME AVG. FORCE       2.995E+03
 LARGEST RESIDUAL FORCE             7.292E-02   AT NODE        836   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.852E-03   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.596E-06   AT NODE       4150   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      14.1       TIME AVG. MOMENT       2.00    
 LARGEST RESIDUAL MOMENT           -3.976E-04   AT NODE         21   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.238E-02   AT NODE        802   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.670E-05   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     719.    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    34 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          7.067    
     TOTAL ARC LENGTH   3.782E+03, INCREMENT OF ARC LENGTH    479.    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.242E+04   TIME AVG. FORCE       3.265E+03
 LARGEST RESIDUAL FORCE            -1.094E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.345E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.345E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      19.3       TIME AVG. MOMENT       2.50    
 LARGEST RESIDUAL MOMENT             13.0       AT NODE       6055   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.892E-02   AT NODE        305   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.892E-02   AT NODE        305   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.248E+04   TIME AVG. FORCE       3.266E+03
 LARGEST RESIDUAL FORCE             -584.       AT NODE        276   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.343E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.029E-04   AT NODE       4299   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      19.3       TIME AVG. MOMENT       2.50    
 LARGEST RESIDUAL MOMENT            0.669       AT NODE       6121   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.733E-02   AT NODE        241   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.079E-03   AT NODE        833   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.250E+04   TIME AVG. FORCE       3.267E+03
 LARGEST RESIDUAL FORCE             -195.       AT NODE        287   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.346E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.193E-04   AT NODE       3987   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      19.4       TIME AVG. MOMENT       2.50    
 LARGEST RESIDUAL MOMENT            2.176E-02   AT NODE      14745   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.605E-02   AT NODE        242   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.951E-03   AT NODE        287   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.250E+04   TIME AVG. FORCE       3.267E+03
 LARGEST RESIDUAL FORCE             0.280       AT NODE        826   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.346E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.995E-05   AT NODE       4160   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      19.4       TIME AVG. MOMENT       2.50    
 LARGEST RESIDUAL MOMENT           -6.530E-04   AT NODE        819   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.602E-02   AT NODE        242   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.112E-04   AT NODE        263   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.079E+03

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    35 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          7.380    
     TOTAL ARC LENGTH   4.501E+03, INCREMENT OF ARC LENGTH    719.    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.431E+04   TIME AVG. FORCE       3.573E+03
 LARGEST RESIDUAL FORCE            -2.287E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.994E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.994E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.5       TIME AVG. MOMENT       3.19    
 LARGEST RESIDUAL MOMENT             27.8       AT NODE       6056   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.371E-02   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.371E-02   AT NODE        304   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.435E+04   TIME AVG. FORCE       3.575E+03
 LARGEST RESIDUAL FORCE             2.507E+03   AT NODE        836   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.973E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.441E-03   AT NODE       4448   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.2       TIME AVG. MOMENT       3.19    
 LARGEST RESIDUAL MOMENT            -1.39       AT NODE      12774   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.206E-02   AT NODE        243   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.059E-02   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.442E+04   TIME AVG. FORCE       3.577E+03
 LARGEST RESIDUAL FORCE              842.       AT NODE        819   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.962E-02   AT NODE       8024   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.010E-04   AT NODE       4431   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.0       TIME AVG. MOMENT       3.18    
 LARGEST RESIDUAL MOMENT            0.144       AT NODE      14747   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.088E-02   AT NODE        865   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.413E-03   AT NODE       4440   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.443E+04   TIME AVG. FORCE       3.577E+03
 LARGEST RESIDUAL FORCE             -23.3       AT NODE        273   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.960E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.706E-04   AT NODE       4150   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      27.0       TIME AVG. MOMENT       3.18    
 LARGEST RESIDUAL MOMENT           -6.209E-03   AT NODE       4461   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.063E-02   AT NODE        303   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.014E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    4.358E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.443E+04   TIME AVG. FORCE       3.577E+03
 LARGEST RESIDUAL FORCE            -1.681E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.960E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.964E-06   AT NODE      12657   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      27.0       TIME AVG. MOMENT       3.18    
 LARGEST RESIDUAL MOMENT            6.690E-05   AT NODE      10864   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.063E-02   AT NODE        806   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.380E-05   AT NODE        821   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    36 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          7.855    
     TOTAL ARC LENGTH   5.579E+03, INCREMENT OF ARC LENGTH   1.079E+03


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.617E+04   TIME AVG. FORCE       3.917E+03
 LARGEST RESIDUAL FORCE            -2.248E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.942E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.942E-02   AT NODE      21583   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.4       TIME AVG. MOMENT       4.02    
 LARGEST RESIDUAL MOMENT             26.0       AT NODE      14662   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.964E-02   AT NODE        245   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.964E-02   AT NODE        245   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.644E+04   TIME AVG. FORCE       3.925E+03
 LARGEST RESIDUAL FORCE             -361.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.933E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.590E-04   AT NODE      12929   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.3       TIME AVG. MOMENT       4.02    
 LARGEST RESIDUAL MOMENT           -0.457       AT NODE       4168   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.981E-02   AT NODE        808   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.314E-03   AT NODE      12623   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.645E+04   TIME AVG. FORCE       3.925E+03
 LARGEST RESIDUAL FORCE             -67.2       AT NODE        255   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.931E-02   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.147E-04   AT NODE       4432   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      34.2       TIME AVG. MOMENT       4.02    
 LARGEST RESIDUAL MOMENT            2.108E-02   AT NODE        732   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.995E-02   AT NODE        863   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.751E-03   AT NODE        855   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.645E+04   TIME AVG. FORCE       3.925E+03
 LARGEST RESIDUAL FORCE            -1.159E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.931E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.393E-06   AT NODE      12792   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      34.2       TIME AVG. MOMENT       4.02    
 LARGEST RESIDUAL MOMENT           -6.650E-04   AT NODE        292   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.994E-02   AT NODE        863   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.057E-05   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    37 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          8.364    
     TOTAL ARC LENGTH   6.658E+03, INCREMENT OF ARC LENGTH   1.079E+03


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.818E+04   TIME AVG. FORCE       4.300E+03
 LARGEST RESIDUAL FORCE            -2.217E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.921E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.921E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      41.5       TIME AVG. MOMENT       5.00    
 LARGEST RESIDUAL MOMENT            -32.3       AT NODE       6019   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.071E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -6.071E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.851E+04   TIME AVG. FORCE       4.309E+03
 LARGEST RESIDUAL FORCE             -686.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.914E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.004E-04   AT NODE       5929   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      41.4       TIME AVG. MOMENT       5.00    
 LARGEST RESIDUAL MOMENT            0.515       AT NODE        251   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -6.143E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.107E-03   AT NODE        195   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.851E+04   TIME AVG. FORCE       4.309E+03
 LARGEST RESIDUAL FORCE             -48.7       AT NODE        273   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.914E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.832E-04   AT NODE       4597   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      41.4       TIME AVG. MOMENT       5.00    
 LARGEST RESIDUAL MOMENT           -1.453E-02   AT NODE          8   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.172E-02   AT NODE        810   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.490E-03   AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    4.202E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.851E+04   TIME AVG. FORCE       4.309E+03
 LARGEST RESIDUAL FORCE            -6.059E-03   AT NODE       6108   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.914E-02   AT NODE       7875   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.361E-07   AT NODE       5065   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      41.4       TIME AVG. MOMENT       5.00    
 LARGEST RESIDUAL MOMENT           -3.727E-04   AT NODE        292   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      6.172E-02   AT NODE        810   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.026E-06   AT NODE        252   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.618E+03

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    38 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          8.897    
     TOTAL ARC LENGTH   7.737E+03, INCREMENT OF ARC LENGTH   1.079E+03


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.172E+04   TIME AVG. FORCE       4.755E+03
 LARGEST RESIDUAL FORCE            -4.870E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.860E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.860E-02   AT NODE      21732   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      52.3       TIME AVG. MOMENT       6.21    
 LARGEST RESIDUAL MOMENT             85.9       AT NODE       6112   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.441E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.441E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.175E+04   TIME AVG. FORCE       4.756E+03
 LARGEST RESIDUAL FORCE             5.560E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.847E-02   AT NODE      21881   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.733E-04   AT NODE      12643   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      52.2       TIME AVG. MOMENT       6.21    
 LARGEST RESIDUAL MOMENT            -1.77       AT NODE        294   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.585E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.061E-03   AT NODE      12636   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.171E+04   TIME AVG. FORCE       4.755E+03
 LARGEST RESIDUAL FORCE             1.579E+03   AT NODE        836   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.855E-02   AT NODE      21881   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.492E-04   AT NODE      13388   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      52.2       TIME AVG. MOMENT       6.21    
 LARGEST RESIDUAL MOMENT           -0.159       AT NODE      14139   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.650E-02   AT NODE        812   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.554E-03   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.171E+04   TIME AVG. FORCE       4.755E+03
 LARGEST RESIDUAL FORCE            -0.471       AT NODE      13833   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.855E-02   AT NODE      21881   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.818E-05   AT NODE      11898   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      52.2       TIME AVG. MOMENT       6.21    
 LARGEST RESIDUAL MOMENT           -1.343E-02   AT NODE      14729   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -9.644E-02   AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.210E-04   AT NODE      12941   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    2.427E+03

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    39 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          9.727    
     TOTAL ARC LENGTH   9.355E+03, INCREMENT OF ARC LENGTH   1.618E+03


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.884E+04   TIME AVG. FORCE       5.357E+03
 LARGEST RESIDUAL FORCE            -1.041E+07   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.259E-02   AT NODE       7577   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.259E-02   AT NODE       7577   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      68.9       TIME AVG. MOMENT       7.78    
 LARGEST RESIDUAL MOMENT             301.       AT NODE       6063   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.148       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.148       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.680E+04   TIME AVG. FORCE       5.306E+03
 LARGEST RESIDUAL FORCE             3.501E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.224E-02   AT NODE       7577   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.588E-03   AT NODE       5929   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      68.8       TIME AVG. MOMENT       7.78    
 LARGEST RESIDUAL MOMENT            -10.5       AT NODE      14704   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.149       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.238E-03   AT NODE        288   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.670E+04   TIME AVG. FORCE       5.304E+03
 LARGEST RESIDUAL FORCE             4.999E+03   AT NODE        836   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.245E-02   AT NODE      22179   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.245E-03   AT NODE      13686   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      68.8       TIME AVG. MOMENT       7.78    
 LARGEST RESIDUAL MOMENT            0.816       AT NODE       6087   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.146       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.561E-02   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.670E+04   TIME AVG. FORCE       5.304E+03
 LARGEST RESIDUAL FORCE             -9.28       AT NODE        273   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.243E-02   AT NODE      22179   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        9.376E-05   AT NODE      12047   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      68.8       TIME AVG. MOMENT       7.78    
 LARGEST RESIDUAL MOMENT           -0.104       AT NODE      14729   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.146       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.009E-03   AT NODE      12941   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.670E+04   TIME AVG. FORCE       5.304E+03
 LARGEST RESIDUAL FORCE             0.139       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.243E-02   AT NODE       7428   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.725E-06   AT NODE       5491   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      68.8       TIME AVG. MOMENT       7.78    
 LARGEST RESIDUAL MOMENT            1.115E-03   AT NODE      14583   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.146       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.226E-05   AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    40 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          10.98    
     TOTAL ARC LENGTH   1.178E+04, INCREMENT OF ARC LENGTH   2.427E+03


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.406E+04   TIME AVG. FORCE       6.005E+03
 LARGEST RESIDUAL FORCE            -9.570E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.221E-02   AT NODE      22328   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.221E-02   AT NODE      22328   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      85.9       TIME AVG. MOMENT       9.68    
 LARGEST RESIDUAL MOMENT             280.       AT NODE       6063   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.144       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.144       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.214E+04   TIME AVG. FORCE       5.958E+03
 LARGEST RESIDUAL FORCE             3.155E+04   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.170E-02   AT NODE      22328   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -5.099E-03   AT NODE      12345   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      85.6       TIME AVG. MOMENT       9.67    
 LARGEST RESIDUAL MOMENT            -29.8       AT NODE      14729   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.147       AT NODE        296   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.335E-02   AT NODE      12191   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.176E+04   TIME AVG. FORCE       5.949E+03
 LARGEST RESIDUAL FORCE            -5.128E+04   AT NODE        291   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.157E-02   AT NODE       7130   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.733E-03   AT NODE       4001   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      85.9       TIME AVG. MOMENT       9.68    
 LARGEST RESIDUAL MOMENT            -23.0       AT NODE       3549   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.148       AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.188E-02   AT NODE        292   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.192E+04   TIME AVG. FORCE       5.953E+03
 LARGEST RESIDUAL FORCE             2.796E+03   AT NODE      12645   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.173E-02   AT NODE       7130   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.903E-03   AT NODE      12792   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      85.6       TIME AVG. MOMENT       9.67    
 LARGEST RESIDUAL MOMENT            -5.45       AT NODE      12649   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.138       AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.649E-02   AT NODE       4007   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.194E+04   TIME AVG. FORCE       5.953E+03
 LARGEST RESIDUAL FORCE            -5.236E+03   AT NODE        292   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.162E-02   AT NODE      22477   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.916E-03   AT NODE      11898   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      85.5       TIME AVG. MOMENT       9.67    
 LARGEST RESIDUAL MOMENT             2.74       AT NODE      14748   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.140       AT NODE        812   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.665E-02   AT NODE        817   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      3.195E+04   TIME AVG. FORCE       5.953E+03
 LARGEST RESIDUAL FORCE              36.3       AT NODE        818   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.161E-02   AT NODE      22328   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.039E-05   AT NODE      12941   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      85.5       TIME AVG. MOMENT       9.67    
 LARGEST RESIDUAL MOMENT            0.197       AT NODE        254   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.140       AT NODE        297   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.354E-03   AT NODE        818   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      3.195E+04   TIME AVG. FORCE       5.954E+03
 LARGEST RESIDUAL FORCE              1.23       AT NODE       5921   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.161E-02   AT NODE      22328   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.205E-05   AT NODE      14002   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      85.5       TIME AVG. MOMENT       9.67    
 LARGEST RESIDUAL MOMENT            9.688E-03   AT NODE      14595   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.140       AT NODE        812   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.209E-04   AT NODE        818   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    41 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          12.16    
     TOTAL ARC LENGTH   1.421E+04, INCREMENT OF ARC LENGTH   2.427E+03


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.937E+04   TIME AVG. FORCE       6.749E+03
 LARGEST RESIDUAL FORCE            -8.703E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.169E-02   AT NODE       7130   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        4.169E-02   AT NODE       7130   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      103.       TIME AVG. MOMENT       11.9    
 LARGEST RESIDUAL MOMENT             283.       AT NODE       5323   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.139       AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     0.139       AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.666E+04   TIME AVG. FORCE       6.685E+03
 LARGEST RESIDUAL FORCE             2.199E+05   AT NODE       3552   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         4.181E-02   AT NODE       6981   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.083E-02   AT NODE       3554   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      104.       TIME AVG. MOMENT       11.9    
 LARGEST RESIDUAL MOMENT            -283.       AT NODE       3103   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.169       AT NODE        273   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.131       AT NODE       3559   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.679E+04   TIME AVG. FORCE       6.688E+03
 LARGEST RESIDUAL FORCE             4.746E+05   AT NODE        816   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.802E-02   AT NODE      11749   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.804E-02   AT NODE      11600   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      106.       TIME AVG. MOMENT       12.0    
 LARGEST RESIDUAL MOMENT            -301.       AT NODE      14727   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.318       AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.162       AT NODE        816   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.612E+04   TIME AVG. FORCE       6.672E+03
 LARGEST RESIDUAL FORCE             2.660E+04   AT NODE      11750   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.679E-02   AT NODE      11898   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.642E-03   AT NODE       2676   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      105.       TIME AVG. MOMENT       11.9    
 LARGEST RESIDUAL MOMENT            -62.2       AT NODE      14707   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.321       AT NODE        836   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.945E-02   AT NODE      11900   DOF  5
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.989E+04   TIME AVG. FORCE       6.762E+03
 LARGEST RESIDUAL FORCE             3.017E+06   AT NODE        815   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         6.056E-02   AT NODE      12047   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.098E-02   AT NODE      14005   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      109.       TIME AVG. MOMENT       12.0    
 LARGEST RESIDUAL MOMENT            4.250E+03   AT NODE      14750   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -0.558       AT NODE        816   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -0.423       AT NODE        815   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    42 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.305E+04   TIME AVG. FORCE       6.599E+03
 LARGEST RESIDUAL FORCE            -5.588E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.042E-02   AT NODE      22477   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.042E-02   AT NODE      22477   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      89.7       TIME AVG. MOMENT       11.6    
 LARGEST RESIDUAL MOMENT             17.6       AT NODE      14658   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.487E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.487E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.329E+04   TIME AVG. FORCE       6.604E+03
 LARGEST RESIDUAL FORCE              133.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.041E-02   AT NODE       7130   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.336E-04   AT NODE       5658   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      89.7       TIME AVG. MOMENT       11.6    
 LARGEST RESIDUAL MOMENT            0.573       AT NODE        202   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.490E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.171E-03   AT NODE        254   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.329E+04   TIME AVG. FORCE       6.604E+03
 LARGEST RESIDUAL FORCE             -4.40       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.041E-02   AT NODE      22477   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.220E-05   AT NODE      13965   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      89.7       TIME AVG. MOMENT       11.6    
 LARGEST RESIDUAL MOMENT           -2.345E-02   AT NODE       6109   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.480E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.904E-04   AT NODE        855   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    1.599E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    42 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          12.44    
     TOTAL ARC LENGTH   1.481E+04, INCREMENT OF ARC LENGTH    607.    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.442E+04   TIME AVG. FORCE       7.251E+03
 LARGEST RESIDUAL FORCE            -5.439E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.040E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.040E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      94.0       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT             17.4       AT NODE       6158   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.481E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.481E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.467E+04   TIME AVG. FORCE       7.257E+03
 LARGEST RESIDUAL FORCE             -130.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.038E-02   AT NODE       6981   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.425E-04   AT NODE       5658   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      94.0       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT           -0.613       AT NODE        907   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.484E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.472E-03   AT NODE        254   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.467E+04   TIME AVG. FORCE       7.257E+03
 LARGEST RESIDUAL FORCE              4.96       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.039E-02   AT NODE       6981   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.745E-05   AT NODE       5361   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      94.0       TIME AVG. MOMENT       13.5    
 LARGEST RESIDUAL MOMENT           -3.334E-02   AT NODE       6109   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.474E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.043E-04   AT NODE        254   DOF  4
   INSTANCE: PANEL-1                                                                         
 ESTIMATE OF ROTATION CORRECTION    2.743E-05
 MOMENT    EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     910.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    43 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          12.71    
     TOTAL ARC LENGTH   1.542E+04, INCREMENT OF ARC LENGTH    607.    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.630E+04   TIME AVG. FORCE       7.917E+03
 LARGEST RESIDUAL FORCE            -1.183E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.557E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.557E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      100.       TIME AVG. MOMENT       15.5    
 LARGEST RESIDUAL MOMENT            -42.3       AT NODE      13964   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.212E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.212E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.671E+04   TIME AVG. FORCE       7.927E+03
 LARGEST RESIDUAL FORCE              537.       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.554E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.185E-04   AT NODE       5808   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      100.       TIME AVG. MOMENT       15.5    
 LARGEST RESIDUAL MOMENT             1.57       AT NODE       6087   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      5.219E-02   AT NODE        811   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.950E-03   AT NODE        253   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.672E+04   TIME AVG. FORCE       7.927E+03
 LARGEST RESIDUAL FORCE              126.       AT NODE        856   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.555E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.332E-04   AT NODE       5510   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      100.       TIME AVG. MOMENT       15.5    
 LARGEST RESIDUAL MOMENT           -0.197       AT NODE       6109   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.193E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.674E-03   AT NODE        253   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.672E+04   TIME AVG. FORCE       7.927E+03
 LARGEST RESIDUAL FORCE             0.536       AT NODE       5808   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.555E-02   AT NODE      22626   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.386E-06   AT NODE      14113   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      100.       TIME AVG. MOMENT       15.5    
 LARGEST RESIDUAL MOMENT            7.508E-03   AT NODE        253   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -5.193E-02   AT NODE        298   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.133E-05   AT NODE        253   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO    1.365E+03

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    44 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.09    
     TOTAL ARC LENGTH   1.633E+04, INCREMENT OF ARC LENGTH    910.    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.928E+04   TIME AVG. FORCE       8.623E+03
 LARGEST RESIDUAL FORCE            -2.518E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.329E-02   AT NODE       6981   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.329E-02   AT NODE       6981   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       17.6    
 LARGEST RESIDUAL MOMENT            -113.       AT NODE      13963   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.812E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -7.812E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.60000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.60000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.973E+04   TIME AVG. FORCE       8.633E+03
 LARGEST RESIDUAL FORCE             2.265E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.321E-02   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        6.970E-04   AT NODE      14411   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       17.6    
 LARGEST RESIDUAL MOMENT             4.36       AT NODE       6087   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.845E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.517E-03   AT NODE        856   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.973E+04   TIME AVG. FORCE       8.633E+03
 LARGEST RESIDUAL FORCE            -1.993E+03   AT NODE        253   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.327E-02   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -4.834E-04   AT NODE       5511   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       17.6    
 LARGEST RESIDUAL MOMENT            -1.11       AT NODE       6106   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.874E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.083E-02   AT NODE        253   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.973E+04   TIME AVG. FORCE       8.633E+03
 LARGEST RESIDUAL FORCE              10.6       AT NODE        815   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.327E-02   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -3.967E-05   AT NODE       5511   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       17.6    
 LARGEST RESIDUAL MOMENT            0.140       AT NODE        856   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      7.875E-02   AT NODE        862   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     8.649E-04   AT NODE        253   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.973E+04   TIME AVG. FORCE       8.633E+03
 LARGEST RESIDUAL FORCE             0.502       AT NODE      14751   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.327E-02   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.820E-06   AT NODE      14113   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       17.6    
 LARGEST RESIDUAL MOMENT            5.237E-03   AT NODE      14706   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -7.875E-02   AT NODE        247   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.574E-05   AT NODE        857   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    45 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.58    
     TOTAL ARC LENGTH   1.770E+04, INCREMENT OF ARC LENGTH   1.365E+03


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.220E+04   TIME AVG. FORCE       9.363E+03
 LARGEST RESIDUAL FORCE            -2.317E+06   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.325E-02   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.325E-02   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      120.       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT             143.       AT NODE       5321   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.431E-02   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -8.431E-02   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.50000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.273E+04   TIME AVG. FORCE       9.375E+03
 LARGEST RESIDUAL FORCE             7.868E+03   AT NODE        871   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.317E-02   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.277E-03   AT NODE      14247   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      120.       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT            -8.77       AT NODE       5810   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -8.950E-02   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.539E-02   AT NODE        871   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.264E+04   TIME AVG. FORCE       9.373E+03
 LARGEST RESIDUAL FORCE             3.740E+04   AT NODE        872   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.326E-02   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        2.083E-03   AT NODE      14097   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      120.       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT            -14.5       AT NODE      14693   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      9.155E-02   AT NODE        875   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     5.454E-02   AT NODE        872   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.269E+04   TIME AVG. FORCE       9.374E+03
 LARGEST RESIDUAL FORCE            -5.570E+03   AT NODE        308   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.326E-02   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -9.152E-04   AT NODE       5456   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      120.       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT             6.09       AT NODE       6123   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      8.958E-02   AT NODE        877   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.208E-02   AT NODE        308   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.255E+04   TIME AVG. FORCE       9.371E+03
 LARGEST RESIDUAL FORCE             7.133E+04   AT NODE        802   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         2.329E-02   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.862E-03   AT NODE      14316   DOF  2
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      120.       TIME AVG. MOMENT       19.8    
 LARGEST RESIDUAL MOMENT            -38.9       AT NODE      14763   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      0.111       AT NODE        801   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     7.743E-02   AT NODE        802   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    46 STARTS. ATTEMPT NUMBER  2

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.039E+04   TIME AVG. FORCE       9.324E+03
 LARGEST RESIDUAL FORCE            -1.477E+05   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.814E-03   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.814E-03   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      113.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT            -9.28       AT NODE       6106   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.108E-02   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.108E-02   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.049E+04   TIME AVG. FORCE       9.326E+03
 LARGEST RESIDUAL FORCE              280.       AT NODE        871   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.811E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.798E-04   AT NODE      14098   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      113.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT            0.393       AT NODE       5810   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.239E-02   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -4.621E-03   AT NODE        871   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.30000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.050E+04   TIME AVG. FORCE       9.326E+03
 LARGEST RESIDUAL FORCE              408.       AT NODE        871   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.810E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.163E-04   AT NODE      14098   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      113.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT           -0.170       AT NODE      13647   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.403E-02   AT NODE        879   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -5.497E-03   AT NODE        871   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 5 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.054E+04   TIME AVG. FORCE       9.327E+03
 LARGEST RESIDUAL FORCE             1.951E+04   AT NODE        801   DOF  2
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.793E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.477E-03   AT NODE      14168   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      113.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT            -2.31       AT NODE      14767   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.034E-02   AT NODE      13721   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.736E-02   AT NODE        801   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    46 STARTS. ATTEMPT NUMBER  3

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.30000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.978E+04   TIME AVG. FORCE       9.311E+03
 LARGEST RESIDUAL FORCE              664.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.634E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.634E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT            0.173       AT NODE      14712   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.317E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.317E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.978E+04   TIME AVG. FORCE       9.311E+03
 LARGEST RESIDUAL FORCE             -1.08       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.633E-04   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -7.899E-07   AT NODE      14154   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT            3.498E-03   AT NODE       5957   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.314E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.086E-05   AT NODE        815   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.978E+04   TIME AVG. FORCE       9.311E+03
 LARGEST RESIDUAL FORCE             1.647E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.633E-04   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.829E-08   AT NODE       5621   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       19.6    
 LARGEST RESIDUAL MOMENT           -1.336E-04   AT NODE      14712   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.313E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -9.762E-07   AT NODE        292   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    46 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.59    
     TOTAL ARC LENGTH   1.778E+04, INCREMENT OF ARC LENGTH    85.3    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.983E+04   TIME AVG. FORCE       9.960E+03
 LARGEST RESIDUAL FORCE              652.       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.634E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.634E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       21.5    
 LARGEST RESIDUAL MOMENT           -0.173       AT NODE       6104   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -1.326E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.326E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.983E+04   TIME AVG. FORCE       9.960E+03
 LARGEST RESIDUAL FORCE              1.08       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.633E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -8.261E-07   AT NODE       5470   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       21.5    
 LARGEST RESIDUAL MOMENT           -3.496E-03   AT NODE      14561   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.322E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     2.164E-05   AT NODE        294   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.983E+04   TIME AVG. FORCE       9.960E+03
 LARGEST RESIDUAL FORCE            -1.634E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         3.633E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        3.724E-08   AT NODE      14301   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      110.       TIME AVG. MOMENT       21.5    
 LARGEST RESIDUAL MOMENT           -1.323E-04   AT NODE      14712   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      1.322E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     9.543E-07   AT NODE        817   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     128.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    47 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.60    
     TOTAL ARC LENGTH   1.787E+04, INCREMENT OF ARC LENGTH    85.3    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.991E+04   TIME AVG. FORCE       1.058E+04
 LARGEST RESIDUAL FORCE            -1.406E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.450E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.450E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       23.4    
 LARGEST RESIDUAL MOMENT           -0.258       AT NODE       6104   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.003E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.003E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.991E+04   TIME AVG. FORCE       1.058E+04
 LARGEST RESIDUAL FORCE              1.67       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.449E-04   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.344E-06   AT NODE      14154   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       23.4    
 LARGEST RESIDUAL MOMENT            5.273E-03   AT NODE       6064   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -2.001E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.600E-05   AT NODE        815   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.991E+04   TIME AVG. FORCE       1.058E+04
 LARGEST RESIDUAL FORCE             2.672E-02   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         5.449E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        5.265E-08   AT NODE       5659   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       23.4    
 LARGEST RESIDUAL MOMENT            2.091E-04   AT NODE       6104   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      2.001E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -1.394E-06   AT NODE        254   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     192.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    48 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.61    
     TOTAL ARC LENGTH   1.800E+04, INCREMENT OF ARC LENGTH    128.    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.002E+04   TIME AVG. FORCE       1.118E+04
 LARGEST RESIDUAL FORCE            -3.067E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.175E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        8.175E-04   AT NODE       6832   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       25.2    
 LARGEST RESIDUAL MOMENT            0.386       AT NODE       6068   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.040E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     3.040E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.002E+04   TIME AVG. FORCE       1.118E+04
 LARGEST RESIDUAL FORCE              2.63       AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.173E-04   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -2.600E-06   AT NODE       5526   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       25.2    
 LARGEST RESIDUAL MOMENT            9.565E-03   AT NODE       5810   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -3.051E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     6.379E-05   AT NODE        238   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.20000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.002E+04   TIME AVG. FORCE       1.118E+04
 LARGEST RESIDUAL FORCE            -4.501E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         8.173E-04   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        7.075E-08   AT NODE      14263   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       25.2    
 LARGEST RESIDUAL MOMENT            3.357E-04   AT NODE       6104   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      3.050E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     1.989E-06   AT NODE        855   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     288.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    49 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.62    
     TOTAL ARC LENGTH   1.819E+04, INCREMENT OF ARC LENGTH    192.    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.50000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.018E+04   TIME AVG. FORCE       1.176E+04
 LARGEST RESIDUAL FORCE            -6.745E+03   AT NODE          4   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.226E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.        1.226E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       26.9    
 LARGEST RESIDUAL MOMENT            0.706       AT NODE       6068   DOF  5
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.661E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION     4.661E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
          MOMENT    EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.019E+04   TIME AVG. FORCE       1.176E+04
 LARGEST RESIDUAL FORCE             -4.25       AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.226E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.111E-05   AT NODE      14098   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       26.9    
 LARGEST RESIDUAL MOMENT           -2.040E-02   AT NODE       5768   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION     -4.728E-03   AT NODE        231   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -2.814E-04   AT NODE        871   DOF  4
   INSTANCE: PANEL-1                                                                         
          ROTATION CORRECTION TOO LARGE COMPARED TO ROTATION INCREMENT

  NUMBER OF EQUATIONS =    158195     NUMBER OF RHS =    2
  NUMBER OF FLOATING PT. OPERATIONS =  1.72E+10

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.40000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.40000    
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.019E+04   TIME AVG. FORCE       1.176E+04
 LARGEST RESIDUAL FORCE            -7.873E-02   AT NODE          3   DOF  1
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF DISP.         1.226E-03   AT NODE      22775   DOF  3
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO DISP.       -1.845E-07   AT NODE      13947   DOF  3
   INSTANCE: PANEL-1                                                                         
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      111.       TIME AVG. MOMENT       26.9    
 LARGEST RESIDUAL MOMENT           -5.978E-04   AT NODE      14712   DOF  6
   INSTANCE: PANEL-1                                                                         
 LARGEST INCREMENT OF ROTATION      4.726E-03   AT NODE        878   DOF  4
   INSTANCE: PANEL-1                                                                         
 LARGEST CORRECTION TO ROTATION    -3.622E-06   AT NODE        874   DOF  4
   INSTANCE: PANEL-1                                                                         
          THE MOMENT    EQUILIBRIUM EQUATIONS HAVE CONVERGED
          ARC LENGTH INCREMENT MAY NOW INCREASE TO     432.    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.


                                INCREMENT    50 SUMMARY


     CURRENT LOAD PROPORTIONALITY FACTOR          13.64    
     TOTAL ARC LENGTH   1.848E+04, INCREMENT OF ARC LENGTH    288.    


     THE NUMBER OF INCS ON THE *STEP CARD HAS BEEN COMPLETED - THE STEP ENDS



     ANALYSIS SUMMARY:
     TOTAL OF         50  INCREMENTS
                       4  CUTBACKS IN AUTOMATIC INCREMENTATION
                     176  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     176  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     176  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       0  WARNING MESSAGES DURING USER INPUT PROCESSING
                      92  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                      92  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   88.200    
       SYSTEM TIME (SEC)    =   2.0000    
       TOTAL CPU TIME (SEC) =   90.200    
       WALLCLOCK TIME (SEC) =        271
