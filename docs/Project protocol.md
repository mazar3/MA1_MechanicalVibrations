**BRUFACE**
**Mechanical Vibrations**
**MA1 Mechanical Engineering**
**Academic year 2025-2026**

Arnaud Deraemaeker (Arnaud.Deraemaeker@ulb.be)
Wout Weijtjens (Wout.Weijtjens@vub.be)

# Satellite in microvibrations in orbit

## Model and equations

Consider a satellite (**Figure 1**) in orbit consisting of a main (rigid) body, two (flexible) solar arrays, a (rigid) deployable antenna, and a (flexible) deployable arm as depicted below. We wish to analyze the dynamic behavior of the satellite restricted to an (x,y) plane motion. The dynamic response will be investigated under excitation from the motion control devices on the main body (vertical force $F_c$ or torque $M_c$ applied to the main body).

A finite element model of this satellite has been built using Euler-Bernoulli beam elements for the solar arrays and the deployable arm, and point masses (including inertia) for the main body and deployable antenna (assumed to be rigid). The main properties are given in **Table 1**.

> **[Figure 1: Satellite drawing (left), degrees of freedom retained in the dynamic model]**

> **[Table 1: Parameters of the satellite]**

The model and its properties are taken from [1], the finite element model was built in the Structural Dynamics Toolbox running in Matlab. It has then been reduced to 10 degrees of freedom using Guyan static condensation [2]:

*   $x_c, y_c, \theta_c$, the main body translation $x$ and $y$ and rotation
*   $y_{12}$ and $y_2$, the vertical translation of the center and tip of the left solar panel
*   $y_{31}$ and $y_{41}$, the vertical translation of the center and tip of the right solar panel
*   $x_a, y_a, \theta_a$, the deployable antenna translation in $x$ and $y$ and rotation

The retained DOFs are depicted in Figure 1, the numbers refer to node numbers in the initial, full finite element model (not represented here). The resulting mass and stiffness matrices (10x10) matrices are provided, where the DOFs are sorted in the same order as listed above. At this stage, they should be considered as "granted" and the satellite can be dealt with as a 10 DOFs system.

## Frequency domain computations

In a first step, in order to grasp the dynamic behavior of the satellite, we will perform frequency domain computations both with the full model (10 DOFs) and in the modal basis.

1.  Compute the mode shapes and the natural frequencies of the satellite. You should have three modes with natural frequencies = 0 Hz. These are the so-called rigid body modes (there is no strain energy associated to these modes). What do they represent physically and why do they appear for this specific system?
2.  Draw a schematic and give a physical interpretation of the first 5 flexible modes (so starting from the 4th computed mode) of the satellite, and give the value of the natural frequency related to each of these 5 modes.
3.  Compute and represent the transfer functions $y_c/F_c$ and $(y_2 - y_c)/F_c$ using the full model with the coupled equations in the frequency band from 0.05 to 3 Hz. For the damping, assume a global loss factor $\eta = 0.02$. $F_c$ is a vertical force applied to the main body of the satellite. What happens to the response $y_c/F_c$ when the frequency is very low? Do you observe the same behavior for $(y_2 - y_c)/F_c$? Give a physical interpretation.
4.  Compute and represent the transfer functions $y_c/M_c$, $(y_2 + 8m * \theta_c)/M_c$ and $(\theta_a - \theta_c)/M_c$ using the full model with the coupled equations in the frequency band from 0.05 to 3 Hz. For the damping, assume a global loss factor $\eta = 0.02$. $M_c$ is a torque applied to the main body of the satellite. What do $(y_2 + 8m * \theta_c)$ and $(\theta_a - \theta_c)$ represent physically? And why are these values of interest?
5.  Project the equations of motion in the modal basis after performing truncation. How many modes should you use for the frequency band given above (Use the truncation rule explained in the course)? How would you approximate the damping with a loss factor in the modal space? Compare the transfer functions obtained when applying $M_c$ (the three transfer functions computed for subquestion 3 [sic - likely refers to subquestion 4]) using the full model and in the modal basis using truncation and an appropriate modal damping. Comment on the potential differences.

## Design of a tuned mass damper (TMD)

The solar arrays are flexible and lightly damped and can be subjected to a high number of vibration cycles during their lifetime, which could cause fatigue failure. The source of these vibrations is the position control module on the satellite, represented in this study by $F_c$ or $M_c$. You are asked to design a tuned mass damper system to reduce the risk of fatigue failure and prolong the lifetime of the solar arrays.

1.  Looking at $(y_2 - y_c)/F_c$ and $(y_2 + 8m * \theta_c)/M_c$, which global mode(s) of the system is (are) the most important to damp to preserve the solar arrays? Be specific as to which mode is important in which transfer function.
2.  As the satellite system is symmetric, we will consider two TMDs placed symmetrically (one on each solar array). The mass of each TMD should not exceed 3% of the total mass of one solar array. Our target is to damp the mode which is the most important when the excitation is given by $F_c$. Find the stiffness and damping coefficients of the two TMDs (which are assumed to be identical) which lead to optimal tuning according to Den Hartog.
3.  Compute and represent the transfer functions $(y_2 - y_c)/F_c$ and $(y_2 + 8m * \theta_c)/M_c$ when the two TMDs are attached to the satellite, and compare with the case without TMD. Do you observe equal peaks? Is the TMD efficient for both transfer functions? Explain and comment.
4.  The TMD introduces two peaks near the original natural frequency. Based on the maximum of these two peaks, can you estimate an equivalent damping for the initial system (without TMD) which would lead to the same maximum of the transfer function around this natural frequency? Give this estimate for the two transfer functions.

## Time domain computations

We assume that the main body of the satellite is subjected to a torque $M_c$ of the form given in **Figure 2**.

> **[Figure 2: Torque ($M_c$) applied to the satellite for position control]**

1.  Compute and represent the rotation of the main body $\theta_c(t)$ for a duration of 60 seconds using a convolution between the force vector and the impulse responses, after a projection in the modal basis. As the rigid body modes have a natural frequency=0 Hz, their impulse response would lead to an infinite response so you are asked not to consider them in your computation.
2.  Compute and represent $(y_2(t) + 8m * \theta_c(t))$ for a duration of 60 seconds using the same methodology. Comment on the differences between this curve and the one computed in subquestion 1, and given their physical meaning.
3.  Use the equivalent damping estimated from both transfer functions in subquestion 4 of the questions related to the TMD to approximate the effect of the two TMDs, and compute and plot again $\theta_c(t)$, and $(y_2(t) + 8m * \theta_c(t))$ (superpose the curves to the ones without the TMDs). Comment on the effect of the TMD on $\theta_c(t)$ and $(y_2(t) + 8m * \theta_c(t))$ and give a physical interpretation.

### References

[1] J. Wei, W. Liu, J. Liu, and T. Yu. Dynamic modeling and analysis of spacecraft with multiple large flexible structures. *Actuators*, 12(7)(286), 2023.
[2] J. Guyan. Reduction of stiffness and mass matrices. *AIAA journal*, 3:380–380, 1965.