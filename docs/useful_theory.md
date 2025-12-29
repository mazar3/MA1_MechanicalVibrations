# Useful Theoretical Rules & Formulas

This document summarizes the specific rules and formulas extracted from the course material ("Dynamic response computation", "Vibration damping", "Tuned vibration absorbers") required to complete the Python project.

## 1. Modal Truncation Rule (from `08 - Dynamic response computation.pdf`)

When applying the modal superposition method, you must select a sufficient number of modes to ensure accuracy. The course applies the following "Rule of Thumb":

* **Rule:** Keep all modes with natural frequencies $f_n$ up to **1.5 times** the maximum frequency of the excitation signal.
* **Formula:**
    $$f_{cutoff} \ge 1.5 \times f_{max\_excitation}$$
* **Application:**
    * Check the frequency content of your input signal (force/torque).
    * Select all modes $i$ such that $f_i \le 1.5 \times f_{max}$.
    * *Note:* The "static correction" (or residual mode) is often used to account for the missing high-frequency modes, but this rule focuses on which dynamic modes to retain.

## 2. Damping Relationship (from `15 - Vibration damping.pdf`)

The material damping is often given as a **Loss Factor** ($\eta$), while the modal superposition method in Python uses the **Viscous Damping Ratio** ($\xi$).

* **Conversion Formula:**
    $$\xi_i = \frac{\eta}{2}$$
* **Context:**
    * At resonance, for light damping, the viscous damping ratio is half the structural loss factor.
    * If a global loss factor $\eta$ is given (e.g., for the solar arrays), use $\xi = \eta/2$ for the modes associated with those structures.

## 3. Den Hartog's Optimal Tuning (from `16 - Tuned vibration absorbers.pdf`)

To design a Tuned Mass Damper (TMD) that minimizes the resonant response of an undamped primary system (Equal Peak Method), use the following classical Den Hartog formulas.

**Definitions:**
* $\mu = \frac{m_{TMD}}{M_{primary}}$ : Mass ratio
* $\nu_{opt} = \frac{\omega_{TMD}}{\omega_{primary}}$ : Optimal frequency tuning ratio
* $\xi_{opt}$ : Optimal damping ratio of the TMD

**Formulas:**
1.  **Optimal Frequency Tuning:**
    $$\nu_{opt} = \frac{1}{1 + \mu}$$
    * *Implication:* The stiffness of the TMD should be $k_{TMD} = m_{TMD} \times (\nu_{opt} \times \omega_{primary})^2$.

2.  **Optimal Damping:**
    $$\xi_{opt} = \sqrt{\frac{3\mu}{8(1 + \mu)}}$$
    * *Implication:* The damping coefficient of the TMD is $c_{TMD} = 2 \xi_{opt} m_{TMD} \omega_{TMD}$.

---
*Reference files: 08 - Dynamic response computation.pdf, 15 - Vibration damping.pdf, 16 - Tuned vibration absorbers.pdf*